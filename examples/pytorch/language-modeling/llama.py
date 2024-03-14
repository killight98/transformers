import argparse
import os
import time

import logging
import math

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers.trainer_pt_utils import distributed_concat
from transformers.utils.profiler import ProfilerConfig, ProfilerWrapper
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)

from peft import LoraConfig, TaskType, get_peft_model

XPU_SUPPORT = True
try:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch
except ImportError:
    XPU_SUPPORT = False

from dataclasses import dataclass
from typing import Optional

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Training on XPU or CUDA")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        )
    parser.add_argument(
        "--device_type",
        type=str,
        default="cuda",
        help="The device to run on"
        )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="The length for model input"
        )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="The length for model input"
        )
    parser.add_argument(
        "--num_train_epoch",
        type=int,
        default=1,
        )
    parser.add_argument(
        "--profile_step",
        type=int,
        default=None,
        )
    return parser.parse_args()

class Llama:
    def __init__(self, args):
        """Llama init to set config parameters"""
        # model parameters
        self.model_name_or_path = args.model_name_or_path
        self.dataset_name = args.dataset_name
        self.max_length = args.max_length
        self.batch_size = args.per_device_train_batch_size
        self.num_epochs = args.num_train_epoch
        self.profile_step = args.profile_step

        # device config
        device = args.device_type
        device_ops = ('cuda', 'xpu')
        if device not in device_ops:
            raise ValueError("Valid device is cuda and xpu")
        if device == 'xpu' and not XPU_SUPPORT:
            raise ValueError("xpu device is not support by the python library")
        self.device = device
        self.use_ddp, self.rank, self.local_rank, self.world_size = self._setup_device()
        self._device = f'{device}:{self.rank}'
        logger.info(self._device)
        seed = 42
        set_seed(seed)

    def process_dataset(self):
        # dataset config
        dataset_name = self.dataset_name
        max_length = self.max_length
        text_column = 'text'
        target_column = 'output'
        batch_size = self.batch_size
        train_split = 90

        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
        )
        tokenizer.pad_token_id = tokenizer.bos_token_id
        tokenizer.padding_side = 'left'

        def preprocess_function(examples):
            model_inputs = tokenizer(examples[text_column],
                                padding="max_length",
                                max_length=max_length,
                                truncation=True)
            target_tokens = tokenizer(examples[target_column],
                                padding="max_length",
                                max_length=max_length,
                                truncation=True)
            model_inputs["labels"] = target_tokens['input_ids']
            return model_inputs

        raw_train_dataset = load_dataset(dataset_name, split=f'train[:{train_split}%]')
        train_dataset = raw_train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=raw_train_dataset.column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
        raw_eval_dataset = load_dataset(dataset_name, split=f'train[{train_split}%:]')
        eval_dataset = raw_eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=raw_eval_dataset.column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
        if self.rank == 0:
            logger.info(f"train: {train_dataset}, {train_dataset.shape} \n")
            logger.info(f"eval: {eval_dataset}, {eval_dataset.shape} \n")

        extra_args = {}
        if self.use_ddp:
            extra_args["sampler"] = DistributedSampler(train_dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True)
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=default_data_collator,
            batch_size=batch_size,
            pin_memory=True,
            **extra_args
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=default_data_collator,
            batch_size=batch_size,
            pin_memory=True
        )
        if self.rank == 0:
            logger.info(next(iter(train_dataloader))['input_ids'].shape)

        return train_dataloader, eval_dataloader

    def process_model(self):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8, lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype="auto",
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        logger.info(model)
        return model

    def train(self):
        # parameter
        lr = 3e-5
        num_epochs = self.num_epochs

        # optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        # lr scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(self.train_dataloader) * num_epochs),
        )
        self.model = self.model.to(self._device)
        self.model.train()
        if self.device == 'xpu':
            self.model, optimizer = ipex.optimize(self.model, optimizer=optimizer, inplace=True)

        if self.use_ddp and torch.distributed.get_world_size() > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=True,
                find_unused_parameters=True
            )

        # load items to device
        def pre_batch(batch):
            for k, v in batch.items():
                if isinstance(v, dict):
                    pre_batch(v)
                elif isinstance(v, torch.Tensor):
                    batch[k] = v.to(self._device)

        profiler_config = ProfilerConfig(active=self.profile_step) if self.profile_step is not None else None
        with ProfilerWrapper(self.device, profiler_config) as profiler:
            for epoch in range(num_epochs):
                self.model.train()
                total_loss = 0
                train_iter = tqdm(self.train_dataloader, disable=self.rank > 0)
                for step, batch in enumerate(train_iter):
                    pre_batch(batch)
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    total_loss += loss.detach().float()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    profiler.step()

                    last_lr = lr_scheduler.get_last_lr()[0]
                    postfix_dic = {"loss": loss.item(),
                                   "lr": last_lr}
                    train_iter.set_postfix(postfix_dic)

                    if self.profile_step is not None and step > self.profile_step:
                        break

                if self.profile_step is not None:
                    continue

                train_epoch_loss = total_loss / len(self.train_dataloader)
                train_ppl = torch.exp(train_epoch_loss)
                logger.info(f"Training: {epoch}: {train_ppl} {train_epoch_loss}")

                self.model.eval()
                losses = []
                eval_iter = tqdm(self.eval_dataloader, disable=self.rank > 0)
                for step, batch in enumerate(eval_iter):
                    with torch.no_grad():
                        pre_batch(batch)
                        outputs = self.model(**batch)
                    loss = outputs.loss
                    losses.extend(distributed_concat(loss))

                try:
                    eval_loss = torch.mean(torch.Tensor(losses))
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")

                logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        torch.distributed.barrier()

    def run(self):
        self.train_dataloader, self.eval_dataloader = self.process_dataset()
        self.model = self.process_model()
        self.train()

    def _setup_device(self):
        if self.device == 'cuda':
            device_count = torch.cuda.device_count()
            assert device_count >= 1, "Cannot find cuda"
            if "LOCAL_RANK" in os.environ:
                local_rank = int(os.environ["LOCAL_RANK"])
                rank = int(os.environ['RANK'])
                world_size = int(os.environ['WORLD_SIZE'])
                assert local_rank < device_count, f"Not enough devices for local rank {local_rank}"
                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group(backend="nccl")
                torch.cuda.set_device(local_rank)
                return True, rank, local_rank, world_size
        if self.device == 'xpu':
            device_count = torch.xpu.device_count()
            assert device_count >= 1, "Cannot find xpu"
            if "PMI_SIZE" in os.environ:
                rank = int(os.environ["PMI_RANK"])
                world_size = int(os.environ["PMI_SIZE"])
                assert rank < device_count, f"Not enough device for local rank {rank}"
                # re-config env
                os.environ['RANK'] = str(rank)
                os.environ['WORLD_SIZE'] = str(world_size)
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                torch.distributed.init_process_group(backend="ccl")
                return True, rank, rank, world_size
        return False, 0, -1, -1

def main():
    args = parse_args()
    llama = Llama(args)
    llama.run()

if __name__=="__main__":
    main()
