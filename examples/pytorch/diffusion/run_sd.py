import argparse
from datasets import load_dataset
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from diffusers import UNet2DConditionModel, DDPMScheduler
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel
from transformers import AutoTokenizer
from tqdm import tqdm


def pre_batch(batch, device):
    for k, v in batch.items():
        if isinstance(v, dict):
            pre_batch(v, device)
        elif isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class UNetTrainer:

    def __init__(self, args):
        self._config = args
        self._vae = self._load_vae(args)
        self._noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self._text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        self._text_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
        self._unet = UNetTrainer._build_unet()
        self._train_dataloader = self._init_train_dataloader(args)
        self._device = "cuda"

    @staticmethod
    def _build_unet():
        # same config with stable-diffusion-2
        return UNet2DConditionModel(
            sample_size=96,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(320, 640, 1280, 1280),
            attention_head_dim=(5, 10, 20, 20),
            cross_attention_dim=1024,
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D"
            ),
            use_linear_projection=True,
        )

    def _load_vae(self, args):
        if args.vae_model_name_or_path is not None:
            return AutoencoderKL.from_pretrained(args.vae_model_name_or_path)
        return AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")

    def _init_train_dataloader(self, args):
        image_size = args.resolution
        # TODO: change to LAION-400M-filtered
        # dataset_name = "/home/luoxiaoc/Projects/customize_sd/smithsonian_butterflies_subset"
        dataset_name = "huggan/smithsonian_butterflies_subset"
        dataset = load_dataset(dataset_name, split="train")
        preprocess = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def transform(examples):
            images = [preprocess(image.convert("RGB")) for image in examples["image"]]
            return {"images": images, "name": examples["name"]}

        dataset.set_transform(transform)

        train_batch_size = args.train_batch_size
        return torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, shuffle=True)


    def train_loop(self):
        optimizer = torch.optim.AdamW(self._unet.parameters(), lr=self._config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=(len(self._train_dataloader) * self._config.num_train_epochs),
        )

        self._vae.requires_grad_(False)
        self._vae.to(self._device)
        self._text_encoder.requires_grad_(False)
        self._text_encoder.to(self._device)
        self._unet.to(self._device)
        self._unet.train()

        global_step = 0

        for epoch in range(self._config.num_train_epochs):
            progress_bar = tqdm(total=len(self._train_dataloader), disable=False)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(self._train_dataloader):
                text_input_ids = self._text_tokenizer(batch["name"], truncation=True,
                    padding="max_length", max_length=self._text_tokenizer.model_max_length, return_tensors="pt",).input_ids
                pre_batch(batch, self._device)

                # Convert images to latent space
                latents = self._vae.encode(batch["images"]).latent_dist.sample()
                latents = latents * self._vae.config.scaling_factor

                # Sample noise to add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each latents
                timesteps = torch.randint(
                    0, self._noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device,
                    dtype=torch.int64
                )
                timesteps = timesteps.long()

                # Add noise to latents (this is the forward diffusion process)
                noisy_latents = self._noise_scheduler.add_noise(latents, noise, timesteps)

                # conditioning
                encoder_hidden_states = self._text_encoder(text_input_ids.to(self._device))[0]

                # Predict the noise residual
                noise_pred = self._unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self._unet.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                global_step += 1

def parse_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae model",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Epochs number for the training dataloader."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    unet_trainer = UNetTrainer(args)
    unet_trainer.train_loop()
