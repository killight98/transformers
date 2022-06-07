
import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint
from sentence_transformers import SentenceTransformer, models

from dataset import DenoisingAutoEncoderDataset, read_corpus_for_pretrain, DATA_DIR
from model import TSDAE


logger = logging.getLogger(__name__)

RESULT_DIR = '-tsdae'

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """

    train_file: str = field(default=None, metadata={"help": "The input training data file (a text file)."})
    delete_ratio: float = field(
        default=0.6, metadata={"help": "Ratio of tokens to delete"}
    )

    def __post_init__(self):
        if self.train_file is None:
            raise ValueError("Need a training file.")
        else:
            extension = self.train_file.split(".")[-1]
            if extension not in ["txt"]:
                raise ValueError("`train_file` should be a txt file.")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization."
            )
        },
    )
    pooling_mode: Optional[str] = field(
        default="cls",
        metadata={"help": "Pass a pooling_mode from the list: " + ", ".join(['mean', 'max', 'cls'])},
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    def __post_init__(self):
        assert self.pooling_mode in ['mean', 'max', 'cls'], self.pooling_mode.metadata
        assert self.model_name_or_path is not None, "Please provide model_name_or_path unless training from scratch"


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.output_dir += RESULT_DIR
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Prepare the dataset
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    train_sentences = read_corpus_for_pretrain(data_args.train_file)
    train_dataset = DenoisingAutoEncoderDataset(train_sentences)

    # Prepare the sentencetransformer modules and tsdae models
    word_embedding_model = models.Transformer(model_args.model_name_or_path)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), model_args.pooling_mode)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    # Follow the settings in the original TSDAE implementation
    collate_fn = model.smart_batching_collate
    tsade = TSDAE(model, decoder_name_or_path=model_args.model_name_or_path, tie_encoder_decoder=True)

    # Initialize our Trainer
    trainer = Trainer(
        model=tsade,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=word_embedding_model.tokenizer,
        data_collator=collate_fn
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {checkpoint}."
            )

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()


if __name__ == "__main__":
    main()
