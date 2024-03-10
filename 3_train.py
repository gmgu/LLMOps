# torchrun --nproc_per_node 2 train.py
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, default_data_collator, TrainingArguments, HfArgumentParser
from transformers import set_seed, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_from_disk, load_metric
from dataclasses import dataclass, field
from typing import Optional
from transformers import DataCollatorForLanguageModeling
import evaluate
import math
import os


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(default='solar-25M')
    model_type: Optional[str] = field(default=None)
    config_overrides: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default='solar-25M')
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError("--config_overrides can't be used in combination with --config_name or --model_name_or_path")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default='lm_datasets')
    validation_file: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    block_size: Optional[int] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)
    keep_linebreaks: bool = field(default=True)

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


metric = evaluate.load("accuracy")
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels[:, 1:].reshape(-1)  # A B C D E -> B C D E
    preds = preds[:, :-1].reshape(-1)  # b c d e f -> b c d e
    return metric.compute(predictions=preds, references=labels)


## main
ds_config = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
      "stage": 3,
      "allgather_partitions": True,
      "allgather_bucket_size": 2e8,
      "reduce_scatter": True,
      "reduce_bucket_size": 2e8,
      "overlap_comm": True,
      "contiguous_gradients": True
    },
    "gradient_accumulation_steps": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
#parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
#model_args, data_args, training_args = parser.parse_args_into_dataclasses()
model_args = ModelArguments()
data_args = DataTrainingArguments()
training_args = TrainingArguments(output_dir='output_dir',
                                  num_train_epochs=2,
                                  logging_steps=1,
                                  learning_rate=1e-3,
                                  per_device_train_batch_size=4,
                                  gradient_accumulation_steps=128,
                                  deepspeed=ds_config,
                                  fp16=True)

model_save_location = model_args.model_name_or_path

set_seed(training_args.seed)

config = AutoConfig.from_pretrained(model_save_location, cache_dir=model_args.cache_dir, revision=model_args.model_revision)
model = AutoModelForCausalLM.from_pretrained(model_save_location, config=config, cache_dir=model_args.cache_dir, revision=model_args.model_revision)
tokenizer = AutoTokenizer.from_pretrained(model_save_location, use_fast=True, cache_dir=model_args.cache_dir, revision=model_args.model_revision)
model.resize_token_embeddings(len(tokenizer))

lm_datasets = load_from_disk(data_args.train_file)

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=lm_datasets,
                  tokenizer=tokenizer,
                  data_collator=default_data_collator,
                  compute_metrics=compute_metrics,
                  preprocess_logits_for_metrics=preprocess_logits_for_metrics)

train_result = trainer.train()
trainer.save_model()

