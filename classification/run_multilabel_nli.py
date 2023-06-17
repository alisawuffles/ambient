""" Finetuning the library models for sequence classification on GLUE."""
"""https://github.com/huggingface/transformers/blob/v4.9.0/examples/pytorch/text-classification/run_glue.py"""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils.utils import ensure_dir, sigmoid
# from torch.nn.functional import sigmoid
os.environ["WANDB_DISABLED"] = "True"

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset (useful for naming things)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Truncate the number of training examples to this value"},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of training examples to this value"},
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of training examples to this value"},
    )
    set_prediction: bool = field(
        default=True, metadata={'help': 'Whether to predict single label or set of labels'}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "A json file for training data."})
    validation_file: Optional[str] = field(default=None, metadata={"help": "A json file for validation data."})
    test_file: Optional[str] = field(default=None, metadata={"help": "A json file for test data."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                             "Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                         "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")

    # Set seed before initializing model
    set_seed(training_args.seed)

    data_files = {}
    if training_args.do_train:
        data_files['train'] = data_args.train_file
    if training_args.do_eval:
        data_files['validation'] = data_args.validation_file
    if training_args.do_predict:
        data_files['test'] = data_args.test_file

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    sentence1_key, sentence2_key, label_key = 'premise', 'hypothesis', 'gold'
    
    def filter_function(example):
        return example[label_key] != '-'
    
    raw_datasets = raw_datasets.filter(filter_function) 

    # labels
    label_list = ['contradiction', 'entailment', 'neutral']
    label_list.sort()

    config = RobertaConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        num_labels=len(label_list)
    )
    tokenizer = RobertaTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )
    model = RobertaForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    label_to_id = {v: i for i, v in enumerate(label_list)}
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in label_to_id.items()}
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        args = (examples[sentence1_key], examples[sentence2_key])
        result = tokenizer(*args, padding="max_length", max_length=max_seq_length, truncation=True)
        if not training_args.do_predict:
            result['label'] = []
            for ls in examples[label_key]:
                labels = [0]*len(label_list)
                ls = ls.split(', ')
                for l in ls:
                    id = label_to_id[l]
                    labels[id] = 1
                result['label'].append(labels)
        return result
    
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # log some random samples from the training set
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if data_args.set_prediction:
            preds = sigmoid(preds)
            preds = np.where(preds > 0.5, 1, 0)
            return {
                'exact_accuracy': np.all(preds == p.label_ids, axis=1).astype(np.float32).mean().item(),    # row-wise
                'interpretation_accuracy': (preds == p.label_ids).astype(np.float32).mean().item(), # element-wise
            }
        else:
            preds = np.argmax(preds, axis=1)
            label_ids = np.argmax(p.label_ids, axis=1)  # TODO: this will randomly pick one of the correct labels, if test set has multiple labels
            return {'accuracy': (preds == label_ids).astype(np.float32).mean().item()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # train!
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics(data_args.dataset_name, metrics)
        trainer.save_metrics(data_args.dataset_name, metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_dataset = predict_dataset.remove_columns(label_key)
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions #these are logits
        if data_args.set_prediction:
            predictions = sigmoid(predictions)
            predictions = np.where(predictions > 0.5, 1, 0)
            output_dir = os.path.join(training_args.output_dir, 'set_predictions')
        else:
            predictions = np.argmax(predictions, axis=1)
            output_dir = os.path.join(training_args.output_dir, 'predictions')

        ensure_dir(output_dir)
        output_predict_file = os.path.join(output_dir, f'{data_args.dataset_name}_predictions.txt')

        with open(output_predict_file, "w") as writer:
            logger.info(f"***** Predict results *****")
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
                if data_args.set_prediction:
                    items = [label_list[i] for i in range(len(item)) if item[i] == 1]
                    writer.write(f"{index}\t{', '.join(items)}\n")
                else:
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")
        # TODO: we might be able to check if predictions object has metrics attached and write evaluation file


if __name__ == "__main__":
    main()