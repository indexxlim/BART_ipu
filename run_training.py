import logging
import os
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import math, time

import yaml
from easydict import EasyDict
import datasets
import numpy as np
from datasets import load_dataset, load_metric

import torch
from torch import nn, optim
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version as tf_check_min_version
from transformers.utils.versions import require_version

import poptorch
from optimum.graphcore import IPUSeq2SeqTrainer#, IPUConfig
from optimum.graphcore import IPUSeq2SeqTrainingArguments as Seq2SeqTrainingArguments
from optimum.graphcore.modeling_utils import to_pipelined

from packaging import version

from sum_dataloader import SummaryCollator, get_dataloader, get_train_sampler, ipu_dataloader
from ipu_train import train
from model.pipeline_bart import PipelinedBartForConditionalGeneration
from model.ipu_configuration import IPUConfig


__version__ = "0.2.4.dev"

def check_min_version(min_version):
    if version.parse(__version__) < version.parse(min_version):
        if "dev" in min_version:
            error_message = "This example requires a source install from HuggingFace Optimum-Graphcore"
        else:
            error_message = f"This example requires a minimum version of {min_version},"
        error_message += f" but the version found is {__version__}.\n"
        raise ImportError(error_message)

def main():
    # Will error if the minimal version of Transformers is not installed. Remove at your own risks.
    #tf_check_min_version("4.19.0.dev0")

    # Will error if the minimal version of Optimum Graphcore is not installed. Remove at your own risks.
    check_min_version("0.2.4.dev")

    require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

    logger = logging.getLogger(__name__)

    # A list of all multilingual tokenizer which require lang attribute.



    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = -1 # training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters  ")
    
    argus = {}
    with open('configuration/summrization.yaml') as f:
        hparams = yaml.load_all(f, Loader=yaml.FullLoader)
        for argu in hparams:
            argus[list(argu.keys())[0]]=list(argu.values())[0]

    model_args, data_args, training_args = argus['ModelArguments'], argus['DataTrainingArguments'], argus['IPUSeq2SeqTrainingArguments']
    model_args, data_args = EasyDict(model_args), EasyDict(data_args)
    training_args = Seq2SeqTrainingArguments(**training_args)
    
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
            
            
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    ipu_config = IPUConfig.from_pretrained(
        training_args.ipu_config_name if training_args.ipu_config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model.resize_token_embeddings(len(tokenizer))

    #model = to_pipelined(model, ipu_config, force=False)
    model = PipelinedBartForConditionalGeneration.from_transformers(model, ipu_config)
    model.parallelize()
    if not training_args.fp32:
        model = model.half()
    



    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )


    
    
    summarization_name_mapping = {
        "amazon_reviews_multi": ("review_body", "review_title"),
        "big_patent": ("description", "abstract"),
        "cnn_dailymail": ("article", "highlights"),
        "orange_sum": ("text", "summary"),
        "pn_summary": ("article", "summary"),
        "psc": ("extract_text", "summary_text"),
        "samsum": ("dialogue", "summary"),
        "thaisum": ("body", "summary"),
        "xglue": ("news_body", "news_title"),
        "xsum": ("document", "summary"),
        "wiki_summary": ("article", "highlights"),
    }
    
    #Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        raise

    collator = SummaryCollator(tokenizer, 'cnn_dailymail')
    dataloader = get_dataloader(raw_datasets["train"], collator)

    ipu_train_dataloader = ipu_dataloader(raw_datasets["train"], tokenizer, ipu_config, training_args, collator)

    train(model, ipu_config, training_args, ipu_train_dataloader, logger)
            
if __name__ == "__main__":
    main()