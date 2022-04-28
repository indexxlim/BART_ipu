import logging
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import math, time

import yaml
from easydict import EasyDict
import datasets
import nltk  # Here to have a nice missing dependency error message early on
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
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import get_scheduler
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version as tf_check_min_version
from transformers.utils import is_offline_mode
from transformers.utils.versions import require_version

import poptorch
from poptorch import DataLoaderMode, PoplarExecutor
from poptorch.optim import LAMB, AdamW

from optimum.graphcore import IPUConfig, IPUSeq2SeqTrainer
from optimum.graphcore import IPUSeq2SeqTrainingArguments as Seq2SeqTrainingArguments
from optimum.graphcore.modeling_utils import to_pipelined

from packaging import version

from sum_dataloader import SummaryCollator, get_dataloader, get_train_sampler, ipu_dataloader


__version__ = "0.2.4.dev"

def check_min_version(min_version):
    if version.parse(__version__) < version.parse(min_version):
        if "dev" in min_version:
            error_message = "This example requires a source install from HuggingFace Optimum-Graphcore"
        else:
            error_message = f"This example requires a minimum version of {min_version},"
        error_message += f" but the version found is {__version__}.\n"
        raise ImportError(error_message)
        
        
def get_optimizer_scheduler(model, ipu_config, args, ipu_train_dataloader):
    '''
        Get optimizer and scheduler
    '''
    
    #Count number of training step
    train_dataset_is_sized = True#isinstance(train_dataset, collections.abc.Sized)
    total_train_batch_size = args.per_device_train_batch_size * ipu_config.batch_size_factor()
    if train_dataset_is_sized:
        # No need to divide by the number of gradient accumulation steps as poptorch already accounts for that.
        # num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_update_steps_per_epoch = len(ipu_train_dataloader)
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                args.max_steps % num_update_steps_per_epoch > 0
            )

            # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
            # the best we can do.
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)
            num_train_samples = len(ipu_train_dataloader) * args.num_train_epochs
    else:
        # see __init__. max_steps is set when the dataset has no __len__
        max_steps = args.max_steps
        # Setting a very large number of epochs so we go as many times as necessary over the iterator.
        num_train_epochs = sys.maxsize
        num_update_steps_per_epoch = max_steps
        num_train_samples = args.max_steps * total_train_batch_size



    #Get optimizer

    #if optimizer is None:
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    if args.lamb or args.lamb_no_bias_correction:
        optimizer_cls = LAMB
        optimizer_kwargs = {
            "max_weight_norm": None,
            "bias_correction": not args.lamb_no_bias_correction,
            "eps": args.adam_epsilon,
        }
    else:
        optimizer_cls = AdamW
        optimizer_kwargs = {
            # TODO: disabled max_grad_norm because it make things fail, fix it.
            #  "max_grad_norm": self.args.max_grad_norm,
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
            "bias_correction": False,
        }

    first_order_type = torch.float16 if ipu_config.enable_half_first_order_momentum else torch.float32
    optimizer_kwargs["lr"] = args.learning_rate
    optimizer_kwargs["loss_scaling"] = args.loss_scaling
    optimizer_kwargs["accum_type"] = first_order_type
    optimizer_kwargs["first_order_momentum_accum_type"] = first_order_type
    optimizer_kwargs["second_order_momentum_accum_type"] = torch.float32

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    if args.lamb or args.lamb_no_bias_correction:
        optimizer.variable_attrs.markAsConstant("max_weight_norm")

    optimizer.variable_attrs.markAsConstant("weight_decay")
    
    
    #Get scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.get_warmup_steps(max_steps), #num_training_steps
        num_training_steps=max_steps,
    )
    optimizer._step_count = 1
    
    
    return optimizer, lr_scheduler


def wrap_model(model: Union[PreTrainedModel, PoplarExecutor], opts, training=True) -> PoplarExecutor:
        """
        Wraps a model for poptorch, either for training or for inference.
        Args:
            model (`~transformers.modeling_utils.PreTrainedModel` or `PoplarExecutor`): the model to wrap
            training (`bool`, *optional*, defaults to `True`): whether to wrap the model for training or not.
        Returns:
            The wrapped model.
        """
        wrapped = None
        if isinstance(model, PoplarExecutor):
            wrapped = model
        elif training:
            training_model = poptorch.trainingModel(
                model.train(), options=self.opts, optimizer=self.optimizer
            )
            wrapped = training_model
        else:
            inference_model = poptorch.inferenceModel(model.eval(), options=self.eval_opts)
            wrapped = inference_model

        # Attaching to device when the model that is being access was already compiled but detached from previous loop.
        if wrapped.isCompiled() and not wrapped.isAttachedToDevice():
            wrapped.attachToDevice()
        return wrapped

#def training_step(model):
    

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
   

    model = to_pipelined(model, ipu_config, force=False)
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

    ipu_train_dataloader = ipu_dataloader(raw_datasets["train"], tokenizer, ipu_config, training_args, collator, shuffle=False)
        
    '''
    #Training
    '''

    optimizer, lr_scheduler = get_optimizer_scheduler(model, ipu_config, training_args, ipu_train_dataloader)

    if optimizer is not None and not isinstance(optimizer, poptorch.optim.Optimizer):
    #optimizer = self._pytorch_optimizer_to_poptorch(self.optimizer, model, self.model)
        raise Exception('Error : convert to poptorch optimzier')

    opts = ipu_config.to_options()
    training_model = poptorch.trainingModel(
        model.train(), options=opts, optimizer=optimizer
    )
    training_model = wrap_model(training_model, opts)
    
    sample_data = next(iter(ipu_train_dataloader))

    if training_model.isCompiled():
        pass
    else:
        logger.info("Compiling Model...")
        start_compile = time.perf_counter()

        sample_batch = next(iter(ipu_train_dataloader))
        
        if isinstance(sample_batch, tuple):
            training_model.compile(*dict(sample_data))
        else:
            training_model.compile(**dict(sample_data))
        duration_compilation = time.perf_counter() - start_compile
        logger.info(f"Compiled/Loaded model in {duration_compilation} secs")



    loss = 0
    for step, inputs in enumerate(ipu_train_dataloader):
        

        loss_step = training_model(**inputs)
        logger.info(f"loss is {loss_step}")
        loss += loss_step

        optimizer_was_run = True

        if optimizer_was_run:
            lr_scheduler.step()
            training_model.setOptimizer(optimizer)
            
if __name__ == "__main__":
    main()