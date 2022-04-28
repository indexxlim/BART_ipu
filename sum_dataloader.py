
import json
import collections
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import nltk  # Here to have a nice missing dependency error message early on
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler

import poptorch
from poptorch import Options, OutputMode

_is_torch_generator_available = False
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

class SummaryDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as json_file:
            data= json.load(json_file)
        self.contexts = []
        self.labels = []
        self.summarys = []
        for i_data in data:
            if 'label' in i_data:
                for i in i_data['label'].keys():
                    label = ''
                    for label_list in list(i_data['label'][i]['evidence'].values()):
                        label += ' '.join(label_list)+ ' '
                    self.labels.append(label[:-1])
                    self.summarys.append(i_data['label'][i]['summary'])
            for agenda in i_data['context'].keys():
                context = ''
                for line in i_data['context'][agenda]:
                    context += i_data['context'][agenda][line]
                    context += ' '
                self.contexts.append(context[:-1])


    def __getitem__(self, index):
        if len(self.summarys)>0:
            return {'context':self.contexts[index], 'label':self.labels[index], 'summary':self.summarys[index]}
        else:
            return {'context':self.contexts[index]}

    def __len__(self):
        return len(self.contexts)
    
    

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

def shift_tokens_right(input_ids, pad_token_id, eos_token_id):
    """
        Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
        decoder input: <eos><sos> tok1 tok2 … tokn
        target:<sos> tok1 tok2 … tokn <eos>
        https://github.com/huggingface/transformers/issues/7961
    """
    prev_output_tokens = input_ids.clone()
    #index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = torch.tensor(eos_token_id)
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    
    # last_index = prev_output_tokens.ne(pad_token_id).sum(dim=1)-1
    # for i, last_i in enumerate(last_index):
    #     prev_output_tokens[i][last_i] = pad_token_id
    assert pad_token_id is not None, "self.tokenizer.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    return prev_output_tokens
    

@dataclass
class SummaryCollator:
    def __init__(self, tokenizer, mapping=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        if mapping in summarization_name_mapping:
            self.text = summarization_name_mapping[mapping][0]
            self.summary = summarization_name_mapping[mapping][1]
        else:
            self.text = 'text'
            self.summary = 'summary'    

    def __call__(self, batch):
        if self.text not in batch[0] or self.summary not in batch[0]:
            raise Exception("Error: Undefined data keys")
        #sentence = [self.tokenizer.bos_token+item[self.text]+self.tokenizer.eos_token for item in batch]
        sentence = [item[self.text]+self.tokenizer.eos_token for item in batch]

        source_batch = self.tokenizer.batch_encode_plus(sentence, 
                    padding='longest', 
                    max_length=512,
                    truncation=True, 
                    return_tensors='pt')
        if self.summary in batch[0]:
            labels = [item[self.summary]+self.tokenizer.eos_token for item in batch]
            target_batch = self.tokenizer.batch_encode_plus(labels, 
                    padding='longest', 
                    max_length=512,
                    truncation=True, 
                    return_tensors='pt')
            
            labels = target_batch.input_ids.clone()
            labels[labels == self.pad_token_id] = -100

            
            return {'input_ids':source_batch.input_ids,
                     'attention_mask':source_batch.attention_mask,
                     'labels': labels,
                     'decoder_input_ids': shift_tokens_right(target_batch.input_ids, self.pad_token_id, self.eos_token_id)}
                     #'decoder_input_ids': target_batch.input_ids,
                     #'decoder_attention_mask':target_batch.attention_mask }
                   
        else:
            return {'input_ids':source_batch.input_ids,
                     'attention_mask':source_batch.attention_mask}
        
        

        
def get_dataloader(dataset, collator, batch_size=4, shuffle=False):
    data_loader = DataLoader(dataset, 
                              batch_size=batch_size, 
                              shuffle=shuffle, 
                              collate_fn=collator,
                              num_workers=2)
    
    return data_loader


class _WorkerInit:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, worker_id):
        np.random.seed((self.seed + worker_id) % np.iinfo(np.uint32).max)


def get_train_sampler(dataset, tokenizer, ipu_config, args) -> Optional[torch.utils.data.Sampler]:
    if not isinstance(dataset, collections.abc.Sized):
        return None
    generator = None
    if _is_torch_generator_available:
        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

    combined_batch_size = args.per_device_train_batch_size * ipu_config.batch_size_factor()

    # Build the sampler.
    if args.group_by_length:
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            lengths = (
                dataset[args.length_column_name]
                if args.length_column_name in train_dataset.column_names
                else None
            )
        else:
            lengths = None
        model_input_name = tokenizer.model_input_names[0] if tokenizer is not None else None

        return LengthGroupedSampler(
            combined_batch_size,
            dataset=dataset,
            lengths=lengths,
            model_input_name=model_input_name,
            generator=generator,
        )

    else:
        if args.complete_last_batch:
            num_examples = len(dataset)
            num_missing_examples = num_examples % combined_batch_size
            if num_missing_examples > 0:
                indices = torch.cat(
                    [torch.arange(num_examples), torch.randint(0, num_examples, size=(num_missing_examples,))]
                )
            return SubsetRandomSampler(indices, generator)
        return RandomSampler(dataset)        

        
def ipu_dataloader(dataset, tokenizer, ipu_config, args ,collator, shuffle=True) -> poptorch.DataLoader:
    
    opts = ipu_config.to_options()
    
    if isinstance(dataset, torch.utils.data.IterableDataset):
        return poptorch.DataLoader(
            opts,
            dataset,
            batch_size=args.train_batch_size,
            collate_fn=collator,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.dataloader_pin_memory,
            **poptorch_specific_kwargs,
        )

    
    should_drop_last = not hasattr(collator, "__wrapped__") and not args.complete_last_batch
    poptorch_specific_kwargs = {
        "drop_last": should_drop_last,  # Not dropping last will end up causing NaN during training if the combined batch size does not divide the number of steps
        "auto_distributed_partitioning": not isinstance(dataset, torch.utils.data.IterableDataset),
        "mode": args.dataloader_mode,
        "worker_init_fn": _WorkerInit(123),
    }    
    
    
    
    train_sampler = get_train_sampler(dataset, tokenizer, ipu_config, args)

    
    combined_batch_size = args.per_device_train_batch_size * ipu_config.batch_size_factor()
    rebatched_worker_size = (
        2 * (combined_batch_size // args.dataloader_num_workers)
        if args.dataloader_num_workers
        else combined_batch_size
    )

    return poptorch.DataLoader(
        opts,
        dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler if shuffle else None,
        collate_fn=collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
        rebatched_worker_size=rebatched_worker_size,
        **poptorch_specific_kwargs,
    )
