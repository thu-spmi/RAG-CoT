# coding=utf-8
# Copyright 2023, Tasi Inc. team.
"""RAG model implementation."""

import argparse
import copy
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import math
import time
import gc
import ssl
from datetime import datetime
from argparse import Namespace

import torch
from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from torch import nn
import random
import numpy as np
#import faiss
import json

import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, get_scheduler, BitsAndBytesConfig, set_seed
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
import contextlib

from tqdm import tqdm
import logging

IGNORE_INDEX = -100

logger = logging.getLogger(__name__)

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))

def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1

def is_distributed():
    return dist.is_available() and dist.is_initialized()

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

def right_pad_to_max_length(input_tokens, max_length, pad_token_id):
    input_tokens = input_tokens[:max_length]
    padded_tokens = input_tokens + [pad_token_id] * (max_length - len(input_tokens))
    return padded_tokens

def left_pad_to_max_length(input_tokens, max_length, pad_token_id):
    input_tokens = input_tokens[:max_length]
    padded_tokens = [pad_token_id] * (max_length - len(input_tokens)) + input_tokens
    return padded_tokens

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            logger.info(f'trainable: {name} {param.shape} {param.dtype}')
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    lora_lr=5e-4,
    no_decay_name_list=["bias", "LayerNorm.weight"],
    lora_name_list=["lora_A", "lora_B"],
):
    optimizer_grouped_parameters_ = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n
                                                for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
            "lr":
            lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]
    optimizer_grouped_parameters = []
    for i, ogp in enumerate(optimizer_grouped_parameters_):
        logger.info(f'optimizer_group[{i}]: {len(ogp["params"])}')
        if ogp['params']: optimizer_grouped_parameters.append(ogp)
    return optimizer_grouped_parameters

def sample_dataset(dataset, sample_prob_or_size):
    result = []
    if sample_prob_or_size < 1: 
        sample_prob_or_size = int(sample_prob_or_size * len(dataset))
    samples = np.random.randint(len(dataset), size = sample_prob_or_size)
    for k in samples:
        result.append(dataset[k])
    return result

def save_hf_format(model, tokenizer, output_dir):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

class CatFidDataset(Dataset):
    def __init__(self, pathname, tokenizer, args, eval_mode=False):
        super(CatFidDataset, self).__init__()
        self.n_docs = args.n_docs
        self.source_max_len = args.source_max_len
        self.target_max_len = args.target_max_len
        self.instruction_template = args.instruction_template
        if not eval_mode:
            self.random_pick = args.random_pick if hasattr(args, 'random_pick') else 0
            self.random_shuffle = args.random_shuffle if hasattr(args, 'random_shuffle') else False
        else:
            self.random_pick = 0
            self.random_shuffle = False
        self.args = args
        model_name_or_path = args.generator_model_name
        self.tokenizer = tokenizer
        self.args = args
        self.samples = []
        logger.info(f'random_pick: {self.random_pick}')
        logger.info(f'random_shuffle: {self.random_shuffle}')
        with open(pathname, 'r') as f:
            print(f'processing {pathname} ...')
            for ln in tqdm(f):
                ln = ln.strip()
                record = json.loads(ln)
                question = record['question']
                answer = record['answer']
                facts = record['match_info'] + record['random_info']
                self.random_shuffle = True
                if self.random_shuffle:
                    random.shuffle(facts)
                concat_facts = '\n'.join(facts)
                
                # concate rag inputs: (query, context)
                rag_source = self.tokenizer.bos_token + self.instruction_template.format_map({'question': question, 'input': concat_facts})
                rag_target = answer + self.tokenizer.eos_token

                tokenized_source = self.tokenizer(
                    rag_source,
                    max_length=self.source_max_len * self.n_docs,  # source_max_len is per fact
                    truncation=True,
                    add_special_tokens=False,
                )
                
                tokenized_target = self.tokenizer(
                    rag_target,
                    max_length=self.target_max_len,
                    truncation=True,
                    add_special_tokens=False,
                )
                
                # Build the input and labels for causal LM
                input_ids = torch.tensor(tokenized_source['input_ids'] + tokenized_target['input_ids'])
                labels = torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source['input_ids']))] + copy.deepcopy(tokenized_target['input_ids']))
                self.samples.append({'input_ids': input_ids, 'labels': labels})

        self.num_samples = len(self.samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx = idx % self.num_samples
        return self.samples[idx]



def get_tokenizer(args):
    model_name_or_path = args.generator_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if 'gpt' in model_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif 'llama' in model_name_or_path:
        tokenizer.add_special_tokens({'pad_token': tokenizer.decode([0])})
    elif not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def get_model(args):
    model_name_or_path = args.generator_model_name
    peft_model_path = args.generator_peft_model_name        
    device_map = None if should_distribute() else "auto"
    if not args.flash_attention:
        if args.use_bf16:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype = torch.bfloat16, low_cpu_mem_usage = True, device_map = device_map)        
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype = torch.float16, low_cpu_mem_usage = True, device_map = device_map)        
    else:
        logger.info('Use flash attention v2')
        if args.use_bf16:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype = torch.bfloat16, low_cpu_mem_usage = True, attn_implementation="flash_attention_2", device_map = device_map)        
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype = torch.float16, low_cpu_mem_usage = True, attn_implementation="flash_attention_2", device_map = device_map)        

    if args.generator_use_peft:
        if peft_model_path is None:
            logger.info('preparing Lora ...')
            lora_target = args.lora_target
            lora_r = args.lora_r
            lora_alpha = args.lora_alpha
            lora_dropout = args.lora_dropout
            if isinstance(lora_target, list):
                modules = lora_target
            else:
                modules = [item.strip() for item in lora_target.split(",")]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
        else:
            logger.info(f'Loading Lora from {peft_model_path} ...')
            model = PeftModel.from_pretrained(model, peft_model_path, is_trainable = True)

    if not args.use_bf16:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    if args.use_gradient_checkpoint_generator:
        # from functools import partial
        # notfailing_checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
        # torch.utils.checkpoint.checkpoint = notfailing_checkpoint
        # model.gradient_checkpointing_enable()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
 
    return model

def save_model(model, tokenizer, save_path, ckpt_id, save_latest = None):
    os.makedirs(save_path, exist_ok=True)
    if save_latest:
        with open(os.path.join(save_path, save_latest), 'w') as fd:
            fd.write(ckpt_id)
    output_dir = os.path.join(save_path, ckpt_id)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f'saving generator model checkpoint to {output_dir}')
    # save_hf_format(model, tokenizer, output_dir)
    if tokenizer: tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=False)

def check_zero_and_negative(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if torch.any(value == 0):
                print(f"Tensor '{key}' contains zero values.")
            if torch.any(value < 0):
                print(f"Tensor '{key}' contains negative values.")

def evaluation(model, eval_dataloader):
    torch.cuda.empty_cache()
    model.eval()
    total_loss = 0
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            batch = to_device(batch, model.device)
            outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
    total_loss /= (step + 1)
    model.train()
    return total_loss
             
logger = logging.getLogger(__name__)

"""
params_file
{
    "generator_model_name": "mistralai/Mistral-7B-v0.1",
    "generator_use_peft": true,
    "use_oss": true,
    "use_bf16": true,
    "generator_peft_model_name": null,
    "n_docs": 5,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 3,
    "eval_size": 200,
    "learning_rate": 1e-5,
    "lora_lr": 1e-5,
    "weight_decay": 0.01,
    "lora_target": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    "lora_r": 8,
    "lora_alpha": 8.0,
    "lora_dropout": 0.0,
    "lr_scheduler_type": "cosine",
    "num_warmup_steps": 100,
    "num_train_epochs": 5,
    "save_steps": 5000,
    "save_dir": "./NQ_CATFID",
    "max_grad_norm": 0,
    "source_max_len": 512,
    "target_max_len": 256,
    "instruction_template": "[INST]Give a short answer to the Question based on relevant information given in Input.\nInput:{input}\nQuestion: {question}\n[/INST]"
}
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--params_file", type=str,
                        help="JSON configuration file")    
    parser.add_argument("--train_dataset", type=str,
                        help="train dataset JSONL file",
                        default='nq_train.miniX5.jsonl')
    parser.add_argument("--eval_dataset", type=str,
                        help="eval dataset JSONL file",
                        default='nq_dev.miniX5.jsonl')
    parser.add_argument('--use_gradient_checkpoint_generator', action='store_true',
                        help='Use gradient checkpointing in generator')                        
    parser.add_argument('--flash_attention', action='store_true',
                        help='Use flash attention')
    parser.add_argument("--seed", type=int,
                        default=1234, help="A seed for reproducible training.")
    parser.add_argument('--load_checkpoint', type=int,
                        default=0, help='Use model checkpoint at specified step')                        
    parser.add_argument("--save_latest", type=str,
                        help="save latest tag", default="latest")    
    parser.add_argument("--general_encoder_model_name", type=str,
                        help="general encoder model name or path", default=None)    
    parser.add_argument("--generator_model_name", type=str,
                        help="generator base model name or path", default=None)    
    parser.add_argument("--generator_peft_model_name", type=str,
                        help="generator peft model name or path", default=None)    
    parser.add_argument("--learning_rate", type=float,
        help= "Initial learning rate (after the potential warmup period) to use.", default=None)
    parser.add_argument("--lora_lr", type=float,
        help= "Initial lora learning rate (after the potential warmup period) to use.", default=None)


    args = parser.parse_args()

    local_rank = 0
    ddp_rank = 0

    if should_distribute():
        print('Using distributed PyTorch with NCCL backend')
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.getenv('RANK', -1))
        local_rank = int(os.getenv('LOCAL_RANK', 0))

    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s")
    fileHandler = logging.FileHandler(f"./tasi_modeling_catfid_miniX2.{datetime.now().strftime('%Y%m%d%H%M%S')}.{ddp_rank}.log")
    fileHandler.setFormatter(log_formatter)
    consoleHandler = logging.StreamHandler(sys.stdout) 
    consoleHandler.setFormatter(log_formatter)
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(logging.INFO)

    logger.info("using params from " + args.params_file)

    torch.autograd.set_detect_anomaly(True)

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = {k: v for k, v in vars(args).items() if v != None}
        # args.update(params)
        # args = Namespace(**args)
        params.update(args)
        args = Namespace(**params)

        for key in vars(args):
            logger.info(str(key) + " = " + str(vars(args)[key]))

    # tokenizer
    tokenizer = get_tokenizer(args)

    def catfid_collate_fn(batch):
        input_ids = []
        labels = []
        for record in batch:
            input_ids.append(record['input_ids'])
            # logger.info(f'len(input_ids): {len(record["input_ids"])}')
            labels.append(record['labels'])
    
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        # logger.info(f'input_ids.shape: {input_ids.shape}')
    
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}

    # dataset
    train_dataset = CatFidDataset(args.train_dataset, tokenizer, args, eval_mode = False)
    eval_dataset = sample_dataset(CatFidDataset(args.eval_dataset, tokenizer, args, eval_mode = True), args.eval_size)

    if is_distributed():
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None
    eval_sampler = SequentialSampler(eval_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=(train_sampler == None),
                                  collate_fn=catfid_collate_fn,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=catfid_collate_fn,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if is_distributed():
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()

    # models
    model = get_model(args)

    logger.info('generator info:')
    print_trainable_parameters(model)

    if is_distributed():
        model.to(device)
    # else:
    #     # generator has already been placed on GPUs via auto device_map

    # optimizer and scheduler
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    lora_lr = args.lora_lr

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, weight_decay = weight_decay, lora_lr= lora_lr)

    if is_distributed() and args.use_oss:
        optimizer = OSS(params = optimizer_grouped_parameters, optim = torch.optim.AdamW, lr=learning_rate, betas=(0.9, 0.95))
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, betas=(0.9, 0.95))

    if is_distributed():
        # model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model = nn.parallel.DistributedDataParallel(model)
        # model = ShardedDDP(model, optimizer)

    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps)

    lr_scheduler_type = args.lr_scheduler_type
    num_warmup_steps = args.num_warmup_steps
    num_train_epochs = args.num_train_epochs
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_epochs * num_update_steps_per_epoch,
    )

    ckpt_step = args.load_checkpoint

    # training loop
    save_steps = args.save_steps
    save_dir = args.save_dir
    save_flag = ddp_rank == 0 if should_distribute() else True
    global_steps = 0
    model_to_save = model.module if hasattr(model, 'module') else model
    logger.info(f"type(model): {type(model)}")
    logger.info(f"type(model_to_save): {type(model_to_save)}")
    logger.info(f"***** Evaluating loss, on start *****")
    loss = evaluation(model, eval_dataloader)
    logger.info(f"loss: {loss}")
    torch.cuda.empty_cache()
    gc.collect()
    # # test saving
    # if save_flag:
    #     ckpt_id = f'{global_steps}_{loss}'
    #     logger.info(f'Saving checkpoint {ckpt_id}')
    #     save_model(model_to_save, None if args.generator_use_peft else tokenizer, save_dir, ckpt_id, args.save_latest)
    # if is_distributed(): torch.distributed.barrier()
    for epoch in range(num_train_epochs):
        if train_sampler: train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            if ckpt_step != None and ckpt_step > 0 and global_steps <= ckpt_step: 
                global_steps += 1
                continue
            start = time.time()
            batch = to_device(batch, model.device)
            outputs = model(**batch, use_cache = False)
            loss = outputs.loss
            end = time.time()
            e2e_time = end - start
            logger.info(f"Epoch: {epoch}, Step: {step}, Latency: {e2e_time:.2f}s, loss = {loss}")
            total_loss += loss.item()
            loss = loss / gradient_accumulation_steps
            loss.backward()
            if ((step + 1) % gradient_accumulation_steps == 0):
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.get_parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # Add memory release here
                torch.cuda.empty_cache()
                gc.collect()
            global_steps += 1
            if save_steps > 0 and save_dir and (global_steps % save_steps) == 0:
                logger.info(f"***** Evaluating loss, Epoch {epoch}:{step} *****")
                loss = evaluation(model, eval_dataloader)
                logger.info(f"loss: {loss}")
                if save_flag:
                    ckpt_id = f'{global_steps}_{loss}'
                    logger.info(f'Saving checkpoint {ckpt_id}')
                    save_model(model_to_save, None if args.generator_use_peft else tokenizer, save_dir, ckpt_id, args.save_latest)
                if is_distributed(): torch.distributed.barrier()

        # Evaluate loss on the validation set.
        total_loss = total_loss / len(train_dataloader)
        logger.info(f"***** Evaluating loss, Epoch {epoch+1}/{num_train_epochs} *****, train_loss: {total_loss}")
        loss = evaluation(model, eval_dataloader)
        logger.info(f"loss: {loss}")
        if save_dir:
            if save_flag:
                ckpt_id = f'{global_steps}_{loss}'
                logger.info(f'Saving checkpoint {ckpt_id}')
                save_model(model_to_save, None if args.generator_use_peft else tokenizer, save_dir, ckpt_id, args.save_latest)
            if is_distributed(): torch.distributed.barrier()
