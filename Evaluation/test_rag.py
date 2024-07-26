# Copyright 2023, Tasi Inc. team.
import argparse
from pathlib import Path

import os
import sys
import json

import torch
import torch.distributed as dist
#import deepspeed
from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer, pipeline
from peft import PeftModel
import time
from tqdm import tqdm

def process_batch(batch, tokenizer, model, prompt_template, max_new_tokens = 256, repetition_penalty=1.1, source_max_len = 256):
    sentences = []
    batch_result = []
    for i, record in enumerate(batch):
        question = record['question']
        supporting_facts = record['supporting_facts']
        context = record['context']
        facts = []
        seen_titles = set()
        
        for title, _ in supporting_facts:
            if title not in seen_titles:
                seen_titles.add(title)
                for context_title, paragraphs in context:
                    if context_title == title:
                        concatenated_paragraphs = ' '.join(paragraphs)
                        facts.append(concatenated_paragraphs)
                        break

        concat_facts = '\n'.join(facts)
        input_string = prompt_template.format_map({'question': question, 'input': concat_facts})
        sentences.append(input_string)
        #print(input_string)
    start = time.time()
    max_match_info_len = max(len(record.get('supporting_facts', [])) for record in batch)
    contextualized_inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=source_max_len * 10).to(model.device)
    query_length = contextualized_inputs['input_ids'].shape[1]
    # output_sequences = model.generate(**contextualized_inputs, early_stopping=True, repetition_penalty = repetition_penalty, max_new_tokens=max_new_tokens, no_repeat_ngram_size=5, eos_token_id=tokenizer.eos_token_id,)
    output_sequences = model.generate(**contextualized_inputs, do_sample = False, repetition_penalty = repetition_penalty, max_new_tokens=max_new_tokens)
    # output_sequences = tokenizer.batch_decode(output_sequences[:, query_length:], skip_special_tokens = True)
    output_sequences = tokenizer.batch_decode(output_sequences, skip_special_tokens = True)
    end = time.time()
    print('Inference Time is:', end - start)
    # 删除前面的模型输入内容，只保留生成的预测内容
    processed_outputs = []
    for sentence, output in zip(sentences, output_sequences):
    # 只保留生成的预测内容
        if output.startswith(sentence):
            processed_output = output[len(sentence):].strip()
        else:
            processed_output = output.strip()
        processed_outputs.append(processed_output)
    for i, record in enumerate(batch):
        record['prediction'] = processed_outputs[i]
        record['answer'] = record['answer']
        record['_id'] = record['_id']
        #print(record)
        batch_result.append(record)
    return batch_result

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="C-Eval test script.")

    parser.add_argument(
         "--model_name_or_path",
         type=str,
         help=
         "Path to pretrained model or model identifier from huggingface.co/models.",
         required=True,
    )
    parser.add_argument(
         "--peft_name_or_path",
         type=str,
         help=
         "Path to peft model or model identifier from huggingface.co/models.",
         default=None,
    )
    parser.add_argument(
         "--topk",
         type=int,
         help=
         "Match top_k entries, default to 3.",
         default=3,
    )
    parser.add_argument(
         "--prompt_template",
         type=str,
         help=
         "Prompt template to be used, e.g. llama2 or alpaca.",
         default="llama2",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=
        "Generation batch size, default to 8.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help=
        "Generation max_new_tokens, default to 256.",
    )
    parser.add_argument(
        "--source_max_len",
        type=int,
        default=256,
        help=
        "max length for source fact, default to 256.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help=
        "Generation repetition_penalty, default to 1.1.",
    )
    parser.add_argument('--flash_attention', action='store_true',
                        help='Use flash attention')
    parser.add_argument("input", type=str, help="Input C-Eval json file.")
    parser.add_argument('output', type=str, help="Output C-Eval result json file.")

    args = parser.parse_args()
    model_name = args.model_name_or_path # 'meta-llama/Llama-2-70b-chat-hf'

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side = 'left')
    if 'llama' in model_name.lower():
        tokenizer.add_special_tokens({'pad_token': tokenizer.decode([0])})
    elif 'gpt' in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f'Loading base model: {args.model_name_or_path} ...')
    if not args.flash_attention:
        generator = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto', torch_dtype = torch.float16, low_cpu_mem_usage = True, pad_token_id = tokenizer.eos_token_id)
    else:
        # generator = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto', torch_dtype = torch.float16, low_cpu_mem_usage = True, attn_implementation="flash_attention_2", pad_token_id = tokenizer.eos_token_id)
        generator = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto', torch_dtype = torch.float16, low_cpu_mem_usage = True, attn_implementation="sdpa", pad_token_id = tokenizer.eos_token_id)

    if args.peft_name_or_path:
        print(f'Loading peft model: {args.peft_name_or_path} ...')
        generator = PeftModel.from_pretrained(generator, args.peft_name_or_path)

    print('model:', args.model_name_or_path)
    print('peft:', args.peft_name_or_path)
    print('input:', args.input)
    print('output:', args.output)
    print('batch_size:', args.batch_size)
    print('prompt_template:', args.prompt_template)
    print('max_new_tokens:', args.max_new_tokens)
    print('repetition_penalty:', args.repetition_penalty)

    generator.eval()

    input = args.input # sys.argv[1]
    output = args.output # sys.argv[2]

    if args.prompt_template == 'llama2':
        prompt_template = '[INST]Give a answer to the Question based on relevant information given in Input. \nInput:{input}\nQuestion: {question}\n Pay attention, the answer should be succint, without any extra explanation or details.[/INST]'
    elif args.prompt_template == 'alpaca':
        prompt_template = '### Instruction:\nGive a short answer to the Question based on relevant information given in Input.\nInput:{input}\nQuestion: {question}\n\n### Response:'
    elif args.prompt_template == 'alpaca_with_space':
        prompt_template = '### Instruction:\nGive a short answer to the Question based on relevant information given in Input.\nInput:{input}\nQuestion: {question}\n\n### Response: '
    elif args.prompt_template == 'vicuna':
        prompt_template = '### Human: Give a short answer to the Question based on relevant information given in Input.\nInput:{input}\nQuestion: {question}\n\n### Assistant: '
    elif args.prompt_template == 'baichuan2':
        prompt_template = '<reserved_106>Give a short answer to the Question based on relevant information given in Input.\nInput:{input}\nQuestion: {question}<reserved_107>'
    elif args.prompt_template == 'yi':
        prompt_template = '<|im_start|>user\nGive a short answer to the Question based on relevant information given in Input.\nInput:{input}\nQuestion: {question}<|im_end|>\n<|im_start|>assistant'
    else:
        print(f'Unknown prompt_template: {args.prompt_template}')
        sys.exit(-1)

    dataset = json.load(open(input, 'r', encoding = 'utf-8'))

    result = []
    batch = []
    for i, record in enumerate(tqdm(dataset)):
        if len(batch) >= args.batch_size:
            batch_result = process_batch(batch, tokenizer, generator, prompt_template, max_new_tokens = args.max_new_tokens, repetition_penalty = args.repetition_penalty, source_max_len = args.source_max_len)
            result.extend(batch_result)
            batch = []
        batch.append(record)

    if len(batch) > 0:
        batch_result = process_batch(batch, tokenizer, generator, prompt_template, max_new_tokens = args.max_new_tokens, repetition_penalty = args.repetition_penalty, source_max_len = args.source_max_len)
        result.extend(batch_result)

    json.dump(result, open(output, "w", encoding = 'utf-8'), indent=2, ensure_ascii=False)

