# An Empirical Study of Retrieval Augmented Generation with Chain-of-Thought

This is the official code for the paper "[An Empirical Study of Retrieval Augmented Generation with Chain-of-Thought](https://arxiv.org/abs/2407.15569)". 

This is an empirical study of RAG with CoT, which is a method called RAFT to improve the performance of Generative dialogue models. RAFT is a method that combines the chain-of-thought (CoT) with retrieval augmented generation (RAG) for Supervised Fine-Tuning (SFT) small-scale models to optimize their performance in reasoning tasks. We evaluated the RAFT method across multiple datasets and analysed its performance in various reasoning tasks, including long-form QA and short-form QA tasks, tasks in both Chinese and English, etc. Notably, it addresses the gaps in previous research regarding long-form QA tasks and Chinese datasets.

## Datasets

+ [HotpotQA](https://hotpotqa.github.io/)
+ [PubMedQA](https://pubmedqa.github.io/)
+ [DuReader_robust](https://github.com/baidu/DuReader/tree/master/DuReader-Robust)

## Baselines

+ LLaMA2-7B-chat / Qwen-1.5-7B-chat + zero-shot prompting
+ LLaMA2-7B-chat / Qwen-1.5-7B-chat + RAG
+ DSF + zero-shot prompting
+ DSF + RAG

### Supervised finetune

Run the following command to finetune your model. The format of the training dataset can be referenced in [train.jsonl](http://39.99.172.130/YuetongZhao/graduation_thesis_zhaoyt/-/blob/main/Code/DSF/train.jsonl). The fine-tuning parameters are in [params_file.json](http://39.99.172.130/YuetongZhao/graduation_thesis_zhaoyt/-/blob/main/Code/DSF/params_file.json)

```
nohup python3 -m torch.distributed.launch --nproc_per_node=8 --use_env --master_port 29600 DSF_finetune.py \
    --params_file params_file.json \
    --train_dataset train.jsonl \
    --eval_dataset eval.jsonl \
    --use_gradient_checkpoint_generator \
    > output.log 2>&1 &
```

## RAFT

### Generate RAFT dataset

Get your own API key for GPT-4 / GPT-3.5, and run the following command to generate the RAFT training dataset. 

You can modify parameter `distractors` to change the number of distractor documents. Additionally, you can modify parameter `input_type` to handle other open-source datasets. Note that if you are processing datasets other than the ones used in this study, you need to modify the code accordingly to adapt to the format of those datasets. 

Script [raft_en.py](http://39.99.172.130/YuetongZhao/graduation_thesis_zhaoyt/-/blob/main/Code/RAFT/raft_en.py) can process the HotpotQA and PubMedQA datasets, while script [raft_zh.py](http://39.99.172.130/YuetongZhao/graduation_thesis_zhaoyt/-/blob/main/Code/RAFT/raft_zh.py) can process the DuReader_robust dataset.

```
python3 raft.py \
    --datapath original_dataset.json \
    --output ./raft_training_dataset \
    --distractors 4 \
    --doctype json \
    --input_type hotpot \
    --openai_key your_openai_api_key
```

For each question-answer pair, append 4 randomly selected chunks as distractor documents and ganerate a Chain of Thought style response as the answer.  An example of the generated raft training dataset would like:

```
{
	"id":"seed_task_6",
	"type":"general",
	"question":"Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?",
	"context":{
		"sentences":[
			["James P. Comer (born James Pierpont Comer, September 25, 1934 in East Chicago, Indiana) is ......For his work and scholarship, Dr. Comer has been awarded 47 honorary degrees and has been recognized by numerous organizations.",
			"Eenasul Fateh (born 3 April 1959), also known by his stage name Aladin, is a Bangladeshi-British cultural practitioner, magician, live artist and former international management consultant.",
			"Christopher Nicholas Sarantakos (born December 19, 1967), known by ...... \"Phenomenon\" on NBC, and the 2014 stage show \"Criss Angel Magicjam\".",
			"Amaruk Caizapanta Anchapacxi (Quito, January 30, 1970), whose stage name is Amaruk ...... and balance for the awakening of a collective conscience.",
			"Management consulting is the practice of helping ...... advice and access to the consultants' specialized expertise.",
			"Mick Batyske (known by his stage name Mick, sometimes styled as MICK, and formerly Mick Boogie) is an American DJ and entrepreneur.......which he is also an advisor and consultant."]
			],
			"title":[
				["placeholder_title",
				"placeholder_title",
				"placeholder_title",
				"placeholder_title",
				"placeholder_title",
				"placeholder_title"]
				]
			},
	"oracle_context":[
		"Eenasul Fateh (born 3 April 1959), also known by his stage name Aladin, is a Bangladeshi-British cultural practitioner, magician, live artist and former international management consultant.",
		"Management consulting is the practice of helping ...... advice and access to the consultants' specialized expertise."
		],
		"cot_answer":"To answer the question, we need to identify the individual known by his stage name Aladin who helped organizations improve their performance as a consultant. From the context, we see that Eenasul Fateh is the real name of the individual known by the stage name Aladin. ##begin_quote## Eenasul Fateh (Bengali: born 3 April 1959), also known by his stage name Aladin, is a Bangladeshi-British cultural practitioner, magician, live artist and former international management consultant. ##end_quote## It is mentioned that management consulting involves helping organizations improve their performance through the analysis of existing organizational problems and the development of plans for improvement. ##begin_quote## Organizations may draw upon the services of management consultants for a number of reasons, including gaining external (and presumably objective) advice and access to the consultants' specialized expertise. ##end_quote## Therefore, the individual known as Aladin who helped organizations improve their performance as a consultant is Eenasul Fateh. <ANSWER>: Eenasul Fateh",
		"instruction":"<DOCUMENT>James P. Comer (born James Pierpont Comer, September 25, 1934 in East Chicago, Indiana) is currently ...... and has been recognized by numerous organizations.<\/DOCUMENT>\n<DOCUMENT>Eenasul Fateh (born 3 April 1959), also known by his stage name Aladin, is a Bangladeshi-British ...... international management consultant.<\/DOCUMENT>\n<DOCUMENT>Christopher Nicholas Sarantakos (born December 19, 1967), known by ...... on NBC, and the 2014 stage show \"Criss Angel Magicjam\".<\/DOCUMENT>\n<DOCUMENT>Amaruk Caizapanta Anchapacxi (Quito, January 30, 1970), whose stage name is ...... transformation and balance for the awakening of a collective conscience.<\/DOCUMENT>\n<DOCUMENT>Management consulting is the practice of ...... access to the consultants' specialized expertise.<\/DOCUMENT>\n<DOCUMENT>Mick Batyske (known by his stage name Mick, sometimes styled as MICK, and formerly Mick Boogie) is an American DJ ......which he is also an advisor and consultant.<\/DOCUMENT>\nWho was known by his stage name Aladin and helped organizations improve their performance as a consultant?"
}
```

### RAFT finetune

Run the following command to finetune your model with the RAFT method. The format of the training dataset can be referenced in [train_raft.jsonl](http://39.99.172.130/YuetongZhao/graduation_thesis_zhaoyt/-/blob/main/Code/RAFT/train_raft.jsonl)

```
nohup python3 -m torch.distributed.launch --nproc_per_node=8 --use_env --master_port 29600 raft_finetune.py \
    --params_file params_file_raft.json \
    --train_dataset train_raft.jsonl \
    --eval_dataset eval_raft.jsonl \
    --use_gradient_checkpoint_generator \
    > output.log 2>&1 &
```

## Evaluation

Run the following command to obtain the prediction results from the fine-tuned model or the pretrained model. The dataset with the model's prediction results is saved in `prediction_file.json`. 

The script [test.py](http://39.99.172.130/YuetongZhao/graduation_thesis_zhaoyt/-/blob/main/Code/Evaluation/test.py) is used for zero-shot prompting experiments, while [test_rag.py](http://39.99.172.130/YuetongZhao/graduation_thesis_zhaoyt/-/blob/main/Code/Evaluation/test_rag.py) is used for  RAG experiments.

```
nohup env CUDA_VISIBLE_DEVICES=0 python3 test.py \
    --model_name_or_path your/pretrained/model \
    --peft_name_or_path your/peft/path \
    --source_max_len 512 \
    --batch_size 4 \
    --prompt_template llama2 \
    test_dataset.json \
    prediction_file.json \
    > output1.log 2>&1 &
```

Run the script [evaluate_en.py](http://39.99.172.130/YuetongZhao/graduation_thesis_zhaoyt/-/blob/main/Code/Evaluation/evaluate_en.py) or [evaluate_zh.py](http://39.99.172.130/YuetongZhao/graduation_thesis_zhaoyt/-/blob/main/Code/Evaluation/evaluate_zh.py) to evaluate the model's prediction results, these two script are used to process English datasets and Chinese datasets respectively. You need to make some simple modifications to the scriptd to adapt it to the specific format of your own dataset.

```
python3 evaluate.py prediction_file.json answer_file.json
```

## References

The evaluation script is adapted from https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py and https://github.com/baidu/DuReader/blob/master/DuReader-Robust/evaluate.py

The RAFT method is proposed in paper [RAFT: Adapting Language Model to Domain Specific RAG ](https://arxiv.org/abs/2403.10131)
