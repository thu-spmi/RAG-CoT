#Acknowlegement: This code is referenced from https://github.com/ShishirPatil/gorilla/blob/main/raft/raft.py
from typing import Literal, Any
import argparse
from openai import OpenAI
from typing import List, Union, Tuple
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import json
import PyPDF2
import random
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

DocType = Literal["api", "pdf", "json", "txt"]

def get_args() -> argparse.Namespace:
    """
    Parses and returns the arguments specified by the user's command
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--datapath", type=str, default="", help="The path at which the document is located")
    parser.add_argument("--output", type=str, default="./", help="The path at which to save the dataset")
    parser.add_argument("--distractors", type=int, default=3, help="The number of distractor documents to include per data point / triplet")
    parser.add_argument("--p", type=float, default=1.0, help="The percentage that the oracle document is included in the context")
    parser.add_argument("--questions", type=int, default=5, help="The number of data points / triplets to generate per chunk")
    parser.add_argument("--chunk_size", type=int, default=512, help="The size of each chunk in number of tokens")
    parser.add_argument("--doctype", type=str, default="pdf", help="The type of the document, must be one of the accepted doctypes", choices=["pdf", "txt", "json", "api"])
    parser.add_argument("--openai_key", type=str, default="", help="Your OpenAI key used to make queries to GPT-3.5 or GPT-4")
    parser.add_argument("--input_type", type=str, default="doc", help="You generate the dataset from a document or overwrite an existing dataset")
    parser.add_argument("--answer_type", type=str, default="none", help="You generate the dataset from a document or overwrite an existing dataset")

    args = parser.parse_args()
    return args

def read_dataset_hotpot(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    chunks = []
    qs = []
    sf = []
    answers = []
    
    # 遍历数据集中的每一项
    for data in dataset:
        # 读取question字段并添加到qs列表
        qs.append(data['question'])
        
        # 读取context字段并添加到chunks列表
        chunks.append(data['context'])
        
        # 读取supporting_facts字段并添加到sf列表
        sf.append(data['supporting_facts'])

        answers.append(data['answer'])
    
    return qs, chunks, sf, answers

def read_dataset_pubmed(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    chunks = []
    qs = []
    answers = []
    
    for id, data in dataset.items():
        qs.append(data['QUESTION'])
        
        context = data['CONTEXTS']
        chunks.append(' '.join(context))

        answers.append(data['final_decision'])
    
    return qs, chunks, answers

def encode_question_gen(answer_type: str, question: str, chunk: Any, answer: str) -> list[str]:
    """
    Encode multiple prompt instructions into a single string for the general case (`pdf`, `json`, or `txt`).
    """
    
    prompts = []
        
    prompt_pubmed_long = """
        Question: {question}\nContext: {context}\nAnswer:{answer}\n
        Please provide the reasoning process that shows how the above question leads to the answer by using the information in the context. Here is things to pay attention to: 
        - provide step-by-step reasoning on how to answer the question using the the information given in the context. 
        - In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context. 
        - End your response with final answer in the form <ANSWER>: $answer, The answer should be exactly as same as the Answer above.
    """.format(question=question, context=str(chunk), answer = answer)
    prompt_pubmed_short = """
        Question: {question}\nContext: {context}\n
        Answer this question using the information given in the context above. Here is things to pay attention to: 
        - First provide step-by-step reasoning on how to answer the question. 
        - In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context. 
        - End your response with final answer in the form <ANSWER>: $answer, The answer should be only in "yes" or "no" or "maybe".
    """.format(question=question, context=str(chunk))
    prompt= """
        Question: {question}\nContext: {context}\n
        Answer this question using the information given in the context above. Here is things to pay attention to: 
        - First provide step-by-step reasoning on how to answer the question. 
        - In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context. 
        - End your response with final answer in the form <ANSWER>: $answer, The answer should be succint.
    """.format(question=question, context=str(chunk))
    prompts.append({"role": "system", "content": "You are a helpful question answerer who can provide an answer given a question and relevant context."})
    
    if answer_type == "long":
        prompts.append({"role": "user", "content": prompt_pubmed_long})
    elif answer_type == "short":
        prompts.append({"role": "user", "content": prompt_pubmed_short})
    else:
        prompts.append({"role": "user", "content": prompt})
    #print(prompts)
    return prompts

def generate_label(answer:str, answer_type: str, question: str, context: Any, doctype: DocType = "pdf") -> str | None:
    """
    Generates the label / answer to `question` using `context` and GPT-4.
    """
    question = encode_question_gen(answer_type, question, context, answer)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=question,
        n=1,
        temperature=0
    )
    response = response.choices[0].message.content
    #print(response)
    return response


def add_chunk_to_dataset_hotpotQA(
    answer_type: str,
    chunk: List[List[Union[str, List[str]]]], 
    q: str,
    sf: List[List[Union[str, int]]],
    answer: str,
    doctype: DocType = "api",
    num_distract: int = 3
) -> None:
    """
    Given a chunk, create {Q, A, D} triplets and add them to the dataset.
    """
    global ds
    
    # Integrate paragraphs under each title in the chunk
    integrated_chunk = []
    for title, paragraphs in chunk:
        integrated_paragraph = ' '.join(paragraphs)
        integrated_chunk.append([title, integrated_paragraph])
    
    # Create oracle context
    oracle_titles = [item[0] for item in sf]
    oracle_context = []
    for title, text_chunk in integrated_chunk:
        if title in oracle_titles:
            oracle_context.append(text_chunk)
    
    # Shuffle and select distractor documents
    distractor_indices = [i for i in range(len(integrated_chunk)) if integrated_chunk[i][0] not in oracle_titles]
    distractor_docs = []
    for j in random.sample(distractor_indices, min(num_distract, len(distractor_indices))):
        distractor_docs.append(integrated_chunk[j][1])
    
    # Combine oracle and distractor documents
    docs = oracle_context + distractor_docs
    
    # Shuffle the docs list
    random.shuffle(docs)
    
    # Construct data point
    datapt = {
        "id": f"seed_task_{0 if not ds else ds.num_rows}",
        "type": "general",
        "question": q,
        "context": {
            "title": [["placeholder_title"] * len(docs)],
            "sentences": [docs]
        },
        "oracle_context": oracle_context,
        "cot_answer": generate_label(q, oracle_context, doctype),
        "cot_answer": answer,
        "instruction": ""
    }
    
    # Construct instruction
    instruction_context = ""
    for doc in docs:
        instruction_context += "<DOCUMENT>" + str(doc) + "</DOCUMENT>\n"
    instruction_context += q
    datapt["instruction"] = instruction_context
    
    # Add to dataset
    if not ds:
        datapt = {
            key: [value] for key, value in datapt.items()
        }
        ds = Dataset.from_dict(datapt)
    else:
        ds = ds.add_item(datapt)


def add_chunk_to_dataset_pubmedQA(
    answer:str,
    answer_type: str,
    chunks: list[str], 
    chunk: str, 
    q: str,
    doctype: DocType = "api",  
    num_distract: int = 3, 
    p: float = 1.0
) -> None:
    """
    Given a chunk, create {Q, A, D} triplets and add them to the dataset.
    """
    global ds
    i = chunks.index(chunk)
    
    datapt = {
        "id": None,
        "type": None,
        "question": None,
        "context": None,
        "oracle_context": None,
        "cot_answer": None
    }

    datapt["id"] = f"seed_task_{0 if not ds else ds.num_rows}"
    datapt["type"] = "api call" if doctype == "api" else "general"
    datapt["question"] = q

    # add num_distract distractor docs
    docs = [chunk]
    indices = list(range(0, len(chunks)))
    indices.remove(i)
    for j in random.sample(indices, num_distract):
        docs.append(chunks[j])
    # decides whether to add oracle document
    oracle = random.uniform(0, 1) < p
    if not oracle:
        docs[0] = chunks[random.sample(indices, 1)[0]]
    random.shuffle(docs)

    d = {
        "title": [],
        "sentences": []
    }

    d["title"].append(["placeholder_title"]*(num_distract + 1))
    d["sentences"].append(docs)
    datapt["context"] = d
    datapt["oracle_context"] = chunk

    # add answer to q
    datapt["cot_answer"] = generate_label(answer, answer_type, q, chunk, doctype) 
    datapt["cot_answer"] = answer
    # construct model instruction 
    context = ""
    for doc in docs:
        context += "<DOCUMENT>" + str(doc) + "</DOCUMENT>\n"
    context += q
    datapt["instruction"] = context

    # add to dataset
    if not ds:
        # init ds
        datapt["id"] = [datapt["id"]]
        datapt["type"] = [datapt["type"]]
        datapt["question"] = [datapt["question"]]
        datapt["context"] = [datapt["context"]]
        datapt["oracle_context"] = [datapt["oracle_context"]]
        datapt["cot_answer"] = [datapt["cot_answer"]]
        datapt["instruction"] = [datapt["instruction"]]
        ds = Dataset.from_dict(datapt)
    else:
        ds = ds.add_item(datapt)
    
    #print(ds)

if __name__ == "__main__":
    # run code
    args = get_args()
    
    OPENAPI_API_KEY = args.openai_key

    client = OpenAI(
        api_key=OPENAPI_API_KEY,
    )

    CHUNK_SIZE = args.chunk_size
    NUM_DISTRACT_DOCS = args.distractors

    ds = None

    if args.input_type == "hotpot":
        qs, chunks, sfs, answers = read_dataset_hotpot(args.datapath)
        print("load dataset done")
        cnt = 0
        for chunk, q ,sf, answer in zip(chunks, qs, sfs, answers):
            #print(sf)
            cnt = cnt + 1
            print(cnt)
            add_chunk_to_dataset_hotpotQA(args.answer_type, chunk, q, sf, answer, args.doctype, NUM_DISTRACT_DOCS)
            if cnt == 5000:
                break
        print("add chunk to dataset done")    
    
    elif args.input_type == "pubmed":
        qs, chunks, answers = read_dataset_pubmed(args.datapath)
        print("load dataset done")
        cnt = 0
        for chunk, q, answer in zip(chunks, qs, answers):
           # print(sf)
            cnt = cnt + 1
            print(cnt)
            add_chunk_to_dataset_pubmedQA(answer, args.answer_type, chunks, chunk, q, args.doctype, NUM_DISTRACT_DOCS)
     
    # Save as .arrow format
    ds.save_to_disk(args.output)
    
    # Save as .jsonl format
    ds.to_json(args.output + ".jsonl")
