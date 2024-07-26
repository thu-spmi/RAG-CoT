#Acknowlegement: This code is referenced from https://github.com/ShishirPatil/gorilla/blob/main/raft/raft.py
from typing import Literal, Any
import argparse
from openai import OpenAI
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import json
import PyPDF2
import random
from langchain_experimental.text_splitter_zh import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

DocType = Literal["api", "pdf", "json", "txt"]
MODEL = "gpt-3.5-turbo"

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

    args = parser.parse_args()
    return args

def read_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    chunks = []
    qs = []
    answers = []

    for entry in dataset["data"]:
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"]
            chunks.append(context)

            for qa in paragraph["qas"]:
                question = qa["question"]
                qs.append(question)
                for ans in qa["answers"]:
                    answer = ans["text"]
                    answers.append(answer)
    return chunks, qs, answers

def encode_question_gen(question: str, chunk: Any) -> list[str]:
    """
    Encode multiple prompt instructions into a single string for the general case (`pdf`, `json`, or `txt`).
    """
    
    prompts = []
        
    prompt = """
        Question: {question}\nContext: {context}\n
        使用上文中提供的信息回答这个问题。以下是需要注意的事项：
        - 首先，你需要提供逐步推理过程来说明如何回答的问题。
        - 在推理过程中，如果需要复制粘贴上文中的某些句子，请将它们放在##引用_开始##和##引用_结束##之间。这意味着##引用_开始##和##引用_结束##之外的内容不是直接从上下文中复制粘贴的。
        - 在回答的最后以<答案>: $答案内容 的形式给出最终答案，注意，你只能直接明了地输出答案，不应当有多余的细节和解释。
    """.format(question=question, context=str(chunk))
    prompts.append({"role": "system", "content": "你是一个很专业的答题者，可以根据问题和相关上下文提供答案。"})
    prompts.append({"role": "user", "content": prompt})
    #print(prompts)
    return prompts

def generate_label(question: str, context: Any, doctype: DocType = "pdf") -> str | None:
    """
    Generates the label / answer to `question` using `context` and GPT-4.
    """
    question = encode_question_gen(question, context)
    response = client.chat.completions.create(
        model=MODEL,
        messages=question,
        n=1,
        temperature=0
    )
    response = response.choices[0].message.content
    #print(response)
    return response

def add_chunk_to_dataset_dureader(
    chunks: list[str], 
    chunk: str, 
    q: str,
    answer: str,
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

    d["title"].append(["placeholder_title"]*(num_distract+1))
    d["sentences"].append(docs)
    datapt["context"] = d
    datapt["oracle_context"] = chunk

    # add answer to q
    datapt["cot_answer"] = generate_label(q, chunk, doctype) 
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

def save_to_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line)
            f.write('\n')

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

    if args.input_type == "dureader":
        chunks, qs, answers = read_dataset(args.datapath)
        print("load dataset done")
        cnt = 0
        for chunk, q, answer in zip(chunks, qs, answers):
            add_chunk_to_dataset_dureader(chunks, chunk, q, answer, args.doctype, NUM_DISTRACT_DOCS)
            cnt = cnt + 1
            print(cnt)
            if cnt == 6000:
                break
        print("add chunk to dataset done")    


    #print(ds)

    # Save as .arrow format
    ds.save_to_disk(args.output)
    
    # Save as .jsonl format
    #ds.to_json(args.output + ".jsonl")
    save_to_jsonl(ds, args.output + ".jsonl")
