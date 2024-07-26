#Acknowlegement: This code is referenced from https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
import sys
import ujson as json
import re
import string
from collections import Counter
import pickle
from collections import defaultdict

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _tokenize(text):
    """分词函数：将文本按空格分割成单词列表"""
    return text.split()

def find_lcs(s1, s2):
    """find the longest common subsequence between s1 ans s2"""
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    max_len = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j]+1
                if m[i+1][j+1] > max_len:
                    max_len = m[i+1][j+1]
                    p = i+1
    return s1[p-max_len:p], max_len

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    ground_truth_tokens = _tokenize(normalize_answer(ground_truth))
    prediction_tokens = _tokenize(normalize_answer(prediction))
    lcs, lcs_len = find_lcs(ground_truth_tokens, prediction_tokens)
    if lcs_len == 0:
        return ZERO_METRIC
    precision = 1.0 * lcs_len / len(prediction_tokens)
    recall = 1.0 * lcs_len / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall

def eval(prediction_file, gold_file):
    with open(prediction_file, 'r', encoding='utf-8', errors='ignore') as f:
        prediction = json.load(f)
    with open(gold_file, 'r', encoding='utf-8', errors='ignore') as f:
        gold = json.load(f)
    
    # 创建空列表存储所有 id
    prediction_ids = []

    # 创建字典用于根据 id 找到对应的 answer
    id_to_answer = {}

    # 遍历数据集
    for data in prediction:
        # 提取 id
        id = data["_id"]
        # 将 id 加入列表
        prediction_ids.append(id)
        # 将 id 和对应的 answer 存入字典
        id_to_answer[id] = data["prediction"]

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
    for dp in gold:
        cur_id = dp['_id']
        if cur_id not in prediction_ids:
            print('missing prediction for id: {}'.format(cur_id))
            continue
        em, prec, recall = update_answer(metrics, id_to_answer[cur_id], dp['answer'])
    
    N = len(prediction)
    for k in metrics.keys():
        metrics[k] /= N

    print(metrics)

if __name__ == '__main__':
    eval(sys.argv[1], sys.argv[2])

