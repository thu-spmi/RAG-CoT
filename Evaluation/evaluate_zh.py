#Acknowlegement: This code is referenced from https://github.com/baidu/DuReader/blob/master/DuReader-Robust/evaluate.py
import sys
import ujson as json
import re
import string
from collections import Counter
import pickle
from collections import defaultdict

def _tokenize_chinese_chars(text):
    """
    :param text: input text, unicode string
    :return:
        tokenized text, list
    """

    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    output = []
    buff = ""
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp) or char == "=":
            if buff != "":
                output.append(buff)
                buff = ""
            output.append(char)
        else:
            buff += char

    if buff != "":
        output.append(buff)

    return output

def normalize_answer(in_str):
    """
    normalize the input unicode string
    """
    in_str = in_str.lower()
    sp_char = [
        u':', u'_', u'`', u'，', u'。', u'：', u'？', u'！', u'(', u')',
        u'“', u'”', u'；', u'’', u'《', u'》', u'……', u'·', u'、', u',',
        u'「', u'」', u'（', u'）', u'－', u'～', u'『', u'』', '|'
    ]
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


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

    ground_truth_tokens = _tokenize_chinese_chars(normalize_answer(ground_truth))
    prediction_tokens = _tokenize_chinese_chars(normalize_answer(prediction))
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
        id = data["id"]
        # 将 id 加入列表
        prediction_ids.append(id)
        # 将 id 和对应的 answer 存入字典
        id_to_answer[id] = data["prediction"]

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
    for dp in gold:
        cur_id = dp['id']
        if cur_id not in prediction_ids:
            print('missing prediction for id: {}'.format(cur_id))
            continue
        em, prec, recall = update_answer(metrics, id_to_answer[cur_id], dp['answers'])
    
    N = len(prediction)
    for k in metrics.keys():
        metrics[k] /= N

    print(metrics)

if __name__ == '__main__':
    eval(sys.argv[1], sys.argv[2])

