import json
import re

from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu


def assert_keywords_in_response(response, keywords, msg=None):
    missing = [kw for kw in keywords if kw not in response]
    assert not missing, msg or f"回答中缺少关键词：{missing}"


def assert_no_keywords_in_response(response, keywords, msg=None):
    found = [kw for kw in keywords if kw in response]
    assert not found, msg or f"回答中包含不应出现的关键词：{found}"


def assert_json_format(response, required_fields=None, msg=None):
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        assert False, msg or "输出不是有效的JSON格式"
    if required_fields:
        missing = [f for f in required_fields if f not in result]
        assert not missing, msg or f"JSON中缺少必要字段：{missing}"
    return result


def assert_bertscore_above(candidate, reference, threshold=0.7, lang="zh", msg=None):
    P, R, F1 = bert_score_fn([candidate], [reference], lang=lang)
    score = F1[0].item()
    assert score >= threshold, msg or f"BERTScore F1={score:.4f}低于阈值{threshold}"
    return score


def assert_rouge_above(candidate, reference, threshold=0.5, msg=None):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    scores = scorer.score(reference, candidate)
    rouge1 = scores["rouge1"].fmeasure
    assert rouge1 >= threshold, msg or f"ROUGE-1={rouge1:.4f}低于阈值{threshold}"
    return rouge1


def assert_bleu_above(candidate_tokens, reference_tokens_list, threshold=0.3, msg=None):
    score = sentence_bleu(reference_tokens_list, candidate_tokens)
    assert score >= threshold, msg or f"BLEU={score:.4f}低于阈值{threshold}"
    return score


def assert_response_latency(latency, threshold=5.0, msg=None):
    assert latency <= threshold, msg or f"响应时间{latency:.2f}s超过阈值{threshold}s"


def assert_no_hallucination(response, context, msg=None):
    sentences = re.split(r'[。！？\n]', response)
    unsupported = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 5:
            continue
        has_support = any(char in context for char in sent[:10])
        if not has_support and len(sent) > 10:
            unsupported.append(sent)
    if unsupported:
        assert False, msg or f"以下内容可能为幻觉（无上下文支撑）：{unsupported[:3]}"
