import json
import pytest
from utils.llm_client import LLMClient
from utils.assertion_helper import assert_bertscore_above, assert_rouge_above
from config.prompts import SYSTEM_PROMPTS, JUDGE_PROMPT
from config.settings import DEFAULT_MODEL


class TestQuality:
    def test_llm_as_judge(self, golden_set):
        judge_client = LLMClient(model="gpt-4")
        target_client = LLMClient(model=DEFAULT_MODEL)

        item = golden_set[0]
        answer = target_client.simple_chat(
            item["question"],
            system_prompt=SYSTEM_PROMPTS["customer_service"]
        )["content"]

        judge_prompt = JUDGE_PROMPT.format(
            question=item["question"],
            answer=answer,
            reference=item["reference"]
        )
        judge_result = judge_client.simple_chat(judge_prompt, temperature=0)["content"]

        try:
            scores = json.loads(judge_result)
            assert scores["accuracy"] >= 6, f"准确性得分{scores['accuracy']}不达标"
            assert scores["relevance"] >= 6, f"相关性得分{scores['relevance']}不达标"
        except json.JSONDecodeError:
            pytest.fail(f"评审输出格式异常：{judge_result[:200]}")

    def test_bertscore_evaluation(self, golden_set):
        target_client = LLMClient(model=DEFAULT_MODEL)
        item = golden_set[0]
        answer = target_client.simple_chat(
            item["question"],
            system_prompt=SYSTEM_PROMPTS["customer_service"]
        )["content"]
        score = assert_bertscore_above(answer, item["reference"], threshold=0.6)
        print(f"BERTScore F1: {score:.4f}")

    def test_rouge_evaluation(self, golden_set):
        target_client = LLMClient(model=DEFAULT_MODEL)
        item = golden_set[0]
        answer = target_client.simple_chat(
            item["question"],
            system_prompt=SYSTEM_PROMPTS["customer_service"]
        )["content"]
        score = assert_rouge_above(answer, item["reference"], threshold=0.4)
        print(f"ROUGE-1: {score:.4f}")
