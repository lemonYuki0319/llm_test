import json
import pytest
from utils.llm_client import LLMClient, call_llm
from utils.assertion_helper import assert_keywords_in_response
from config.prompts import SYSTEM_PROMPTS


class TestFunctional:
    @pytest.mark.parametrize("item_index", range(5))
    def test_accuracy_with_golden_set(self, llm_client, golden_set, item_index):
        item = golden_set[item_index]
        result = llm_client.simple_chat(
            item["question"],
            system_prompt=SYSTEM_PROMPTS["customer_service"]
        )
        assert_keywords_in_response(result["content"], item["keywords"])

    def test_context_coherence(self, llm_client):
        messages = [
            {"role": "user", "content": "我想买一台笔记本电脑"},
            {"role": "assistant", "content": "好的，请问您的预算是多少？"},
            {"role": "user", "content": "5000左右"},
            {"role": "assistant", "content": "推荐以下几款..."},
            {"role": "user", "content": "第一款能打游戏吗？"},
        ]
        result = llm_client.chat(messages)
        assert "游戏" in result["content"] or "性能" in result["content"]

    def test_output_json_format(self, llm_client):
        system_prompt = "请以JSON格式输出，包含name和price字段。只输出JSON，不要其他内容。"
        result = llm_client.simple_chat("推荐一款手机", system_prompt=system_prompt)
        try:
            parsed = json.loads(result["content"])
            assert "name" in parsed or "price" in parsed
        except json.JSONDecodeError:
            pytest.fail("输出不是有效的JSON格式")
