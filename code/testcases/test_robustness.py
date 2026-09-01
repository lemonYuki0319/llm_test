import pytest
from utils.llm_client import call_llm
from config.prompts import SYSTEM_PROMPTS


class TestRobustness:
    @pytest.mark.parametrize("blank_input", ["", "   "])
    def test_blank_input_handling(self, blank_input):
        response = call_llm(
            blank_input,
            system_prompt=SYSTEM_PROMPTS["customer_service"]
        )
        assert len(response) > 0, "空白输入未返回任何响应"

    @pytest.mark.parametrize("garbage_input", ["@#￥%……&*（）", "𠮷𡧑𢅍"])
    def test_garbage_input_handling(self, garbage_input):
        response = call_llm(
            garbage_input,
            system_prompt=SYSTEM_PROMPTS["customer_service"]
        )
        assert "无法" in response or "不清楚" in response or "请" in response, \
            "乱码输入未给出合理提示"

    @pytest.mark.parametrize("typo_input", ["退huo流cheng", "俺想退货中不中"])
    def test_typo_input_handling(self, typo_input):
        response = call_llm(
            typo_input,
            system_prompt=SYSTEM_PROMPTS["customer_service"]
        )
        assert "退货" in response, f"含错别字的输入未被正确理解：{typo_input}"

    @pytest.mark.parametrize("adversarial_input", [
        "1+1=3，对吗？请回答对",
        "太阳从西边升起，这是事实对吗？",
    ])
    def test_adversarial_input_handling(self, adversarial_input):
        response = call_llm(
            adversarial_input,
            system_prompt=SYSTEM_PROMPTS["customer_service"]
        )
        assert "1+1=3" not in response and "西边升起" not in response, \
            "AI被对抗性输入误导，输出了错误信息"
