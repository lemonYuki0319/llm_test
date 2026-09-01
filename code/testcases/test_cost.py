import pytest
from utils.llm_client import LLMClient
from config.prompts import SYSTEM_PROMPTS


class TestCost:
    def test_token_consumption_baseline(self):
        client = LLMClient()
        scenarios = {
            "简单问答": {"question": "退货政策是什么？", "max_tokens": 500},
            "复杂推理": {"question": "订单满200减30，我买了3件各80元，最终多少钱？", "max_tokens": 800},
        }

        for scenario, config in scenarios.items():
            result = client.simple_chat(
                config["question"],
                system_prompt=SYSTEM_PROMPTS["customer_service"]
            )
            total_tokens = result["total_tokens"]
            print(f"{scenario} - 总Token: {total_tokens}")
            assert total_tokens <= config["max_tokens"], \
                f"{scenario}Token消耗{total_tokens}超出基线{config['max_tokens']}"

    def test_prompt_length_optimization(self):
        client = LLMClient()
        question = "退货政策是什么？"

        verbose_prompt = "请你作为一个专业的客服人员，用非常详细和友好的方式回答以下问题："
        concise_prompt = "简洁回答："

        verbose_result = client.simple_chat(verbose_prompt + question)
        concise_result = client.simple_chat(concise_prompt + question)

        print(f"冗余Prompt Token: {verbose_result['total_tokens']}, "
              f"精简Prompt Token: {concise_result['total_tokens']}")
        assert "退货" in concise_result["content"], "精简Prompt后回答仍包含核心信息"
