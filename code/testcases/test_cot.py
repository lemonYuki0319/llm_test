import pytest
from utils.llm_client import call_llm
from config.prompts import COT_PROMPT_TEMPLATE


class TestCoT:
    @pytest.mark.parametrize("item_index", range(3))
    def test_cot_completeness(self, cot_questions, item_index):
        item = cot_questions[item_index]
        cot_prompt = COT_PROMPT_TEMPLATE.format(question=item["question"])
        response = call_llm(cot_prompt)

        for step in item["expected_steps"]:
            assert step in response, f"推理步骤中缺少关键计算：{step}"

    def test_cot_logic_correctness(self):
        question = "一件商品原价200元，先打8折，再减20元优惠券，最终价格是多少？"
        cot_prompt = COT_PROMPT_TEMPLATE.format(question=question)
        response = call_llm(cot_prompt)

        assert "160" in response, "第一步计算结果错误（200×0.8=160）"
        assert "140" in response, "最终计算结果错误（160-20=140）"

    def test_cot_stability(self):
        question = "如果3个苹果5元，买12个苹果需要多少钱？"
        cot_prompt = COT_PROMPT_TEMPLATE.format(question=question)

        correct_count = 0
        for _ in range(5):
            response = call_llm(cot_prompt, temperature=0)
            if "20" in response:
                correct_count += 1

        assert correct_count >= 4, f"CoT推理稳定性不足，5次中仅{correct_count}次结论正确"
