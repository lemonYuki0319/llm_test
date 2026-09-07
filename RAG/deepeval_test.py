"""
DeepEval 框架 RAG 评估 + AI 单元测试示例

包含:
    1. 基础评估(evaluate 批量)
    2. AI 单元测试(assert_test + pytest)
    3. 阈值断言(threshold)
    4. 自定义指标(GEval)
    5. 参数化测试(@pytest.mark.parametrize)

运行:
    pytest deepeval_test.py -v          # 运行单元测试
    python deepeval_test.py            # 运行批量评估
"""
import os

import pytest

from deepeval import assert_test, evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# ==================== 配置 ====================
os.environ["OPENAI_API_KEY"] = "sk-cp-lBv_rdTB9r_0E0UVw6GStQhZfqCZWx2z_y9GoVmkErjwXN_egYkEGq-ZQ9KI3GTufvoeScdDzgQh5LMrWsybadDwqE5FWPdDW3rzOBMQjqyQBAC8pPb-vVA"
os.environ["OPENAI_API_BASE"] = "https://api.minimaxi.com/v1"
MODEL = "MiniMax-M3"

# 阈值:低于此分数则测试失败
THRESHOLD = 0.7


# ==================== 测试数据 ====================
TEST_DATA = [
    {
        "input": "数据安全法是什么时候施行的?",
        "actual_output": "《数据安全法》于2021年9月1日起施行。",
        "expected_output": "2021年9月1日起施行。",
        "retrieval_context": [
            "《中华人民共和国数据安全法》于2021年6月10日通过,"
            "自2021年9月1日起施行。"
        ],
    },
    {
        "input": "数据安全法建立了什么制度?",
        "actual_output": "数据安全法建立了数据分类分级保护制度。",
        "expected_output": "建立了数据分类分级保护、数据安全审查、"
        "数据出口管制等制度。",
        "retrieval_context": [
            "数据安全法建立了数据分类分级保护制度、"
            "数据安全审查制度和数据出口管制制度。"
        ],
    },
]


def make_test_case(data: dict) -> LLMTestCase:
    """从字典构建 LLMTestCase"""
    return LLMTestCase(
        input=data["input"],
        actual_output=data["actual_output"],
        expected_output=data["expected_output"],
        retrieval_context=data["retrieval_context"],
    )


def build_metrics():
    """配置指标,统一设置阈值"""
    return [
        AnswerRelevancyMetric(threshold=THRESHOLD, model=MODEL),
        FaithfulnessMetric(threshold=THRESHOLD, model=MODEL),
        ContextualRecallMetric(threshold=THRESHOLD, model=MODEL),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=MODEL),
    ]


def build_custom_metric():
    """
    自定义指标:用 GEval 定义评分标准
    LLMTestCaseParams 指定评估时用哪些字段
    """
    return GEval(
        name="准确性",
        criteria="判断实际输出与期望输出在事实层面是否一致,"
        "关键信息(时间、制度名称)不能遗漏或错误",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=MODEL,
        threshold=THRESHOLD,
    )


# ================================================================
# 方式一:AI 单元测试(pytest 风格,每条用例独立断言)
# 运行: pytest deepeval_test.py::test_rag_faithfulness -v
# ================================================================
@pytest.mark.parametrize("data", TEST_DATA)
def test_rag_faithfulness(data: dict):
    """测试忠实度:回答是否基于检索上下文(阈值 0.7)"""
    test_case = make_test_case(data)
    metric = FaithfulnessMetric(threshold=THRESHOLD, model=MODEL)
    assert_test(test_case, [metric])


@pytest.mark.parametrize("data", TEST_DATA)
def test_rag_answer_relevancy(data: dict):
    """测试答案相关性:回答是否切题(阈值 0.7)"""
    test_case = make_test_case(data)
    metric = AnswerRelevancyMetric(threshold=THRESHOLD, model=MODEL)
    assert_test(test_case, [metric])


@pytest.mark.parametrize("data", TEST_DATA)
def test_rag_context_recall(data: dict):
    """测试上下文召回:金标要点是否被检索到(阈值 0.7)"""
    test_case = make_test_case(data)
    metric = ContextualRecallMetric(threshold=THRESHOLD, model=MODEL)
    assert_test(test_case, [metric])


# ================================================================
# 方式二:综合单元测试(一次跑多个指标 + 自定义指标)
# 运行: pytest deepeval_test.py::test_rag_full_suite -v
# ================================================================
@pytest.mark.parametrize("data", TEST_DATA)
def test_rag_full_suite(data: dict):
    """综合测试:同时验证 4 个内置指标 + 1 个自定义指标"""
    test_case = make_test_case(data)
    metrics = build_metrics() + [build_custom_metric()]
    assert_test(test_case, metrics)


# ================================================================
# 方式三:批量评估(非 pytest,直接运行看结果)
# 运行: python deepeval_test.py
# ================================================================
def main():
    test_cases = [make_test_case(d) for d in TEST_DATA]
    metrics = build_metrics() + [build_custom_metric()]

    print("\n===== 批量评估 =====")
    results = evaluate(test_cases=test_cases, metrics=metrics)

    print("\n===== 逐条结果 =====")
    for result in results.test_results:
        print(f"\n问题: {result.input}")
        for mr in result.metrics_data:
            status = "PASS" if mr.success else "FAIL"
            print(f"  [{status}] {mr.name}: {mr.score:.4f} "
                  f"(阈值: {THRESHOLD})")
            print(f"         原因: {mr.reason}")


if __name__ == "__main__":
    main()
