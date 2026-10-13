"""
RAGAS 评估示例（v0.4+ API）

演示:创建样本 → 配置 LLM → 运行评估 → 查看结果
"""
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextRecall,
    LLMContextPrecisionWithReference,
)


def build_samples():
    """创建评估样本"""
    return [
        SingleTurnSample(
            user_input="数据安全法是什么时候施行的?",
            retrieved_contexts=[
                "《中华人民共和国数据安全法》于2021年6月10日通过,自2021年9月1日起施行。"
            ],
            response="《数据安全法》于2021年9月1日起施行。",
            reference="2021年9月1日起施行。",
        ),
        SingleTurnSample(
            user_input="数据安全法建立了什么制度?",
            retrieved_contexts=[
                "数据安全法建立了数据分类分级保护制度、数据安全审查制度和数据出口管制制度。"
            ],
            response="数据安全法建立了数据分类分级保护制度。",
            reference="建立了数据分类分级保护、数据安全审查、数据出口管制等制度。",
        ),
    ]


def minimax_model():
    return LangchainLLMWrapper(ChatOpenAI(
        model="MiniMax-M3",
        base_url="https://api.minimaxi.com/v1",
        api_key="sk-cp-lBv_rdTB9r_0E0UVw6GStQhZfqCZWx2z_y9GoVmkErjwXN_egYkEGq-ZQ9KI3GTufvoeScdDzgQh5LMrWsybadDwqE5FWPdDW3rzOBMQjqyQBAC8pPb-vVA",
        temperature=0.3
    ))


def main():
    # 1. 配置 LLM
    llm_mini = minimax_model()

    # 2. 配置指标(大写类,需传 llm)
    metrics = [
        Faithfulness(llm=llm_mini),
        ResponseRelevancy(llm=llm_mini),
        LLMContextRecall(llm=llm_mini),
        LLMContextPrecisionWithReference(llm=llm_mini),
    ]

    # 3. 创建数据集
    dataset = EvaluationDataset(build_samples())

    # 4. 运行评估
    result = evaluate(dataset=dataset, metrics=metrics)

    # 5. 查看结果
    print("\n===== 评估结果摘要 =====")
    print(result)

    print("\n===== 逐条明细 =====")
    df = result.to_pandas()
    print(df[["user_input", "faithfulness", "answer_relevancy", "context_recall", "llm_context_precision_with_reference"]])


if __name__ == "__main__":
    main()
