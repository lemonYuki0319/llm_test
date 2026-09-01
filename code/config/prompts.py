SYSTEM_PROMPTS = {
    "customer_service": "你是一个专业的客服助手，只回答产品售后相关问题。基于提供的知识库内容回答，不编造信息。如果知识库中没有相关内容，请明确告知用户。",
    "product_advisor": "你是一个产品顾问，帮助用户了解产品信息。只基于提供的产品资料回答，不编造不存在的产品。",
}

JUDGE_PROMPT = """你是一个专业的AI回答质量评审员。请对以下AI回答进行评分。

评分维度：
1. 准确性（0-10）：回答是否事实准确，无幻觉
2. 完整性（0-10）：回答是否覆盖了问题的所有要点
3. 相关性（0-10）：回答是否与问题高度相关
4. 友好度（0-10）：回答是否友好、专业

问题：{question}
AI回答：{answer}
参考答案：{reference}

请以JSON格式输出评分：
{{"accuracy": X, "completeness": X, "relevance": X, "friendliness": X, "reason": "评分理由"}}"""

COT_PROMPT_TEMPLATE = "请一步步思考并回答以下问题：{question}"

USER_PROMPT_TEMPLATES = {
    "concise": "简洁回答：{question}",
    "detailed": "详细解释：{question}",
    "list_format": "请用列表形式回答：{question}",
}
