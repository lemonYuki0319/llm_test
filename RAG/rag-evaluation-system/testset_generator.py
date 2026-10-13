# testset_generator.py
"""
基于切分文档生成合成测试集
"""
import json
import os
import re
from typing import List, Dict

from langchain_openai import ChatOpenAI

from config import (
    MINIMAX_API_KEY,
    MINIMAX_BASE_URL,
    GENERATION_LLM_MODEL,
    DEFAULT_QA_PER_CHUNK,
    DEFAULT_NEGATIVE_COUNT,
    DEFAULT_MULTI_HOP_COUNT,
    get_testcases_dir,
)
from document_processor import load_knowledge_base, split_document, save_chunks


def get_llm():
    """获取生成用 LLM 实例"""
    model = ChatOpenAI(
        model_name=GENERATION_LLM_MODEL,
        api_key=MINIMAX_API_KEY,
        base_url=MINIMAX_BASE_URL,
    )
    return model


def _parse_llm_json(response_text: str, context: str = "") -> List[Dict]:
    """
    从 LLM 响应中提取并解析 JSON

    Args:
        response_text: LLM 响应文本
        context: 错误提示上下文

    Returns:
        解析后的列表，失败返回空列表
    """
    # 尝试提取 JSON（处理可能的额外文字）
    json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(0)

    try:
        qa_list = json.loads(response_text)
    except json.JSONDecodeError:
        # 容错：修复中文中未转义的双引号
        try:
            fixed = re.sub(
                r'([\u4e00-\u9fa5])"([\u4e00-\u9fa5])', r"\1「\2", response_text
            )
            fixed = re.sub(
                r'([\u4e00-\u9fa5])"([\u4e00-\u9fa5])', r"\1」\2", fixed
            )
            qa_list = json.loads(fixed)
        except json.JSONDecodeError:
            print(f"⚠️ JSON解析失败: {context}")
            print(f"响应内容: {response_text[:200]}")
            return []

    # 确保返回的是列表格式
    if isinstance(qa_list, dict):
        qa_list = qa_list.get("qa_pairs", [qa_list])

    return qa_list if isinstance(qa_list, list) else []


def generate_qa_pairs(chunk: Dict, qa_count: int = 3) -> List[Dict]:
    """
    用LLM从单个文档片段生成问答对

    Args:
        chunk: {"chunk_id": "DOC-01", "content": "..."}
        qa_count: 建议生成的问题数量

    Returns:
        [{"question": "...", "reference": "...", "chunk_id": "DOC-01"}, ...]
    """
    prompt = f"""你是一个测试工程师，需要根据以下知识库片段生成测试用例。

知识库片段ID: {chunk['chunk_id']}
知识库内容:
{chunk['content']}

请生成 {qa_count} 个测试问题，要求：
1. 问题应该是用户真实会问的问题，口语化表达
2. 答案必须能从知识库片段中找到依据
3. 问题类型要多样化：简单查询、操作步骤、条件判断等
4. 参考答案要准确、简洁，不添加片段外的信息
5. 如果片段内容较少，可以生成少于{qa_count}个问题
6. 【重要】中文内容中不要使用双引号，如需引号请用「」或单引号 ' '，避免破坏 JSON 格式

请严格按照以下JSON格式返回：
[
  {{
    "question": "问题1",
    "reference": "参考答案1"
  }},
  {{
    "question": "问题2",
    "reference": "参考答案2"
  }}
]
"""

    response = get_llm().invoke(prompt)
    response_text = response.content

    qa_list = _parse_llm_json(response_text, chunk["chunk_id"])

    # 添加 chunk_id
    for qa in qa_list:
        qa["chunk_id"] = chunk["chunk_id"]

    return qa_list


def generate_negative_cases(knowledge_base: str, count: int = 5) -> List[Dict]:
    """
    生成库外问题（知识库中不应该有答案的问题）

    Args:
        knowledge_base: 知识库内容
        count: 生成数量

    Returns:
        [{"question": "...", "reference": "..."}, ...]
    """
    # 根据知识库大小决定读取策略
    if len(knowledge_base) <= 3000:
        knowledge_summary = knowledge_base
        summary_note = "完整内容"
    elif len(knowledge_base) <= 10000:
        knowledge_summary = knowledge_base[:3000]
        summary_note = "主要内容（前3000字）"
    else:
        knowledge_summary = (
            knowledge_base[:2000]
            + "\n\n...（中间内容省略）...\n\n"
            + knowledge_base[-1000:]
        )
        summary_note = "核心内容摘要"

    prompt = f"""你是一个测试工程师，需要生成"库外问题"——这些问题不应该从以下知识库中找到答案。

知识库内容{summary_note}:
{knowledge_summary}

请生成 {count} 个库外测试问题，要求：
1. 这些问题明显超出知识库范围（如推荐股票、医疗诊断、法律咨询等）
2. 问题要真实，是用户可能误问的
3. 参考答案应该是拒答或引导话术

请严格按照以下JSON格式返回：
[
  {{
    "question": "问题1",
    "reference": "抱歉，我无法回答该问题，请咨询..."
  }}
]
"""

    response = get_llm().invoke(prompt)
    response_text = response.content

    return _parse_llm_json(response_text, "库外问题")


def generate_multi_hop_cases(documents: List[Dict], count: int = 5) -> List[Dict]:
    """
    生成多跳问题（需要综合多个片段回答）

    Args:
        documents: 文档片段列表
        count: 生成数量

    Returns:
        [{"question": "...", "reference": "...", "relevant_chunk_ids": [...]}, ...]
    """
    prompt = f"""你是一个测试工程师，需要生成"多跳问题"——这些问题需要综合多个知识片段才能回答。

以下是多个知识库片段：
{json.dumps([d['content'][:200] for d in documents[:5]], ensure_ascii=False, indent=2)}

请生成 {count} 个多跳测试问题，要求：
1. 问题需要结合2-3个片段的信息才能完整回答
2. 问题要自然，不要明显看出是拼接的
3. 参考答案要综合多个片段的信息

请严格按照以下JSON格式返回：
[
  {{
    "question": "问题1",
    "reference": "综合回答（需引用多个片段信息）",
    "relevant_chunk_ids": ["DOC-01", "DOC-02"]
  }}
]
"""

    response = get_llm().invoke(prompt)
    response_text = response.content

    return _parse_llm_json(response_text, "多跳问题")


def build_testset(
    knowledge_file: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    qa_per_chunk: int = None,
    negative_count: int = None,
    multi_hop_count: int = None,
    output_file: str = None,
) -> str:
    """
    完整的测试集生成流程

    Args:
        knowledge_file: 知识库文档路径
        chunk_size: 文档切片大小
        chunk_overlap: 切片重叠大小
        qa_per_chunk: 每个片段建议生成的问答对数量
        negative_count: 库外问题建议数量
        multi_hop_count: 多跳问题建议数量
        output_file: 输出文件路径

    Returns:
        测试集文件路径
    """
    # 使用默认值
    from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

    chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
    chunk_overlap = chunk_overlap or DEFAULT_CHUNK_OVERLAP
    qa_per_chunk = qa_per_chunk or DEFAULT_QA_PER_CHUNK
    negative_count = negative_count or DEFAULT_NEGATIVE_COUNT
    multi_hop_count = multi_hop_count or DEFAULT_MULTI_HOP_COUNT

    # 读取并切片文档
    print(f"[INFO] 读取知识库文档: {knowledge_file}")
    content = load_knowledge_base(knowledge_file)
    print(f"[INFO] 文档总字符数: {len(content)}")

    print(
        f"[INFO] 文档分段处理 (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})..."
    )
    documents = split_document(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"[OK] 生成 {len(documents)} 个片段")

    # 1. 生成可答问题
    print(f"[INFO] 生成可答问题（每个片段 {qa_per_chunk} 个）...")
    all_cases: List[Dict] = []

    for i, chunk in enumerate(documents, 1):
        print(f"[{i}/{len(documents)}] 处理片段: {chunk['chunk_id']}")
        qa_pairs = generate_qa_pairs(chunk, qa_count=qa_per_chunk)

        for qa in qa_pairs:
            case = {
                "question": qa["question"],
                "reference": qa["reference"],
                "relevant_ids": [qa["chunk_id"]],
                "type": "可答",
                "priority": "P0" if i <= 3 else "P1",  # 前几个片段优先级更高
            }
            all_cases.append(case)

        print(f"[OK] 已生成 {len(all_cases)} 条")

    # 2. 生成库外问题
    print(f"[INFO] 生成库外问题 {negative_count} 条...")
    negative_cases = generate_negative_cases(content, count=negative_count)
    for case in negative_cases:
        all_cases.append(
            {
                "question": case["question"],
                "reference": case["reference"],
                "relevant_ids": [],
                "type": "库外",
                "priority": "P0",
            }
        )

    # 3. 生成多跳问题
    print(f"[INFO] 生成多跳问题 {multi_hop_count} 条...")
    multi_cases = generate_multi_hop_cases(documents, count=multi_hop_count)
    for case in multi_cases:
        all_cases.append(
            {
                "question": case["question"],
                "reference": case["reference"],
                "relevant_ids": case.get("relevant_chunk_ids", []),
                "type": "多跳",
                "priority": "P1",
            }
        )

    # 4. 回填 contexts
    print("[INFO] 回填 contexts...")
    chunk_map = {d["chunk_id"]: d["content"] for d in documents}
    for case in all_cases:
        ids = case.get("relevant_ids", [])
        case["contexts"] = [chunk_map[i] for i in ids if i in chunk_map]

    # 5. 保存文件
    if output_file is None:
        output_file = os.path.join(get_testcases_dir(), "testset.jsonl")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存测试集
    with open(output_file, "w", encoding="utf-8") as f:
        for case in all_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    # 保存切片文件
    chunks_file = os.path.join(os.path.dirname(output_file), "chunks.jsonl")
    save_chunks(documents, chunks_file)

    # 6. 统计
    print(f"\n[DONE] 测试集生成完成！")
    print(f"   可答问题: {len(all_cases) - negative_count - multi_hop_count} 条")
    print(f"   库外问题: {negative_count} 条")
    print(f"   多跳问题: {multi_hop_count} 条")
    print(f"   总计: {len(all_cases)} 条")
    print(f"   测试集: {output_file}")
    print(f"   切片文件: {chunks_file}")

    return output_file


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_testset(
        knowledge_file=os.path.join(
            script_dir, "testdate", "中华人民共和国数据安全法.docx"
        )
    )