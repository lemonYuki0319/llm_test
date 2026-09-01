# evaluator.py
"""
核心评测逻辑（使用RAGAS指标 + LLM-as-Judge）
"""
import json
import os
import re
from types import SimpleNamespace
from typing import List, Dict, Optional

import pandas as pd
import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings.base import embedding_factory
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy
from ragas.metrics._answer_relevance import ResponseRelevancePrompt
from ragas.metrics._context_recall import ContextRecallClassificationPrompt
from ragas.metrics._faithfulness import NLIStatementPrompt, StatementGeneratorPrompt
from ragas.run_config import RunConfig

from config import (
    MINIMAX_API_KEY,
    MINIMAX_BASE_URL,
    EVAL_LLM_MODEL,
    EMBED_MODEL,
    AGENT_URL,
    CREATE_URL,
    CREATE_HEADERS,
    STREAM_HEADERS,
    EMPTY_CONTEXT_PLACEHOLDER,
    JSON_OUTPUT_CONSTRAINT,
    EVAL_SYSTEM_HINT,
    EVAL_TIMEOUT,
    EVAL_MAX_WORKERS,
    EVAL_MAX_RETRIES,
)
from document_processor import load_chunks


# =============================================================================
# MiniMax Embeddings 适配器
# =============================================================================
class _MiniMaxEmbeddingsAPI:
    """实现 OpenAI 风格的 embeddings.create，内部转调 MiniMax 原生接口"""

    def __init__(self, api_key: str, base_url: str, default_model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    def create(self, *, input, model: str | None = None, **kwargs):
        texts = [input] if isinstance(input, str) else list(input)
        if not texts:
            return SimpleNamespace(data=[])

        # 单条按 query，批量按 db（MiniMax 语义）
        embed_type = "query" if len(texts) == 1 else "db"
        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model or self.default_model,
                "texts": texts,
                "type": embed_type,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        status = (data.get("base_resp") or {}).get("status_code", 0)
        if status != 0:
            raise RuntimeError(f"MiniMax embedding 失败: {data.get('base_resp')}")

        vectors = data.get("vectors") or []
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=vec, index=i) for i, vec in enumerate(vectors)
            ]
        )


class MiniMaxOpenAICompatClient:
    """OpenAI SDK 形态的同步 client，仅实现 embeddings.create"""

    def __init__(
        self,
        api_key: str = MINIMAX_API_KEY,
        base_url: str = MINIMAX_BASE_URL,
        model: str = EMBED_MODEL,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.embeddings = _MiniMaxEmbeddingsAPI(api_key, base_url, model)


class LegacyEmbeddingAdapter:
    """
    现代 BaseRagasEmbedding (embed_text/embed_texts)
    -> 旧版指标需要的 embed_query/embed_documents
    """

    def __init__(self, modern_embedding):
        self._emb = modern_embedding

    def embed_query(self, text: str) -> List[float]:
        return self._emb.embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._emb.embed_texts(texts)


# =============================================================================
# JSON 清洗工具
# =============================================================================
_CN_KEY_MAP = {
    "陈述": "statements",
    "语句": "statements",
    "声明": "statements",
    "问题": "question",
    "理由": "reason",
    "原因": "reason",
    "判决": "verdict",
    "判定": "verdict",
    "裁决": "verdict",
    "归因": "attributed",
    "归属": "attributed",
    "分类": "classifications",
    "非承诺": "noncommittal",
    "含糊": "noncommittal",
}


def sanitize_json_text(text: str) -> str:
    """清洗模型输出，尽量得到可 json.loads 的纯净 JSON 字符串"""
    if not text:
        return text

    # 去掉完整/未闭合的思考块
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<think>[\s\S]*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```(?:json|JSON)?\s*", "", text)
    text = text.replace("```", "")

    # 去掉前后解释，优先截取最像目标 schema 的 JSON 段
    candidates = []
    for m in re.finditer(r"[\{\[]", text):
        start = m.start()
        chunk = text[start:]
        end_obj = chunk.rfind("}")
        end_arr = chunk.rfind("]")
        end = max(end_obj, end_arr)
        if end >= 0:
            candidates.append(chunk[: end + 1])

    if candidates:
        preferred_keys = (
            "statements",
            "classifications",
            "noncommittal",
            "question",
            "verdict",
            "attributed",
        )
        ranked = sorted(
            candidates,
            key=lambda c: sum(1 for k in preferred_keys if f'"{k}"' in c),
            reverse=True,
        )
        text = ranked[0]

    # 中文标点 -> 英文
    text = (
        text.replace("，", ",")
        .replace("、", ",")
        .replace("：", ":")
        .replace("；", ";")
        .replace('"', '"')
        .replace('"', '"')
        .replace("'", "'")
        .replace("'", "'")
        .replace("「", '"')
        .replace("」", '"')
        .replace("『", '"')
        .replace("』", '"')
    )

    # 去掉尾随逗号
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # 尝试解析并纠正中文键名
    try:
        data = json.loads(text)
        data = _normalize_keys(data)
        return json.dumps(data, ensure_ascii=False)
    except json.JSONDecodeError:
        return text


def _normalize_keys(obj):
    """递归把中文/混写键名映射到 RAGAS schema 英文键"""
    if isinstance(obj, list):
        return [_normalize_keys(x) for x in obj]

    if isinstance(obj, dict):
        fixed = {}
        for k, v in obj.items():
            nk = _CN_KEY_MAP.get(k, k)
            # 半中英混写时尽量落到英文键
            if nk not in (
                "statements",
                "question",
                "reason",
                "verdict",
                "attributed",
                "classifications",
                "noncommittal",
            ):
                for cn, en in _CN_KEY_MAP.items():
                    if cn in str(k):
                        nk = en
                        break
            fixed[nk] = _normalize_keys(v)
        return fixed

    return obj


# =============================================================================
# JSON 清洗 LLM 包装器
# =============================================================================
class CleanJsonLLMWrapper:
    """
    包装 LangchainLLMWrapper：
    - MiniMax 不支持 n>1，强制按 1 次生成
    - 在 RAGAS 解析前对输出做 sanitize_json_text
    """

    def __init__(self, langchain_llm, **kwargs):
        self._inner = LangchainLLMWrapper(langchain_llm, bypass_n=True, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def generate_text(
        self, prompt, n: int = 1, temperature=0.01, stop=None, callbacks=None
    ):
        result = self._inner.generate_text(
            prompt, n=1, temperature=temperature, stop=stop, callbacks=callbacks
        )
        return self._clean_result(result)

    async def agenerate_text(
        self, prompt, n: int = 1, temperature=0.01, stop=None, callbacks=None
    ):
        result = await self._inner.agenerate_text(
            prompt, n=1, temperature=temperature, stop=stop, callbacks=callbacks
        )
        return self._clean_result(result)

    def _clean_result(self, result):
        for gens in result.generations:
            for g in gens:
                if getattr(g, "text", None):
                    g.text = sanitize_json_text(g.text)
                msg = getattr(g, "message", None)
                if msg is not None and getattr(msg, "content", None):
                    content = msg.content
                    if isinstance(content, str):
                        msg.content = sanitize_json_text(content)
        return result


# =============================================================================
# 带 JSON 约束的 RAGAS 指标 Prompt
# =============================================================================
class StrictStatementGeneratorPrompt(StatementGeneratorPrompt):
    instruction = (
        StatementGeneratorPrompt.instruction
        + "\n"
        + JSON_OUTPUT_CONSTRAINT
        + '\nOutput schema example: {"statements": ["..."]}'
    )


class StrictNLIStatementPrompt(NLIStatementPrompt):
    instruction = (
        NLIStatementPrompt.instruction
        + "\n"
        + JSON_OUTPUT_CONSTRAINT
        + '\nOutput schema example: {"statements": [{"statement":"...","reason":"...","verdict":1}]}'
    )


class StrictResponseRelevancePrompt(ResponseRelevancePrompt):
    instruction = (
        ResponseRelevancePrompt.instruction
        + "\n"
        + JSON_OUTPUT_CONSTRAINT
        + '\nOutput schema example: {"question":"...","noncommittal":0}'
    )


class StrictContextRecallPrompt(ContextRecallClassificationPrompt):
    instruction = (
        ContextRecallClassificationPrompt.instruction
        + "\n"
        + JSON_OUTPUT_CONSTRAINT
        + '\nOutput schema example: {"classifications":[{"statement":"...","reason":"...","attributed":1}]}'
    )


# =============================================================================
# Agent API 调用
# =============================================================================
def create_conversation(title: str = "RAG评估") -> str:
    """创建会话，返回 conversation_id"""
    resp = requests.post(
        CREATE_URL,
        headers=CREATE_HEADERS,
        json={"chat_type": "100", "title": (title or "RAG评估")[:50]},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("code") != 0:
        raise RuntimeError(f"创建会话失败: {body}")

    conversation_id = (body.get("data") or {}).get("conversation_id")
    if not conversation_id:
        raise RuntimeError(f"创建会话未返回 conversation_id: {body}")

    return conversation_id


def _build_payload(question: str, thread_id: str) -> dict:
    """构造与前端一致的 stream 请求体"""
    return {
        "input": {
            "messages": [{"role": "user", "content": question}],
            "task_type": "knowledge",
        },
        "config": {
            "recursion_limit": 100,
            "configurable": {"thread_id": thread_id},
        },
        "stream_mode": ["updates"],
        "stream_subgraphs": True,
    }


def _parse_sse_event(event_line: str) -> Optional[Dict]:
    """解析 SSE 的 data: 行"""
    if not event_line.startswith("data:"):
        return None

    data_str = event_line[len("data:") :].strip()
    if not data_str or data_str == "[DONE]":
        return None

    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return None

    return data if isinstance(data, dict) else None


def _doc_to_context_text(doc: dict) -> Optional[str]:
    """从检索文档对象抽取正文"""
    for key in ("content", "text", "chunk", "page_content", "snippet", "summary"):
        val = doc.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    name = doc.get("file_real_name") or doc.get("file_name") or ""
    url = doc.get("file_url") or ""
    if name or url:
        return f"检索文档: {name}".strip() if name else f"检索文档: {url}"

    return None


def call_rag_agent(question: str, timeout: int = 120) -> Dict:
    """
    先创建会话，再调用流式接口

    Returns:
        {"answer": "...", "contexts": [...], "retrieved_files": [...], "thread_id": "..."}
    """
    thread_id = create_conversation(title=question)
    payload = _build_payload(question, thread_id)

    answer_parts: List[str] = []
    contexts: List[str] = []
    retrieved_files: List[str] = []

    with requests.post(
        AGENT_URL, headers=STREAM_HEADERS, json=payload, stream=True, timeout=timeout
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            event = _parse_sse_event(raw_line)
            if event is None:
                continue

            node = event.get("model") or {}
            if not isinstance(node, dict):
                continue

            for msg in node.get("messages", []) or []:
                if not isinstance(msg, dict):
                    continue

                content = msg.get("content")
                if isinstance(content, str) and content:
                    answer_parts.append(content)

                retrieved = (
                    (msg.get("additional_kwargs") or {}).get("retrieved_documents")
                    or []
                )
                for doc in retrieved:
                    if not isinstance(doc, dict):
                        continue

                    name = doc.get("file_real_name") or doc.get("file_name") or ""
                    if name and name not in retrieved_files:
                        retrieved_files.append(name)

                    text = _doc_to_context_text(doc)
                    if text and text not in contexts:
                        contexts.append(text)

    answer = "".join(answer_parts).strip() or "(无响应内容)"
    return {
        "answer": answer,
        "contexts": contexts,
        "retrieved_files": retrieved_files,
        "thread_id": thread_id,
    }


# =============================================================================
# 测试集加载
# =============================================================================
def _parse_legacy_testset(raw: str) -> List[Dict]:
    """兼容旧版多行 pretty JSON / 空行分隔格式"""
    blocks = [m.group(0) for m in re.finditer(r"\{.*?\}\s*(?=\{|$)", raw, re.DOTALL)]
    cases = []
    for b in blocks:
        try:
            cases.append(json.loads(b))
        except json.JSONDecodeError:
            continue
    return cases


def load_testset(testset_file: str, chunks_file: str | None = None) -> List[Dict]:
    """
    加载标准 JSONL；若失败则回退旧格式。
    若缺 contexts，则按 relevant_ids 从 chunks.jsonl 回填。
    """
    with open(testset_file, "r", encoding="utf-8") as f:
        raw = f.read()

    test_cases: List[Dict] = []
    line_ok = True

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            test_cases.append(json.loads(line))
        except json.JSONDecodeError:
            line_ok = False
            break

    if not line_ok or not test_cases:
        print("[WARN] 标准 JSONL 解析失败，尝试旧格式兼容加载...")
        test_cases = _parse_legacy_testset(raw)

    if chunks_file is None:
        chunks_file = os.path.join(os.path.dirname(testset_file), "chunks.jsonl")

    chunk_map = load_chunks(chunks_file)

    for case in test_cases:
        contexts = case.get("contexts")
        if isinstance(contexts, list) and contexts:
            continue

        ids = case.get("relevant_ids") or []
        filled = [chunk_map[i] for i in ids if i in chunk_map]
        case["contexts"] = filled

    return test_cases


def resolve_contexts(case: Dict, agent_output: Dict) -> List[str]:
    """
    决定送进 RAGAS 的 retrieved_contexts：
      1) Agent 返回的正文
      2) 否则用测试集金标 contexts
      3) 再否则占位字符串
    """
    agent_ctx = [
        c
        for c in (agent_output.get("contexts") or [])
        if isinstance(c, str)
        and c.strip()
        and not c.startswith("检索文档:")  # 仅文件名不足以做忠实度评估
    ]
    if agent_ctx:
        return agent_ctx

    gold_ctx = [
        c
        for c in (case.get("contexts") or [])
        if isinstance(c, str) and c.strip()
    ]
    if gold_ctx:
        return gold_ctx

    return [EMPTY_CONTEXT_PLACEHOLDER]


# =============================================================================
# 评估器构建
# =============================================================================
def build_evaluator():
    """构建评估 LLM（带 system 约束 + JSON 清洗）和三项指标"""

    class SystemPrefixedChat(ChatOpenAI):
        """每次请求前注入「只输出纯 JSON」的 system message"""

        def _prepend_system(self, prompts):
            fixed = []
            for p in prompts:
                if isinstance(p, ChatPromptValue):
                    msgs = list(p.messages)
                    if not (
                        msgs
                        and getattr(msgs[0], "type", "") == "system"
                        and EVAL_SYSTEM_HINT in str(getattr(msgs[0], "content", ""))
                    ):
                        msgs = [SystemMessage(content=EVAL_SYSTEM_HINT)] + msgs
                    fixed.append(ChatPromptValue(messages=msgs))
                elif isinstance(p, StringPromptValue):
                    fixed.append(
                        ChatPromptValue(
                            messages=[
                                SystemMessage(content=EVAL_SYSTEM_HINT),
                                HumanMessage(content=p.text),
                            ]
                        )
                    )
                else:
                    fixed.append(p)
            return fixed

        def generate_prompt(self, prompts, **kwargs):
            return super().generate_prompt(self._prepend_system(prompts), **kwargs)

        async def agenerate_prompt(self, prompts, **kwargs):
            return await super().agenerate_prompt(
                self._prepend_system(prompts), **kwargs
            )

    chat = SystemPrefixedChat(
        model_name=EVAL_LLM_MODEL,
        api_key=MINIMAX_API_KEY,
        base_url=MINIMAX_BASE_URL,
        temperature=0,
        max_retries=2,
        timeout=120,
        max_tokens=2048,  # 避免输出截断
    )

    evaluator_llm = CleanJsonLLMWrapper(chat)

    # Embeddings
    modern_emb = embedding_factory(
        provider="openai",
        model=EMBED_MODEL,
        client=MiniMaxOpenAICompatClient(),
    )
    evaluator_emb = LegacyEmbeddingAdapter(modern_emb)

    metrics = [
        Faithfulness(
            llm=evaluator_llm,
            statement_generator_prompt=StrictStatementGeneratorPrompt(),
            nli_statements_prompt=StrictNLIStatementPrompt(),
        ),
        ResponseRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_emb,
            strictness=1,
            question_generation=StrictResponseRelevancePrompt(),
        ),
        LLMContextRecall(
            llm=evaluator_llm,
            context_recall_prompt=StrictContextRecallPrompt(),
        ),
    ]

    return metrics


def run_evaluation(
    testset_file: str | None = None, max_cases: int | None = None
) -> pd.DataFrame:
    """
    完整评估入口

    Args:
        testset_file: 测试集文件路径
        max_cases: 最大测试用例数

    Returns:
        评估结果 DataFrame
    """
    from report_builder import (
        build_report_dataframe,
        save_evaluation_report,
        print_summary,
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if testset_file is None:
        testset_file = os.path.join(script_dir, "testcases", "testset.jsonl")

    test_cases = load_testset(testset_file)
    if max_cases is not None:
        test_cases = test_cases[:max_cases]

    print(f"[INFO] 加载测试集: {len(test_cases)} 条 ({testset_file})", flush=True)
    if not test_cases:
        raise RuntimeError("测试集为空，请先运行 testset_generator.py 生成标准 JSONL")

    out_dir = os.path.dirname(testset_file) or script_dir
    agent_cache_path = os.path.join(out_dir, "agent_outputs.jsonl")
    samples = []
    results = []

    # 边调 Agent 边落盘
    with open(agent_cache_path, "w", encoding="utf-8") as cache_f:
        for i, case in enumerate(test_cases, 1):
            question = case.get("question", "")
            print(f"\n[{i}/{len(test_cases)}] 处理: {question}", flush=True)

            try:
                agent_output = call_rag_agent(question)
                print(
                    f"[OK] thread_id={agent_output['thread_id']} "
                    f"files={len(agent_output.get('retrieved_files') or [])}",
                    flush=True,
                )
            except Exception as e:
                print(f"[ERROR] 调用Agent失败: {e}", flush=True)
                agent_output = {
                    "answer": "(调用失败)",
                    "contexts": [],
                    "retrieved_files": [],
                    "thread_id": "",
                }

            retrieved_contexts = resolve_contexts(case, agent_output)
            sample = SingleTurnSample(
                user_input=question,
                retrieved_contexts=retrieved_contexts,
                response=agent_output["answer"],
                reference=case.get("reference", ""),
            )
            samples.append(sample)

            row = {
                "question": question,
                "type": case.get("type", ""),
                "priority": case.get("priority", ""),
                "expected": case.get("reference", ""),
                "actual": agent_output["answer"],
                "contexts": retrieved_contexts,
                "retrieved_files": agent_output.get("retrieved_files", []),
                "relevant_ids": case.get("relevant_ids", []),
                "thread_id": agent_output.get("thread_id", ""),
            }
            results.append(row)
            cache_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            cache_f.flush()

    print("\n[INFO] 开始RAGAS评估...", flush=True)
    metrics = build_evaluator()

    # 降低并发，拉长超时
    run_config = RunConfig(
        timeout=EVAL_TIMEOUT, max_workers=EVAL_MAX_WORKERS, max_retries=EVAL_MAX_RETRIES
    )
    eval_results = evaluate(
        dataset=EvaluationDataset(samples),
        metrics=metrics,
        run_config=run_config,
    )

    score_df = eval_results.to_pandas()
    report_df = build_report_dataframe(results, score_df)

    report_path = os.path.join(out_dir, "evaluation_report.txt")
    detail_path = os.path.join(out_dir, "evaluation_details.jsonl")

    save_evaluation_report(report_df, report_path)
    with open(detail_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[DONE] 评估完成！报告: {report_path}", flush=True)
    print(f"[DONE] 明细: {detail_path}", flush=True)
    print(f"[DONE] Agent缓存: {agent_cache_path}", flush=True)
    print_summary(report_df)

    return report_df