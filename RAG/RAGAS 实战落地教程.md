# 文档二：RAGAS 实战落地教程 — 读知识库 + 结合输出批量打分

> 目标：搭建一套可直接落地的 RAG 评测流水线。**读取知识库内容**，批量调用公司 AI 项目的 RAG 接口拿到答案，**结合召回的知识库 context**，用 RAGAS 各项指标自动打分，低分/有异议的**转人工审核**，产出报告。

---

## 目录

1. [快速开始](#一快速开始-1)
2. [项目目录结构](#二项目目录结构-1)
3. [核心概念：RAGAS 数据模型](#三核心概念ragas-数据模型)
4. [第一步：读取知识库 + 调用 RAG 接口采集数据](#四第一步读取知识库--调用-rag-接口采集数据)
5. [第二步：配置 RAGAS 指标与评估模型](#五第二步配置-ragas-指标与评估模型)
6. [第三步：批量评估打分](#六第三步批量评估打分)
7. [第四步：人工审核分流](#七第四步人工审核分流)
8. [完整可运行示例](#八完整可运行示例)
9. [运行与报告](#九运行与报告)
10. [常见问题与最佳实践](#十常见问题与最佳实践)

---

## 一、快速开始

### 1.1 安装依赖

```bash
# RAGAS 核心 + LangChain 集成 + 数据处理
pip install ragas langchain-openai datasets pandas

# 如果用本地向量库读知识库（可选）
pip install langchain-community faiss-cpu
```

### 1.2 配置 API 密钥

```bash
# RAGAS 内部用的评估模型（Judge，必须≥被测生成模型）
export OPENAI_API_KEY="sk-xxxxxx"

# 你公司 RAG 项目的接口
export MY_RAG_API_BASE="https://your-project.com/api/rag/query"
export MY_RAG_API_TOKEN="your-token"
```

> ⚠️ **版本提示**：本教程基于 RAGAS **新版 API**（`SingleTurnSample` / `EvaluationDataset` / `evaluate`）。请用 `pip show ragas` 确认版本，旧版 `0.1.x` 的 API 名称不同。

---

## 二、项目目录结构

```bash
rag-eval-project/
├── kb_loader.py            # 读取知识库内容
├── rag_client.py           # 调用公司 RAG 接口采集(question,context,answer)
├── build_dataset.py        # 构建 RAGAS 评估数据集
├── run_eval.py             # RAGAS 批量打分（核心）
├── human_review.py         # 人工审核分流
├── testset.jsonl           # 测试集（question + ground_truth + 相关片段ID）
├── results/
│   ├── ragas_scores.csv    # RAGAS 打分结果
│   └── review_queue.csv    # 待人工审核队列
└── .env
```

---

## 三、核心概念：RAGAS 数据模型

RAGAS 评估**每条样本**需要 4 个字段，正好对应 RAG 链路：

**RAGAS数据模型流程**：
```
user_input(用户问题) + retrieved_contexts(知识库召回内容) + response(AI项目的答案) + reference(参考答案)
→ SingleTurnSample(单条评估样本)
→ EvaluationDataset(评估数据集)
→ evaluate()(批量打分)
```

| 字段 | 说明 | 来源 | 哪些指标需要 |
|------|------|------|------------|
| `user_input` | 用户问题 | 测试集 | 全部 |
| `retrieved_contexts` | **知识库召回的内容**（list[str]） | RAG 接口返回 | Precision/Recall/Faithfulness |
| `response` | **公司 AI 模型的答案** | RAG 接口返回 | Faithfulness/Relevancy |
| `reference` | 参考答案（ground_truth） | 业务标注 | Recall/Correctness |

> 📌 **核心逻辑**：RAGAS 拿"**知识库 context**"当事实基准，去检验"**AI 项目的 response**"——这正是您要的"读知识库内容 + 结合模型输出打分"。

---

## 四、第一步：读取知识库 + 调用 RAG 接口采集数据

### 4.1 读取知识库内容 `kb_loader.py`

> 两种场景：① RAG 接口直接返回 context（最常见，推荐）；② 需要自己从向量库检索。

```python
# kb_loader.py
"""
方式A：如果你公司RAG接口在返回答案时，同时返回了召回的知识片段，
      直接从接口响应里取 retrieved_contexts 即可（见 rag_client.py），
      本文件可跳过。

方式B：如果需要独立从向量知识库检索，用下面的代码。
"""
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def load_kb(index_path="./kb_index"):
    """加载已构建的向量知识库"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(index_path, embeddings,
                                   allow_dangerous_deserialization=True)
    return vectorstore

def retrieve_context(vectorstore, query: str, top_k: int = 4) -> list[str]:
    """从知识库检索 top_k 个相关片段，返回纯文本列表"""
    docs = vectorstore.similarity_search(query, k=top_k)
    return [doc.page_content for doc in docs]
```

### 4.2 调用公司 RAG 接口 `rag_client.py`

```python
# rag_client.py
import os
import json
import requests

def query_rag(question: str) -> dict:
    """
    调用公司 RAG 接口，返回 答案 + 召回的知识库内容。
    ⚠️ 按你项目实际的响应结构修改字段路径。
    """
    api_base = os.environ["MY_RAG_API_BASE"]
    token = os.environ.get("MY_RAG_API_TOKEN", "")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"query": question, "top_k": 4}

    resp = requests.post(api_base, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # ===== 按你项目实际结构修改这两行 =====
    answer = data.get("answer", "")                          # AI模型的答案
    contexts = data.get("retrieved_chunks", [])              # 知识库召回内容(list[str])
    # 若 contexts 是对象列表，提取文本：
    # contexts = [c["content"] for c in data.get("retrieved_chunks", [])]

    return {"answer": answer, "contexts": contexts}


def query_rag_stream(question: str) -> dict:
    """
    如果你的 RAG 接口是流式(SSE)的，用这个版本。
    先拼接流式答案，context 通常在首个或末个 event 里。
    """
    api_base = os.environ["MY_RAG_API_BASE"]
    token = os.environ.get("MY_RAG_API_TOKEN", "")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {"query": question, "top_k": 4, "stream": True}

    full_answer = ""
    contexts = []
    with requests.post(api_base, headers=headers, json=payload,
                       stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                # 召回内容通常随首个 event 下发（按实际结构改）
                if "retrieved_chunks" in chunk:
                    contexts = chunk["retrieved_chunks"]
                # 增量答案
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                full_answer += delta or ""
            except json.JSONDecodeError:
                full_answer += data_str

    return {"answer": full_answer, "contexts": contexts}
```

---

## 五、第二步：配置 RAGAS 指标与评估模型

### 5.1 测试集格式 `testset.jsonl`

```json
{"question": "7天无理由退货怎么操作？", "reference": "登录账号进入我的订单，7天内点击申请退货并按提示寄回商品。", "relevant_ids": ["DOC-12", "DOC-13"], "type": "可答", "priority": "P0"}
{"question": "这款产品保修多久？", "reference": "整机保修1年。", "relevant_ids": ["DOC-20"], "type": "可答", "priority": "P0"}
{"question": "帮我推荐一只股票", "reference": "抱歉，我无法提供投资建议，仅能回答产品售后相关问题。", "relevant_ids": [], "type": "库外", "priority": "P0"}
{"question": "海外购买的产品能退货吗？", "reference": "海外购买需联系海外客服，15天内申请退货。", "relevant_ids": ["DOC-30", "DOC-45"], "type": "多跳", "priority": "P1"}
```

### 5.2 配置评估模型（Judge）

```python
# 在 run_eval.py 顶部
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ⚠️ 评估模型必须 >= 被测生成模型能力（对齐 Judge 铁律）
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
```

### 5.3 选择指标

```python
from ragas.metrics import (
    LLMContextPrecisionWithReference,  # 上下文精确率（需 reference）
    LLMContextRecall,                  # 上下文召回率（需 reference）
    Faithfulness,                      # 忠实度（防幻觉核心）
    ResponseRelevancy,                 # 答案相关性
)

metrics = [
    LLMContextPrecisionWithReference(llm=evaluator_llm),
    LLMContextRecall(llm=evaluator_llm),
    Faithfulness(llm=evaluator_llm),
    ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
]
```

> 💡 **指标选型建议**：
> - 有 `reference`（参考答案）→ 用 `LLMContextRecall` + `LLMContextPrecisionWithReference`
> - 无 `reference`（只想测幻觉）→ 只用 `Faithfulness` + `ResponseRelevancy`（无需 ground_truth）
> - 库外拒答用例 → `retrieved_contexts` 可能为空，`Faithfulness` 需特殊处理（见常见问题）

---

## 六、第三步：批量评估打分

### 6.1 构建评估数据集 `build_dataset.py`

```python
# build_dataset.py
import json
from ragas import SingleTurnSample, EvaluationDataset
from rag_client import query_rag

def load_testset(path="testset.jsonl") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def build_eval_dataset(testset: list[dict]) -> tuple[EvaluationDataset, list[dict]]:
    """
    对每条测试用例：调用公司RAG接口拿 答案+知识库context，
    组装成 RAGAS 的 SingleTurnSample。
    """
    samples = []
    meta = []   # 保留优先级/类型等元数据，供后续人工审核分流
    for case in testset:
        # ===== 调用公司 RAG 接口 =====
        rag_out = query_rag(case["question"])

        sample = SingleTurnSample(
            user_input=case["question"],
            retrieved_contexts=rag_out["contexts"],   # 知识库召回内容
            response=rag_out["answer"],               # AI模型答案
            reference=case.get("reference", ""),      # 参考答案
        )
        samples.append(sample)
        meta.append({
            "question": case["question"],
            "type": case.get("type", ""),
            "priority": case.get("priority", ""),
            "answer": rag_out["answer"],
            "contexts": rag_out["contexts"],
        })

    return EvaluationDataset(samples), meta
```

### 6.2 执行评估 `run_eval.py`（核心）

```python
# run_eval.py
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    LLMContextPrecisionWithReference, LLMContextRecall,
    Faithfulness, ResponseRelevancy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from build_dataset import load_testset, build_eval_dataset

# ===== 阈值配置（对齐 RAG 团队教程第十节）=====
THRESHOLDS = {
    "llm_context_precision_with_reference": 0.7,
    "context_recall": 0.7,
    "faithfulness": 0.8,        # 幻觉红线
    "answer_relevancy": 0.7,
}

def main():
    # 1) 评估模型（Judge >= 被测模型）
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
    evaluator_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    metrics = [
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_emb),
    ]

    # 2) 采集数据 + 构建数据集
    testset = load_testset("testset.jsonl")
    eval_dataset, meta = build_eval_dataset(testset)

    # 3) 批量评估
    print(f"开始评估 {len(testset)} 条用例 ...")
    results = evaluate(dataset=eval_dataset, metrics=metrics)

    # 4) 转 DataFrame + 合并元数据
    df = results.to_pandas()
    meta_df = pd.DataFrame(meta)
    df["type"] = meta_df["type"]
    df["priority"] = meta_df["priority"]

    # 5) 逐条判定 + 归因
    df["verdict"], df["attribution"] = zip(*df.apply(judge_row, axis=1))

    df.to_csv("results/ragas_scores.csv", index=False, encoding="utf-8-sig")
    print_summary(df)
    return df


def judge_row(row):
    """综合判定 + 失败归因（对齐 RAG 团队教程归因决策表）"""
    recall = row.get("context_recall", 1)
    faith = row.get("faithfulness", 1)
    relevancy = row.get("answer_relevancy", 1)
    precision = row.get("llm_context_precision_with_reference", 1)

    # 库外拒答用例：应无召回、应拒答，单独判定
    if row.get("type") == "库外":
        # 拒答场景 Faithfulness 可能异常，重点看是否乱答
        return ("⚠️ 需人工确认拒答", "库外-需人工")

    # 归因决策表
    if pd.notna(recall) and recall < THRESHOLDS["context_recall"]:
        return ("❌ 失败", "检索漏召(Recall低)")
    if pd.notna(faith) and faith < THRESHOLDS["faithfulness"]:
        return ("❌ 失败", "幻觉(Faithfulness低)")
    if pd.notna(precision) and precision < THRESHOLDS["llm_context_precision_with_reference"]:
        return ("⚠️ 警告", "检索噪声(Precision低)")
    if pd.notna(relevancy) and relevancy < THRESHOLDS["answer_relevancy"]:
        return ("⚠️ 警告", "跑题(Relevancy低)")
    return ("✅ 通过", "检索准+生成实")


def print_summary(df):
    total = len(df)
    passed = (df["verdict"] == "✅ 通过").sum()
    failed = df["verdict"].str.startswith("❌").sum()
    warned = df["verdict"].str.startswith("⚠️").sum()
    print("\n" + "=" * 40)
    print(f"📊 RAGAS 评估报告  共 {total} 条")
    print(f"✅ 通过: {passed} ({passed/total:.0%})")
    print(f"⚠️ 警告/待复核: {warned}")
    print(f"❌ 失败: {failed}")
    print(f"平均 Faithfulness: {df['faithfulness'].mean():.2f}")
    print(f"平均 Context Recall: {df['context_recall'].mean():.2f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
```

---

## 七、第四步：人工审核分流

对齐"**有异议的人工审核**"要求：把低分、幻觉嫌疑、库外、RAGAS 与规则冲突的用例，自动挑到审核队列。

```python
# human_review.py
import pandas as pd

def build_review_queue(df: pd.DataFrame, out="results/review_queue.csv"):
    """
    筛选需人工审核的用例（对齐三层评估 Layer 3）：
      ① 任一核心指标不达标（低分 case）
      ② Faithfulness 低（幻觉嫌疑）→ 必查
      ③ 库外用例 → 必查是否正确拒答
      ④ P0 用例 → 抽样必查
      ⑤ 判定为失败/警告的
    """
    need_review = df[
        (df["verdict"].str.startswith("❌")) |
        (df["verdict"].str.startswith("⚠️")) |
        (df["faithfulness"] < 0.8) |
        (df["type"] == "库外") |
        (df["priority"] == "P0")
    ].copy()

    # 附上人工复核字段（待填）
    need_review["人工结论"] = ""       # 通过/失败
    need_review["人工归因"] = ""       # 与RAGAS是否一致
    need_review["是否与RAGAS冲突"] = ""  # 用于月度校准

    cols = ["question", "type", "priority", "answer", "faithfulness",
            "context_recall", "answer_relevancy", "verdict", "attribution",
            "人工结论", "人工归因", "是否与RAGAS冲突"]
    need_review[cols].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"👤 待人工审核 {len(need_review)} 条 → {out}")
    return need_review
```

### 人工审核 → 月度校准闭环

**人工审核→月度校准闭环**：
```
RAGAS自动打分 → 判断：低分/幻觉/库外/P0？
  是 → 进入人工审核队列 → 人工判定(填写结论+是否冲突)
  否 → ✅自动通过

人工判定 → 每月汇总(RAGAS vs 人工一致性)
→ 判断：一致性≥85%？
  是 → 指标可信,继续
  否 → 重设计指标Prompt/阈值
```

---

## 八、完整可运行示例

把整个流程串起来的入口脚本：

```python
# main.py —— 一键跑完整流程
from run_eval import main as run_ragas
from human_review import build_review_queue

if __name__ == "__main__":
    # Step 1-3: 读知识库+调接口+RAGAS批量打分
    df = run_ragas()

    # Step 4: 人工审核分流
    build_review_queue(df)

    print("\n✅ 全流程完成：")
    print("   - RAGAS打分   → results/ragas_scores.csv")
    print("   - 人工审核队列 → results/review_queue.csv")
```

运行：

```bash
export OPENAI_API_KEY="sk-xxx"
export MY_RAG_API_BASE="https://your-project.com/api/rag/query"
export MY_RAG_API_TOKEN="your-token"

python main.py
```

---

## 九、运行与报告

### 9.1 输出示例

```
开始评估 4 条用例 ...
Evaluating: 100%|████████████| 4/4

========================================
📊 RAGAS 评估报告  共 4 条
✅ 通过: 2 (50%)
⚠️ 警告/待复核: 1
❌ 失败: 1
平均 Faithfulness: 0.75
平均 Context Recall: 0.80
========================================
👤 待人工审核 3 条 → results/review_queue.csv
```

### 9.2 结果矩阵（ragas_scores.csv）

| question | type | Ctx Precision | Ctx Recall | Faithfulness | Ans Relevancy | verdict | attribution |
|----------|------|:---:|:---:|:---:|:---:|:---:|---|
| 7天退货操作 | 可答 | 1.0 | 1.0 | 1.0 | 0.95 | ✅ 通过 | 检索准+生成实 |
| 保修多久 | 可答 | 0.5 | 1.0 | 0.5 | 0.9 | ❌ 失败 | 幻觉(Faithfulness低) |
| 推荐股票 | 库外 | 0.0 | — | — | 0.3 | ⚠️ 需人工确认拒答 | 库外-需人工 |
| 海外退货 | 多跳 | 0.5 | 0.5 | 1.0 | 0.6 | ❌ 失败 | 检索漏召(Recall低) |

### 9.3 集成 CI（每次知识库/Prompt 变更自动回归）

```yaml
# .github/workflows/rag-eval.yml
name: RAG回归评测
on: [push, pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: 安装依赖
        run: pip install ragas langchain-openai datasets pandas requests
      - name: 运行RAGAS评测
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          MY_RAG_API_BASE: ${{ secrets.MY_RAG_API_BASE }}
          MY_RAG_API_TOKEN: ${{ secrets.MY_RAG_API_TOKEN }}
        run: python main.py
      - name: 上传报告
        uses: actions/upload-artifact@v4
        with:
          name: ragas-report
          path: results/
```

---

## 十、常见问题与最佳实践

### 10.1 常见坑与解决

| 坑 | 后果 | 解决 |
|------|------|------|
| RAGAS 版本 API 不对 | import 报错 | 新版用 `SingleTurnSample`；旧版 0.1.x API 不同，先 `pip show ragas` |
| `retrieved_contexts` 传成字符串 | 评估报错 | 必须是 `list[str]` |
| 库外用例 context 为空跑 Faithfulness | 分数异常/报错 | 库外用例单独处理，重点看"是否拒答"而非算 Faithfulness |
| 评估模型能力不足 | 打分不可信 | Judge 必须 ≥ 被测生成模型（铁律） |
| 没有 ground_truth 硬跑 Recall | Recall 全 0 | 无参考答案时改用 `Faithfulness`+`ResponseRelevancy` |
| 一次性评估几千条 | 超时/费用高 | 分批 + 设并发；先小样本验证再全量 |
| 知识库版本没锁定 | 指标不可复现 | 知识库打 tag，评测集与之绑定 |
| 完全信任 RAGAS 不校准 | 偏差累积 | 每月人工 vs RAGAS 校准，<85% 重设计 |

### 10.2 最佳实践清单

```
□ 先跑 3-5 条小样本验证流程通，再全量
□ retrieved_contexts 确保是 list[str]，从接口正确提取
□ 评估模型(Judge) >= 被测生成模型能力
□ 有 ground_truth 用 Recall/Precision；只测幻觉用 Faithfulness/Relevancy
□ 库外拒答用例单独判定逻辑（不硬算 Faithfulness）
□ 幻觉红线 Faithfulness：一般≥0.8，高危业务≥0.95
□ 低分/幻觉/库外/P0 自动进人工审核队列
□ 每月 RAGAS vs 人工校准，一致性≥85%
□ 知识库快照打 tag，与评测集绑定
□ 分批评估控制成本，集成 CI 自动回归
```

---

## 附录：RAGAS 完整流程回顾

**RAGAS完整流程**：
```
① 读知识库/测试集(testset.jsonl)
② 调用公司RAG接口(拿answer + contexts)
③ 组装SingleTurnSample(question/context/answer/reference)
④ 配置指标+Judge模型(gpt-4o)
⑤ evaluate()批量打分(Precision/Recall/Faithfulness/Relevancy)
⑥ 归因判定(检索错 or 生成错)
⑦ 低分/幻觉/库外 → 进人工审核队列
⑧ 月度校准(RAGAS vs 人工)
```

---

> 📎 **一句话总结**：`rag_client.py` 负责**读知识库并调用公司 RAG 接口**拿到"答案 + 召回内容"，RAGAS 把"**知识库 context**"当事实基准，用 `Faithfulness`（防幻觉）、`Context Recall`（检索全不全）、`Answer Relevancy`（答没答到点）等指标对"**AI 模型输出**"批量打分，**低分/幻觉/有异议的自动转人工审核**，月度校准形成闭环。

---

以上两份文档已完成。两份文件均**严格对齐**您现有的《Prompt 测试团队教程》和《Promptfoo 教程》风格：

| 对齐点 | 体现 |
|--------|------|
| **三层评估体系** | RAG 版：自动指标 → RAGAS/LLM → 人工复核 |
| **Judge ≥ 被测模型铁律** | RAGAS 评估模型用 gpt-4o |
| **每月校准 <85% 重设计** | 人工审核 → 月度校准闭环 |
| **五阶段流程 + 失败归因** | 新增 RAG 专属归因（检索错/生成错） |
| **测试用例模板 + 上线清单** | 全部适配 RAG 场景 |
| **代码可直接跑** | 读知识库 + 调接口 + RAGAS 打分 + 人工分流 |

需要我把这两份文档**分别导出为 `.md` 文件**，或再补一份**《RAG vs Prompt 测试差异速查卡》**方便团队快速区分吗？