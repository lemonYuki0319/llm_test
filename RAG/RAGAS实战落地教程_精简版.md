# RAGAS实战落地教程 — 读知识库 + 结合输出批量打分

> 目标：搭建一套可直接落地的RAG评测流水线。**读取知识库内容**，批量调用RAG接口拿答案，**结合召回的知识库context**，用RAGAS各项指标自动打分，低分case转人工审核。

---

## 目录

1. [快速开始](#一快速开始)
2. [项目目录结构](#二项目目录结构)
3. [核心概念](#三核心概念)
4. [第一步：调用RAG接口](#四第一步调用rag接口)
5. [第二步：配置指标](#五第二步配置指标)
6. [第三步：批量评估](#六第三步批量评估)
7. [完整示例](#七完整示例)
8. [运行与报告](#八运行与报告)
9. [常见问题](#九常见问题)
10. [快速上手清单](#十快速上手清单)

---

## 一、快速开始

### 1.1 安装依赖

```bash
pip install ragas langchain-openai datasets pandas requests
```

### 1.2 配置API密钥

```bash
# RAGAS用的评估模型（Judge，必须≥被测生成模型）
export OPENAI_API_KEY="sk-xxxxxx"

# 你公司RAG项目的接口
export MY_RAG_API_BASE="https://your-project.com/api/rag/query"
export MY_RAG_API_TOKEN="your-token"
```

⚠️ **版本提示**：本教程基于RAGAS **新版API**（`SingleTurnSample` / `EvaluationDataset`）。旧版0.1.x的API名称不同。

---

## 二、项目目录结构

```
rag-eval-project/
├── rag_client.py           # 调用公司RAG接口
├── run_eval.py             # RAGAS批量打分（核心）
├── human_review.py         # 人工审核分流
├── testset.jsonl           # 测试集
├── results/
│   ├── ragas_scores.csv    # RAGAS打分结果
│   └── review_queue.csv    # 待人工审核队列
```

---

## 三、核心概念

RAGAS评估每条样本需要4个字段：

| 字段 | 说明 | 来源 | 哪些指标需要 |
|------|------|------|------------|
| `user_input` | 用户问题 | 测试集 | 全部 |
| `retrieved_contexts` | **知识库召回内容**（list[str]） | RAG接口返回 | Precision/Recall/Faithfulness |
| `response` | **AI模型的答案** | RAG接口返回 | Faithfulness/Relevancy |
| `reference` | 参考答案（ground_truth） | 业务标注 | Recall/Correctness |

📌 **核心逻辑**：RAGAS拿"**知识库context**"当事实基准，检验"**AI项目的response**"。

---

## 四、第一步：调用RAG接口

### 4.1 rag_client.py

```python
# rag_client.py
import os, json, requests

def query_rag(question: str) -> dict:
    """
    调用公司RAG接口，返回答案+召回的知识库内容。
    ⚠️ 按你项目实际的响应结构修改字段路径。
    """
    api_base = os.environ["MY_RAG_API_BASE"]
    token = os.environ.get("MY_RAG_API_TOKEN", "")

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"query": question, "top_k": 4}

    resp = requests.post(api_base, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # ===== 按你项目实际结构修改这两行 =====
    answer = data.get("answer", "")                      # AI模型的答案
    contexts = data.get("retrieved_chunks", [])          # 知识库召回内容(list[str])

    return {"answer": answer, "contexts": contexts}
```

⚠️ **关键提示**：`contexts`必须是`list[str]`，不能是字符串。

---

## 五、第二步：配置指标

### 5.1 测试集格式 testset.jsonl

```json
{"question": "7天退货怎么操作？", "reference": "登录账号进入我的订单申请退货。", "relevant_ids": ["DOC-12"]}
{"question": "帮我推荐一只股票", "reference": "抱歉，无法提供投资建议。", "relevant_ids": [], "type": "库外"}
```

### 5.2 配置评估模型

```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ⚠️ 评估模型必须 >= 被测生成模型能力
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
```

### 5.3 选择指标

```python
from ragas.metrics import (
    LLMContextPrecisionWithReference,  # 上下文精确率
    LLMContextRecall,                  # 上下文召回率
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

💡 **指标选型**：
- 有`reference` → 用Recall + Precision
- 无`reference` → 只用Faithfulness + Relevancy
- 库外拒答用例 → Faithfulness需特殊处理（见常见问题）

---

## 六、第三步：批量评估

### 6.1 run_eval.py（核心）

```python
# run_eval.py
import pandas as pd
from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall, Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rag_client import query_rag
import json

# ===== 阈值配置 =====
THRESHOLDS = {
    "context_recall": 0.7,
    "faithfulness": 0.8,        # 幻觉红线
    "answer_relevancy": 0.7,
}

def main():
    # 1) 评估模型
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
    evaluator_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    metrics = [
        LLMContextRecall(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_emb),
    ]

    # 2) 加载测试集 + 调用RAG接口
    testset = load_testset("testset.jsonl")
    samples = []
    for case in testset:
        rag_out = query_rag(case["question"])
        samples.append(SingleTurnSample(
            user_input=case["question"],
            retrieved_contexts=rag_out["contexts"],
            response=rag_out["answer"],
            reference=case.get("reference", ""),
        ))

    # 3) 批量评估
    eval_dataset = EvaluationDataset(samples)
    results = evaluate(dataset=eval_dataset, metrics=metrics)

    # 4) 保存结果
    df = results.to_pandas()
    df.to_csv("results/ragas_scores.csv", index=False)
    print_summary(df)
    return df

def load_testset(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def print_summary(df):
    total = len(df)
    print(f"\n📊 RAGAS评估报告  共 {total} 条")
    print(f"平均 Faithfulness: {df['faithfulness'].mean():.2f}")
    print(f"平均 Context Recall: {df['context_recall'].mean():.2f}")
```

---

## 七、完整示例

### main.py

```python
# main.py —— 一键跑完整流程
from run_eval import main as run_ragas
from human_review import build_review_queue

if __name__ == "__main__":
    df = run_ragas()
    build_review_queue(df)
    print("✅ 全流程完成 → results/")
```

### human_review.py

```python
# human_review.py
import pandas as pd

def build_review_queue(df, out="results/review_queue.csv"):
    """筛选需人工审核的用例"""
    need_review = df[
        (df["faithfulness"] < 0.8) |  # 幻觉嫌疑
        (df.get("type") == "库外")     # 库外拒答
    ].copy()
    need_review["人工结论"] = ""
    need_review.to_csv(out, index=False)
    print(f"👤 待人工审核 {len(need_review)} 条 → {out}")
```

运行：

```bash
export OPENAI_API_KEY="sk-xxx"
export MY_RAG_API_BASE="https://your-project.com/api/rag/query"
export MY_RAG_API_TOKEN="your-token"

python main.py
```

---

## 八、运行与报告

### 输出示例

```
📊 RAGAS评估报告  共 4 条
平均 Faithfulness: 0.75
平均 Context Recall: 0.80
👤 待人工审核 2 条 → results/review_queue.csv
```

### 结果矩阵

| question | Faithfulness | Context Recall | Answer Relevancy | 归因 |
|----------|:---:|:---:|:---:|---|
| 7天退货操作 | 1.0 | 1.0 | 0.95 | ✅ 检索准+生成实 |
| 保修多久 | 0.5 | 1.0 | 0.9 | ❌ 幻觉(Faithfulness低) |
| 推荐股票 | — | — | 0.3 | ⚠️ 库外-需人工 |

---

## 九、常见问题

| 坑 | 后果 | 解决 |
|------|------|------|
| RAGAS版本不对 | import报错 | 新版用`SingleTurnSample`；旧版0.1.x API不同 |
| `contexts`传成字符串 | 评估报错 | 必须是`list[str]` |
| 库外用例context为空 | 分数异常 | 库外用例单独处理，重点看"是否拒答" |
| 评估模型能力不足 | 打分不可信 | Judge必须≥被测生成模型（铁律） |
| 没有`reference`硬跑Recall | Recall全0 | 无参考答案时只用Faithfulness+Relevancy |
| 知识库版本没锁定 | 指标不可复现 | 知识库打tag，评测集绑定 |

---

## 十、快速上手清单

### 一小时快速上手

1. **安装**（5分钟）
   ```bash
   pip install ragas langchain-openai datasets pandas requests
   ```

2. **修改rag_client.py**（15分钟）
   - 复制本文档的rag_client.py
   - 修改字段路径（改成你项目的实际结构）

3. **准备测试集**（20分钟）
   - 复制本文档的testset.jsonl示例
   - 根据你的业务场景调整内容

4. **配置API密钥**（5分钟）
   ```bash
   export OPENAI_API_KEY="sk-xxx"
   export MY_RAG_API_BASE="your-url"
   export MY_RAG_API_TOKEN="your-token"
   ```

5. **运行测试**（10分钟）
   ```bash
   python main.py
   ```

### 最佳实践清单

```
□ 先跑3-5条小样本验证流程通
□ retrieved_contexts确保是list[str]
□ 评估模型(Judge) >= 被测生成模型
□ 有reference用Recall/Precision；只测幻觉用Faithfulness/Relevancy
□ 库外拒答用例单独判定逻辑
□ 幻觉红线Faithfulness：一般≥0.8，高危≥0.95
□ 低分/幻觉自动进人工审核队列
□ 每月RAGAS vs人工校准，一致性≥85%
□ 知识库快照打tag，与评测集绑定
□ 集成CI自动回归
```

---

> 📎 **一句话总结**：`rag_client.py`负责**调用RAG接口拿答案+context**，RAGAS把"**知识库context**"当事实基准，用`Faithfulness`（防幻觉）、`Context Recall`（检索全不全）等指标对"**AI模型输出**"批量打分，**低分自动转人工审核**。

---

**变更记录**

| 版本 | 日期 | 修改 |
|------|------|------|
| v1.0 | 2026-06-29 | 精简优化，对齐Prompt教程风格 |