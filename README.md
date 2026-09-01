# 大模型测试学习手册

> 面向中小企业调用第三方大模型接口场景的测试工程师实战手册
> 适配背景：2年软件测试经验 + 接口自动化 + LangChain/LangGraph基础

---

## 手册定位

本手册严格围绕**中小企业调用第三方大模型接口+业务系统集成**的测试场景，不涉及任何模型自研、训练、微调内容，所有测试方法优先使用已掌握的工具（Requests、Pytest、LangChain、LangGraph），确保可操作、可落地。同时融入大模型评测方法论（Benchmark、lm-eval-harness、RAGAS/DeepEval、red-teaming），覆盖从基础概念到实战项目的完整学习路径。

## 企业AI项目架构

```
用户请求 → 业务系统（前端/后端）
              ↓
         LangGraph编排层（Agent流程控制）
              ↓
         ┌─────┼─────┐
         ↓     ↓     ↓
      RAG检索  工具调用  直接调用LLM
         ↓     ↓     ↓
      向量库  外部API  第三方LLM接口
         ↓           （通义/文心/讯飞/OpenAI）
      知识库文档
```

---

## 目录

| 章节 | 文件 | 核心内容 |
|------|------|---------|
| 第一章 | [第一章_必学基础概念清单.md](第一章_必学基础概念清单.md) | 幻觉、上下文窗口、Prompt、FewShot、RAG、Agent、向量库、鲁棒性、Token、Temperature、Embedding、流式输出、Function Calling、系统提示、Top-P/Top-K、BLEU/ROUGE/BERTScore、LLM-as-a-Judge、CoT、C-Eval/CMMLU/TruthfulQA、Guardrails、LangSmith、Promptfoo/TruLens、TTFT/TPOT、数据漂移、CI/CD，共57个概念，含速查表 |
| 第二章 | [第二章_大模型业务测试全流程.md](第二章_大模型业务测试全流程.md) | 测试计划→测试准备→测试执行→缺陷管理→测试总结，含企业架构总览、Benchmark评测方案、lm-eval-harness/RAGAS/DeepEval工具准备、red-teaming前期准备、业务模型对比规划、7轮测试执行顺序、缺陷等级定义、关键指标 |
| 第三章 | [第三章_AI系统各类测试点及具体测试方法.md](第三章_AI系统各类测试点及具体测试方法.md) | 功能测试、质量评测（LLM输出质量+Benchmark+业务模型对比+LLM-as-a-Judge）、RAG专项测试（RAGAS/DeepEval全维度评估）、Agent流程测试、Prompt测试、安全测试（Red-teaming+间接注入）、性能测试（含TTFT/TPOT）、鲁棒性测试、CoT推理测试、Guardrails测试、可观测性测试、成本优化测试、CI/CD集成测试、数据漂移监控测试、传统NLP指标测试、扩展工具教程（Promptfoo/TruLens/Giskard/PyRIT），19大分类共80+测试点 |
| 第四章 | [第四章_常用测试用例模板.md](第四章_常用测试用例模板.md) | RAG、Agent、Prompt、鲁棒性、安全、性能、功能、质量评测、RAGAS评测、Red-teaming攻防、业务模型对比、Guardrails、CoT推理、成本、可观测性、间接注入、TTFT/TPOT、LLM-as-a-Judge、数据漂移、NLP指标，共20类50+测试用例模板 |
| 第五章 | [第五章_面试高频题+标准答案.md](第五章_面试高频题+标准答案.md) | 40道面试题，覆盖基础概念(6)、测试流程(4)、测试方法(8)、实战场景(4)、LangChain/LangGraph(3)、大模型评测方法(5)、新增概念与工具类(10)，含答题技巧 |
| 第六章 | [第六章_可以写进简历的实战项目思路.md](第六章_可以写进简历的实战项目思路.md) | 3个实战项目：RAG智能客服测试、Agent流程测试、多模型对比评测框架，含项目背景/职责/内容/工具/成果 |
| 第七章 | [第七章_完整可运行代码教程.md](第七章_完整可运行代码教程.md) | 完整可运行代码：项目结构、全局配置、Prompt管理、LLM客户端封装（多模型+流式+Token计数）、RAG辅助工具、自定义断言、测试数据、9类测试用例代码、CI/CD配置、质量阈值检查脚本 |

---

## 大模型测试分类速查

| 测试分类 | 核心目标 | 关键测试点 | 核心工具 |
|---------|---------|-----------|---------|
| 功能测试 | AI回答准确性、业务逻辑正确性 | 准确性、连贯性、适配性、格式规范 | Pytest + Requests |
| 质量评测 | LLM输出质量评估、Benchmark评测、业务模型对比 | 语义相似度、事实一致性、回答忠实度、MMLU/GSM8K/C-Eval、LLM-as-a-Judge | lm-eval-harness + DeepEval + LLM-as-a-Judge |
| 安全测试 | 安全防护能力 | Prompt注入、越狱、信息泄露、违规内容、间接注入 | Red-teaming + Pytest + PyRIT |
| 性能测试 | 性能指标达标 | 响应延迟、TTFT、TPOT、并发、超时、限流 | Requests + 多线程 |
| RAG专项测试 | RAG链路正确性 | 检索准确性、文档匹配、知识库更新、检索召回率、上下文相关性、答案相关性 | RAGAS + DeepEval + LangChain |
| 鲁棒性测试 | 异常场景处理能力 | 异常输入、边界输入、容错、网络异常、无检索结果 | Pytest + Mock |
| Prompt测试 | Prompt有效性 | 话术影响、格式影响、FewShot效果 | Pytest + Promptfoo |
| CoT推理测试 | 推理步骤正确性 | 步骤完整性、逻辑正确性、结论准确性、触发稳定性 | Pytest + LLM |
| Guardrails测试 | 护栏有效性 | 拦截有效性、误拦截率、绕过测试 | NeMo Guardrails + Llama Guard |
| 成本测试 | Token消耗可控 | Token基线、Prompt优化、成本监控 | tiktoken + Pytest |
| 数据漂移测试 | 模型输出稳定性 | 准确率退化、模型版本变更、定期回归 | Pytest + CI/CD |

---

## 快速导航

- **想快速了解基础概念** → 第一章概念速查表（含BLEU/ROUGE/BERTScore、LLM-as-a-Judge、CoT、Guardrails、TTFT/TPOT等57个概念）
- **想了解测试全流程** → 第二章全流程速查图（含评测工具准备和Benchmark适配）
- **想找具体测试方法** → 第三章测试分类总览 + 对应章节（含LLM-as-a-Judge/CoT/Guardrails/间接注入/TTFT-TPOT/数据漂移等方法）
- **想直接用测试用例** → 第四章对应模块的用例模板（含20类50+用例模板）
- **准备面试** → 第五章40道高频题 + 答题技巧（含LLM-as-a-Judge/Guardrails/漂移/CI/CD面试题）
- **写简历** → 第六章3个实战项目 + 简历撰写建议
- **想直接运行代码** → 第七章完整可运行代码教程 + code/目录下的实际代码文件
