code/
├── config/                  # 配置层
│   ├── settings.py          # API密钥(从环境变量读取)、模型配置、质量阈值
│   └── prompts.py           # 系统提示词、LLM-as-Judge评审Prompt、CoT模板
├── conftest.py              # pytest全局fixtures(LLM客户端、测试数据加载)
├── pytest.ini               # pytest配置(标记:smoke/regression/security等)
├── requirements.txt         # 依赖(pytest/openai/langchain/ragas/deepeval等)
├── scripts/
│   └── check_quality_threshold.py  # CI/CD质量门禁脚本(不达标则阻断)
├── testcases/               # 9类测试用例
│   ├── test_functional.py   # 功能测试(金标准集准确性、上下文连贯、JSON格式)
│   ├── test_quality.py      # 质量评测(LLM-as-Judge、BERTScore、ROUGE)
│   ├── test_security.py     # 安全测试(注入、越狱、信息泄露、间接注入)
│   ├── test_performance.py  # 性能测试(延迟、TTFT、TPOT、并发)
│   ├── test_robustness.py   # 鲁棒性测试(空白/乱码/错别字/对抗输入)
│   ├── test_cost.py         # 成本测试(Token基线、Prompt优化)
│   ├── test_guardrails.py   # 护栏测试(话题拦截、误拦截)
│   ├── test_cot.py          # 推理链测试(步骤完整性、逻辑正确、稳定性)
│   └── test_drift.py        # 数据漂移测试(准确率退化监控)
├── testdata/                # 测试数据(golden_set/security/robustness/cot的JSON)
└── utils/                   # 工具层
    ├── llm_client.py        # LLM客户端封装(多模型、流式、Token计数)
    ├── rag_helper.py        # RAG辅助(FAISS向量库、RetrievalQA链)
    └── assertion_helper.py  # 自定义断言(BERTScore/ROUGE/BLEU/幻觉检测)