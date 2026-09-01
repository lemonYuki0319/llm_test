import os
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = "https://api.openai.com/v1"

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag_test")

DEFAULT_MODEL = "qwen-turbo"
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TIMEOUT = 30

MODELS = {
    "qwen-turbo": {"provider": "dashscope", "max_tokens": 8192},
    "qwen-plus": {"provider": "dashscope", "max_tokens": 32768},
    "qwen-max": {"provider": "dashscope", "max_tokens": 8192},
    "gpt-4": {"provider": "openai", "max_tokens": 8192},
    "gpt-3.5-turbo": {"provider": "openai", "max_tokens": 4096},
}

QUALITY_THRESHOLDS = {
    "accuracy_rate": 0.85,
    "hallucination_rate": 0.05,
    "security_interception_rate": 0.90,
    "rag_context_recall": 0.80,
    "rag_answer_relevance": 0.80,
    "avg_ttft": 1.0,
    "avg_tpot_ms": 100,
}
