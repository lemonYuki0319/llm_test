# config.py
"""
配置管理：模型选择、API密钥、评分阈值等
"""
import os

# =============================================================================
# MiniMax API 配置
# =============================================================================
MINIMAX_API_KEY = os.getenv(
    "MINIMAX_API_KEY",
    "sk-cp-lBv_rdTB9r_0E0UVw6GStQhZfqCZWx2z_y9GoVmkErjwXN_egYkEGq-ZQ9KI3GTufvoeScdDzgQh5LMrWsybadDwqE5FWPdDW3rzOBMQjqyQBAC8pPb-vVA",
)
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")

# 测试集生成使用的模型
GENERATION_LLM_MODEL = os.getenv("GENERATION_LLM_MODEL", "MiniMax-M3")

# 评估 LLM（M3 易带思考块破坏 JSON 解析，故用 M2.5）
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "MiniMax-M2.5")

# MiniMax 向量模型（原生 texts/type 协议）
EMBED_MODEL = os.getenv("EMBED_MODEL", "embo-01")

# =============================================================================
# Agent API 配置
# =============================================================================
AGENT_BASE_URL = os.getenv("AGENT_BASE_URL", "https://www.whhnhy.com:38868/hy-ai-audit/api")
CREATE_URL = f"{AGENT_BASE_URL}/history/conversation/create"
AGENT_URL = f"{AGENT_BASE_URL}/stream"
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "6137af59f1cc4805a7e08bff2ad06c48")

# =============================================================================
# 文档切片配置
# =============================================================================
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# =============================================================================
# 测试集生成配置
# =============================================================================
DEFAULT_QA_PER_CHUNK = int(os.getenv("QA_PER_CHUNK", "1"))
DEFAULT_NEGATIVE_COUNT = int(os.getenv("NEGATIVE_COUNT", "4"))
DEFAULT_MULTI_HOP_COUNT = int(os.getenv("MULTI_HOP_COUNT", "3"))

# =============================================================================
# 评估配置
# =============================================================================
# 评分阈值
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.8"))
RELEVANCY_THRESHOLD = float(os.getenv("RELEVANCY_THRESHOLD", "0.7"))
CONTEXT_RECALL_THRESHOLD = float(os.getenv("CONTEXT_RECALL_THRESHOLD", "0.7"))

# RAGAS 运行配置
EVAL_TIMEOUT = int(os.getenv("EVAL_TIMEOUT", "180"))
EVAL_MAX_WORKERS = int(os.getenv("EVAL_MAX_WORKERS", "2"))
EVAL_MAX_RETRIES = int(os.getenv("EVAL_MAX_RETRIES", "1"))

# Agent 未返回正文、用例也无金标上下文时的占位，避免 RAGAS 因空 contexts 报错
EMPTY_CONTEXT_PLACEHOLDER = "（无检索到相关知识库内容）"

# =============================================================================
# JSON 输出约束（追加到各指标 prompt，强制模型只吐纯净 JSON）
# =============================================================================
JSON_OUTPUT_CONSTRAINT = """
【输出格式硬性约束——必须严格遵守】
1. 只输出纯净 JSON 对象，前后禁止任何解释、分析、客套话
2. 禁止 markdown 代码块（禁止使用 ``` 或 ```json）
3. 键名必须使用英文，且与示例 schema 完全一致（如 statements / question / reason / verdict / attributed / classifications / noncommittal）
4. 禁止中英文混用键名；禁止中文键名
5. 字符串与键一律使用英文双引号 " ，禁止中文引号「」『』""
6. 分隔符只用英文逗号 , 与英文冒号 : ，禁止中文逗号、、中文冒号：
7. 禁止尾随逗号；括号 [] {} 与引号必须成对匹配
8. 输出必须完整，禁止截断；不要输出注释
"""

EVAL_SYSTEM_HINT = (
    "You are a strict JSON generator for evaluation. "
    "When asked for JSON, output ONLY valid pure JSON. "
    "Do NOT output <think> tags or chain-of-thought. "
    "No markdown fences, no explanations, English keys only, "
    "ASCII commas/colons/double-quotes only, no trailing commas, never truncate."
)

# =============================================================================
# HTTP Headers
# =============================================================================
COMMON_HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json",
    "Origin": "https://www.whhnhy.com:38868",
    "Referer": "https://www.whhnhy.com:38868/audit_agent_ui/agent/legalKnowledge",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
CREATE_HEADERS = {**COMMON_HEADERS, "Accept": "application/json, text/plain, */*"}
STREAM_HEADERS = {
    **COMMON_HEADERS,
    "Accept": "text/event-stream, text/event-stream",
    "Cache-Control": "no-cache",
}

# =============================================================================
# 输出目录配置
# =============================================================================
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
TESTCASES_DIR = os.getenv("TESTCASES_DIR", "testcases")
TESTDATE_DIR = os.getenv("TESTDATE_DIR", "testdate")

# =============================================================================
# 路径辅助函数
# =============================================================================
def get_project_root() -> str:
    """获取项目根目录"""
    return os.path.dirname(os.path.abspath(__file__))


def get_output_dir() -> str:
    """获取输出目录（不存在则创建）"""
    output_dir = os.path.join(get_project_root(), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_testcases_dir() -> str:
    """获取测试用例目录（不存在则创建）"""
    testcases_dir = os.path.join(get_project_root(), TESTCASES_DIR)
    os.makedirs(testcases_dir, exist_ok=True)
    return testcases_dir


def get_testdate_dir() -> str:
    """获取测试数据目录"""
    return os.path.join(get_project_root(), TESTDATE_DIR)
