# stream_request.py
"""
SSE 流式接口调用（极简版）
"""
import json
import requests

# 配置
BASE_URL = "https://www.whhnhy.com:38868/hy-ai-audit/api"
AUTH_TOKEN = "6137af59f1cc4805a7e08bff2ad06c48"
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json",
    "Host": "www.whhnhy.com:38868",
}


def create_conversation() -> str:
    """创建会话，返回 conversation_id"""
    resp = requests.post(
        f"{BASE_URL}/history/conversation/create",
        headers=HEADERS,
        json={"chat_type": "100", "title": "测试"},
        timeout=30,
    )
    return resp.json()["data"]["conversation_id"]


def stream_chat(question: str) -> str:
    """流式提问，返回完整回答"""
    thread_id = create_conversation()

    payload = {
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

    all_events = []
    received_end = False  # 是否收到 end 事件

    # stream=True 保持连接，逐行读取，先全部收完
    with requests.post(
        f"{BASE_URL}/stream",
        headers=HEADERS,
        json=payload,
        stream=True,
        timeout=120,
    ) as resp:
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue

            # 检测 event: end 事件
            if line.startswith("event:"):
                if line[6:].strip() == "end":
                    received_end = True
                    break
                continue

            if not line.startswith("data:"):
                continue

            data_str = line[5:].strip()
            if data_str in ("[DONE]", "null"):
                continue

            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            all_events.append(event)

    # 必须收到 end 事件才提取，防止没输出完就读了
    if not received_end:
        print("[警告] 未收到 end 事件，响应可能不完整")

    # 全部收完后，从事件列表中找 model 节点，直接取 content
    for event in all_events:
        model = event.get("model")
        if isinstance(model, dict):
            messages = model.get("messages", [])
            if messages and messages[0].get("content"):
                return messages[0]["content"]

    return ""


if __name__ == "__main__":
    question = "数据安全法里说的「数据安全」是什么意思？"
    print(f"[问题] {question}\n{'='*60}\n[回答] ")

    # 提取内容赋值给 llm_answer
    llm_answer = stream_chat(question)

    print(f"\n{'='*60}")
    print(f"[变量 llm_answer] {llm_answer[:200]}...")
