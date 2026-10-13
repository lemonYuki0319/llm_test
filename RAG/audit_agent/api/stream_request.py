# stream_request.py
"""SSE 流式接口:登录 → 创建会话 → 流式提问 → 返回完整回答"""
import json

import requests

from auth import login

API_BASE = "https://www.whhnhy.com:38868/hy-ai-audit/api"
HEADERS = {
    "Authorization": f"Bearer {login('16610000003', 'Aa123456')}",
    "Content-Type": "application/json",
}


def stream_chat(question: str) -> str:
    """流式提问,返回完整回答"""
    # 1. 创建会话
    resp = requests.post(
        f"{API_BASE}/history/conversation/create",
        headers=HEADERS,
        json={"chat_type": "100", "title": "测试"},
        timeout=30,
    )
    thread_id = resp.json()["data"]["conversation_id"]

    # 2. 流式调用
    payload = {
        "input": {
            "messages": [{"role": "user", "content": question}],
            "task_type": "knowledge",
        },
        "config": {"recursion_limit": 100, "configurable": {"thread_id": thread_id}},
        "stream_mode": ["updates"],
        "stream_subgraphs": True,
    }

    events = []
    with requests.post(
        f"{API_BASE}/stream", headers=HEADERS, json=payload, stream=True, timeout=120
    ) as resp:
        for line in resp.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                try:
                    events.append(json.loads(line[5:].strip()))
                except json.JSONDecodeError:
                    pass

    # 3. 从 model 节点提取回答
    for event in events:
        model = event.get("model")
        if isinstance(model, dict):
            messages = model.get("messages", [])
            if messages and messages[0].get("content"):
                return messages[0]["content"]

    return ""


if __name__ == "__main__":
    answer = stream_chat("数据安全法里说的「数据安全」是什么意思?")
    print(answer)
