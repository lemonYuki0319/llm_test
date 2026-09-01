import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.llm_client import LLMClient
from config.prompts import SYSTEM_PROMPTS


class TestPerformance:
    def test_response_latency(self, llm_client):
        latencies = []
        for _ in range(10):
            result = llm_client.simple_chat(
                "退货政策是什么？",
                system_prompt=SYSTEM_PROMPTS["customer_service"]
            )
            latencies.append(result["latency"])

        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        max_latency = max(latencies)

        print(f"响应延迟 - 平均: {avg_latency:.2f}s, P95: {p95_latency:.2f}s, 最大: {max_latency:.2f}s")
        assert avg_latency < 5.0, f"平均响应时间{avg_latency:.2f}s超过5s阈值"

    def test_ttft(self, llm_client):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS["customer_service"]},
            {"role": "user", "content": "退货政策是什么？"}
        ]
        result = llm_client.chat_stream(messages)
        print(f"TTFT: {result['ttft']:.3f}s, TPOT: {result['avg_tpot_ms']:.1f}ms/Token")
        assert result["ttft"] < 2.0, f"TTFT {result['ttft']:.3f}s超过2s阈值"

    def test_tpot(self, llm_client):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS["customer_service"]},
            {"role": "user", "content": "详细说明退货流程"}
        ]
        result = llm_client.chat_stream(messages)
        print(f"TPOT: {result['avg_tpot_ms']:.1f}ms/Token, 总Token数: {result['token_count']}")
        assert result["avg_tpot_ms"] < 150, f"TPOT {result['avg_tpot_ms']:.1f}ms超过150ms阈值"

    def test_concurrent_requests(self):
        def send_request(_):
            client = LLMClient()
            try:
                result = client.simple_chat("退货政策是什么？")
                return {"success": True, "latency": result["latency"]}
            except Exception as e:
                return {"success": False, "error": str(e)}

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(send_request, i) for i in range(20)]
            results = [f.result() for f in as_completed(futures)]

        success_count = sum(1 for r in results if r["success"])
        success_rate = success_count / len(results)
        print(f"并发测试 - 成功率: {success_rate:.0%}, 成功: {success_count}/{len(results)}")
        assert success_rate >= 0.90, f"并发成功率{success_rate:.0%}低于90%阈值"
