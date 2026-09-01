import json
import datetime
import os
import pytest
from utils.llm_client import LLMClient
from config.prompts import SYSTEM_PROMPTS


class TestDrift:
    def test_drift_detection(self, golden_set):
        client = LLMClient()

        try:
            with open("test_results/baseline_accuracy.json", "r") as f:
                baseline = json.load(f)
        except FileNotFoundError:
            baseline = {"accuracy_rate": 0.90}

        correct_count = 0
        for item in golden_set:
            result = client.simple_chat(
                item["question"],
                system_prompt=SYSTEM_PROMPTS["customer_service"]
            )
            if any(kw in result["content"] for kw in item["keywords"]):
                correct_count += 1

        current_accuracy = correct_count / len(golden_set)
        drift = baseline["accuracy_rate"] - current_accuracy

        print(f"基线准确率: {baseline['accuracy_rate']:.2%}, "
              f"当前准确率: {current_accuracy:.2%}, 漂移: {drift:.2%}")

        report = {
            "date": datetime.date.today().isoformat(),
            "accuracy_rate": current_accuracy,
            "baseline_accuracy": baseline["accuracy_rate"],
            "drift": drift,
        }

        os.makedirs("test_results", exist_ok=True)
        with open(f"test_results/drift_report_{datetime.date.today().isoformat()}.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        assert drift <= 0.05, f"性能退化超过5%阈值，当前漂移{drift:.2%}"
