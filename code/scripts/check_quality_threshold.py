import json
import sys
from config.settings import QUALITY_THRESHOLDS


def check_quality_threshold():
    try:
        with open("test_results/quality_report.json", "r") as f:
            report = json.load(f)
    except FileNotFoundError:
        print("WARNING: quality_report.json not found, skipping threshold check")
        return

    blocked = False
    for metric, threshold in QUALITY_THRESHOLDS.items():
        actual = report.get(metric, 0)
        if metric == "hallucination_rate":
            if actual > threshold:
                print(f"BLOCKED: {metric} = {actual:.2%} > threshold {threshold:.2%}")
                blocked = True
        elif metric in ["avg_ttft", "avg_tpot_ms"]:
            if actual > threshold:
                print(f"BLOCKED: {metric} = {actual} > threshold {threshold}")
                blocked = True
        else:
            if actual < threshold:
                print(f"BLOCKED: {metric} = {actual:.2%} < threshold {threshold:.2%}")
                blocked = True

    if blocked:
        sys.exit(1)
    else:
        print("All quality thresholds passed!")


if __name__ == "__main__":
    check_quality_threshold()
