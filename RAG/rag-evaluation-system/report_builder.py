# report_builder.py
"""
生成评测报告，筛选低分样本
"""
import re
from typing import List, Dict

import pandas as pd

from config import FAITHFULNESS_THRESHOLD


# =============================================================================
# 报告字段顺序
# =============================================================================
REPORT_FIELD_ORDER = [
    "question",
    "type",
    "priority",
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "reference",
    "response",
    "retrieved_files",
    "retrieved_contexts",
    "thread_id",
]


def _format_value(value) -> str:
    """
    键值对中的值：列表用分号连接，换行压成空格，保证一行一个字段

    Args:
        value: 任意类型的值

    Returns:
        格式化后的字符串
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    if isinstance(value, list):
        text = "; ".join(str(v) for v in value)
    else:
        text = str(value)

    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = re.sub(r" +", " ", text).strip()

    return text


def build_report_dataframe(results: List[Dict], score_df: pd.DataFrame) -> pd.DataFrame:
    """
    整理报告字段（完整内容，便于键值对导出）

    Args:
        results: 评估结果列表
        score_df: 评分 DataFrame

    Returns:
        报告 DataFrame
    """
    rows = []
    for i, r in enumerate(results):
        score_row = score_df.iloc[i] if i < len(score_df) else pd.Series(dtype=object)
        rows.append(
            {
                "question": r.get("question", ""),
                "type": r.get("type", ""),
                "priority": r.get("priority", ""),
                "faithfulness": score_row.get("faithfulness", "")
                if hasattr(score_row, "get")
                else "",
                "answer_relevancy": score_row.get("answer_relevancy", "")
                if hasattr(score_row, "get")
                else "",
                "context_recall": score_row.get("context_recall", "")
                if hasattr(score_row, "get")
                else "",
                "reference": r.get("expected", ""),
                "response": r.get("actual", ""),
                "retrieved_files": r.get("retrieved_files", []),
                "retrieved_contexts": r.get("contexts", []),
                "thread_id": r.get("thread_id", ""),
            }
        )

    return pd.DataFrame(rows)


def save_evaluation_report(df: pd.DataFrame, report_path: str) -> None:
    """
    写出键值对报告

    格式示例：
    ========== 样本 1/10 ==========
    question: 数据安全法里说的「数据」是什么意思啊？
    type: 可答
    priority: P0
    faithfulness: 0.85
    answer_relevancy: 0.92
    context_recall: 0.75
    ...

    Args:
        df: 报告 DataFrame
        report_path: 输出文件路径
    """
    blocks = []
    total = len(df)

    for idx, row in df.iterrows():
        lines = [f"========== 样本 {int(idx) + 1}/{total} =========="]
        for key in REPORT_FIELD_ORDER:
            if key not in row.index:
                continue
            lines.append(f"{key}: {_format_value(row[key])}")
        blocks.append("\n".join(lines))

    content = "\n\n".join(blocks) + ("\n" if blocks else "")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)


def print_summary(df: pd.DataFrame) -> None:
    """
    控制台打印三项指标均值与低分用例

    Args:
        df: 报告 DataFrame
    """
    print("\n" + "=" * 60)
    print(f"RAGAS评估报告  共 {len(df)} 条")

    for col, label in [
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer Relevancy"),
        ("context_recall", "Context Recall"),
    ]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            print(f"平均 {label}: {vals.mean():.2f}")

    # 筛选低分样本
    if "faithfulness" in df.columns:
        faith = pd.to_numeric(df["faithfulness"], errors="coerce")
        low_score = df[faith < FAITHFULNESS_THRESHOLD]
        if len(low_score) > 0:
            print(f"\n低分(Faithfulness<{FAITHFULNESS_THRESHOLD}): {len(low_score)} 条")
            for _, row in low_score.head(5).iterrows():
                print(f"   - {row.get('question', '')}")

    print("=" * 60)


def export_summary_statistics(df: pd.DataFrame, output_file: str) -> None:
    """
    导出统计摘要到文件

    Args:
        df: 报告 DataFrame
        output_file: 输出文件路径
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"RAGAS 评估统计摘要\n")
        f.write(f"总样本数: {len(df)}\n")
        f.write("=" * 60 + "\n\n")

        for col, label in [
            ("faithfulness", "Faithfulness"),
            ("answer_relevancy", "Answer Relevancy"),
            ("context_recall", "Context Recall"),
        ]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                f.write(f"{label}:\n")
                f.write(f"  平均值: {vals.mean():.4f}\n")
                f.write(f"  中位数: {vals.median():.4f}\n")
                f.write(f"  标准差: {vals.std():.4f}\n")
                f.write(f"  最小值: {vals.min():.4f}\n")
                f.write(f"  最大值: {vals.max():.4f}\n")
                f.write(f"  低分样本数 (<{FAITHFULNESS_THRESHOLD}): {(vals < FAITHFULNESS_THRESHOLD).sum()}\n")
                f.write("\n")


if __name__ == "__main__":
    # 测试代码：读取已有的评估报告并分析
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_file = os.path.join(script_dir, "testcases", "evaluation_report.txt")

    if os.path.exists(report_file):
        print(f"分析报告: {report_file}")
        # 这里可以添加报告分析逻辑
    else:
        print(f"报告文件不存在: {report_file}")