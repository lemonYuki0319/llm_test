# main.py
"""
主流程编排，串联所有功能

用法:
  python main.py generate                    # 生成测试集
  python main.py evaluate                    # 运行评估
  python main.py full                        # 完整流程（生成 + 评估）
  python main.py migrate                     # 迁移旧格式测试集
"""
import argparse
import os

from config import (
    get_testdate_dir,
    get_testcases_dir,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_QA_PER_CHUNK,
    DEFAULT_NEGATIVE_COUNT,
    DEFAULT_MULTI_HOP_COUNT,
)
from document_processor import process_and_save_document
from testset_generator import build_testset
from evaluator import run_evaluation, load_testset
from report_builder import export_summary_statistics


def generate_testset(
    knowledge_file: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    qa_per_chunk: int = DEFAULT_QA_PER_CHUNK,
    negative_count: int = DEFAULT_NEGATIVE_COUNT,
    multi_hop_count: int = DEFAULT_MULTI_HOP_COUNT,
    output_file: str | None = None,
):
    """
    生成测试集

    Args:
        knowledge_file: 知识库文件路径
        chunk_size: 切片大小
        chunk_overlap: 切片重叠
        qa_per_chunk: 每个切片的问答对数
        negative_count: 库外问题数
        multi_hop_count: 多跳问题数
        output_file: 输出文件路径
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 默认知识库文件
    if knowledge_file is None:
        knowledge_file = os.path.join(
            get_testdate_dir(), "中华人民共和国数据安全法.docx"
        )

    # 默认输出文件
    if output_file is None:
        output_file = os.path.join(get_testcases_dir(), "testset.jsonl")

    print("=" * 60)
    print("RAG 测试集生成")
    print("=" * 60)
    print(f"知识库文件: {knowledge_file}")
    print(f"切片大小: {chunk_size}")
    print(f"切片重叠: {chunk_overlap}")
    print(f"每切片问答数: {qa_per_chunk}")
    print(f"库外问题数: {negative_count}")
    print(f"多跳问题数: {multi_hop_count}")
    print("=" * 60)

    # 生成测试集
    testset_file = build_testset(
        knowledge_file=knowledge_file,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        qa_per_chunk=qa_per_chunk,
        negative_count=negative_count,
        multi_hop_count=multi_hop_count,
        output_file=output_file,
    )

    print(f"\n[SUCCESS] 测试集已生成: {testset_file}")
    return testset_file


def evaluate_testset(testset_file: str | None = None, max_cases: int | None = None):
    """
    运行评估

    Args:
        testset_file: 测试集文件路径
        max_cases: 最大用例数（用于测试）
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 默认测试集文件
    if testset_file is None:
        testset_file = os.path.join(get_testcases_dir(), "testset.jsonl")

    print("=" * 60)
    print("RAG 评估")
    print("=" * 60)
    print(f"测试集文件: {testset_file}")
    if max_cases:
        print(f"最大用例数: {max_cases}")
    print("=" * 60)

    # 运行评估
    report_df = run_evaluation(testset_file=testset_file, max_cases=max_cases)

    # 导出统计摘要
    out_dir = os.path.dirname(testset_file) or script_dir
    summary_file = os.path.join(out_dir, "evaluation_summary.txt")
    export_summary_statistics(report_df, summary_file)

    print(f"\n[SUCCESS] 评估完成")
    return report_df


def migrate_testset(testset_file: str | None = None):
    """
    迁移旧格式测试集

    Args:
        testset_file: 测试集文件路径
    """
    import json

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if testset_file is None:
        testset_file = os.path.join(get_testcases_dir(), "testset.jsonl")

    print("=" * 60)
    print("测试集迁移")
    print("=" * 60)
    print(f"测试集文件: {testset_file}")
    print("=" * 60)

    # 备份原文件
    backup_file = testset_file + ".bak"
    if os.path.exists(testset_file) and not os.path.exists(backup_file):
        import shutil

        shutil.copy2(testset_file, backup_file)
        print(f"[INFO] 已备份原文件: {backup_file}")

    # 加载并重新保存
    test_cases = load_testset(testset_file)

    with open(testset_file, "w", encoding="utf-8") as f:
        for case in test_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] 已迁移 {len(test_cases)} 条用例")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="RAG 评估系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py generate                                # 生成测试集
  python main.py generate --knowledge-file mydoc.docx  # 使用指定文档
  python main.py evaluate                                # 运行评估
  python main.py evaluate --max-cases 5                # 只评估前5条
  python main.py full                                    # 完整流程
  python main.py migrate                                 # 迁移旧格式
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # generate 子命令
    gen_parser = subparsers.add_parser("generate", help="生成测试集")
    gen_parser.add_argument("--knowledge-file", help="知识库文件路径")
    gen_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    gen_parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    gen_parser.add_argument("--qa-per-chunk", type=int, default=DEFAULT_QA_PER_CHUNK)
    gen_parser.add_argument(
        "--negative-count", type=int, default=DEFAULT_NEGATIVE_COUNT
    )
    gen_parser.add_argument(
        "--multi-hop-count", type=int, default=DEFAULT_MULTI_HOP_COUNT
    )
    gen_parser.add_argument("--output-file", help="输出文件路径")

    # evaluate 子命令
    eval_parser = subparsers.add_parser("evaluate", help="运行评估")
    eval_parser.add_argument("--testset-file", help="测试集文件路径")
    eval_parser.add_argument("--max-cases", type=int, help="最大用例数")

    # full 子命令
    full_parser = subparsers.add_parser("full", help="完整流程（生成 + 评估）")
    full_parser.add_argument("--knowledge-file", help="知识库文件路径")
    full_parser.add_argument("--max-cases", type=int, help="最大用例数")

    # migrate 子命令
    mig_parser = subparsers.add_parser("migrate", help="迁移旧格式测试集")
    mig_parser.add_argument("--testset-file", help="测试集文件路径")

    args = parser.parse_args()

    if args.command == "generate":
        generate_testset(
            knowledge_file=args.knowledge_file,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            qa_per_chunk=args.qa_per_chunk,
            negative_count=args.negative_count,
            multi_hop_count=args.multi_hop_count,
            output_file=args.output_file,
        )

    elif args.command == "evaluate":
        evaluate_testset(testset_file=args.testset_file, max_cases=args.max_cases)

    elif args.command == "full":
        testset_file = generate_testset(knowledge_file=args.knowledge_file)
        evaluate_testset(testset_file=testset_file, max_cases=args.max_cases)

    elif args.command == "migrate":
        migrate_testset(testset_file=args.testset_file)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()