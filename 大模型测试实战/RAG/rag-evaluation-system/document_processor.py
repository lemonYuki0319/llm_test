# document_processor.py
"""
文档加载与切分
"""
import json
import os
from typing import List, Dict

from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    get_testdate_dir,
    get_testcases_dir,
)


def load_knowledge_base(file_path: str) -> str:
    """
    读取知识库文档（支持 .txt 和 .docx）

    Args:
        file_path: 文档路径

    Returns:
        文档内容字符串

    Raises:
        FileNotFoundError: 文件不存在
        RuntimeError: 读取失败
    """
    try:
        if file_path.endswith(".docx"):
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在：{file_path}")
    except Exception as e:
        raise RuntimeError(f"读取失败：{e}") from e


def split_document(
    content: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, str]]:
    """
    将文档分段，每段生成chunk_id

    Args:
        content: 文档内容
        chunk_size: 切片大小
        chunk_overlap: 切片重叠大小

    Returns:
        [{"chunk_id": "DOC-01", "content": "..."}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；"],
    )

    chunks = splitter.split_text(content)

    documents = []
    for i, chunk in enumerate(chunks, 1):
        documents.append({"chunk_id": f"DOC-{i:03d}", "content": chunk})

    return documents


def save_chunks(chunks: List[Dict[str, str]], output_file: str) -> None:
    """
    保存文档切片到文件

    Args:
        chunks: 切片列表
        output_file: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def load_chunks(chunks_file: str) -> Dict[str, str]:
    """
    读取 chunks.jsonl -> {chunk_id: content}

    Args:
        chunks_file: 切片文件路径

    Returns:
        chunk_id 到内容的映射字典
    """
    mapping: Dict[str, str] = {}
    if not os.path.exists(chunks_file):
        return mapping

    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid = item.get("chunk_id")
            content = item.get("content")
            if cid and isinstance(content, str):
                mapping[cid] = content

    return mapping


# =============================================================================
# 便捷函数
# =============================================================================
def process_and_save_document(
    knowledge_file: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    output_dir: str | None = None,
) -> tuple[str, List[Dict[str, str]]]:
    """
    完整的文档处理流程：读取 -> 切分 -> 保存

    Args:
        knowledge_file: 知识库文件路径
        chunk_size: 切片大小
        chunk_overlap: 切片重叠大小
        output_dir: 输出目录（默认为 testcases 目录）

    Returns:
        (chunks_file_path, chunks_list)
    """
    # 读取文档
    print(f"[INFO] 读取知识库文档: {knowledge_file}")
    content = load_knowledge_base(knowledge_file)
    print(f"[INFO] 文档总字符数: {len(content)}")

    # 切分文档
    print(f"[INFO] 文档分段处理 (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")
    chunks = split_document(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"[OK] 生成 {len(chunks)} 个片段")

    # 保存切片
    if output_dir is None:
        output_dir = get_testcases_dir()

    chunks_file = os.path.join(output_dir, "chunks.jsonl")
    save_chunks(chunks, chunks_file)
    print(f"[OK] 切片已保存: {chunks_file}")

    return chunks_file, chunks


if __name__ == "__main__":
    # 测试代码
    script_dir = os.path.dirname(os.path.abspath(__file__))
    knowledge_file = os.path.join(
        script_dir, "testdate", "中华人民共和国数据安全法.docx"
    )
    chunks_file, chunks = process_and_save_document(knowledge_file)
    print(f"\n[DONE] 处理完成，共 {len(chunks)} 个切片")