"""
将 generated_question_bank 目录下的 jsonl 文件批量导入到 ChromaDB。

本脚本支持两个独立的 Chroma Collection：
  - knowledge_chunks : 知识片段层（每个原子化知识点的完整讲解）
  - question_bank    : 题目层（每道面试题 + 参考答案 + 追问角度）

使用方式：
    # 全量导入（根据文件名自动判断 collection）
    python scripts/ingest_question_bank.py

    # 只导入知识片段
    python scripts/ingest_question_bank.py --collection knowledge_chunks

    # 只导入题目库
    python scripts/ingest_question_bank.py --collection question_bank

    # 指定具体文件
    python scripts/ingest_question_bank.py --files generated_question_bank/cs_fundamentals_chunks.jsonl

自动判断逻辑：
  - 文件名含 "_chunks"  → 入库到 knowledge_chunks
  - 其他 .jsonl 文件     → 入库到 question_bank（向后兼容旧 learn_txt_chunks）
  - "_errors" 和 "_auto_test" 目录的文件自动跳过
"""
import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

LOCAL_DEPS_DIR = BASE_DIR / ".deps"
if LOCAL_DEPS_DIR.exists():
    sys.path.insert(0, str(LOCAL_DEPS_DIR))

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

QB_DIR = BASE_DIR / "generated_question_bank"
PERSIST_DIR = str(BASE_DIR / "chroma_db")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 两个 Collection 的名称
COLLECTION_KNOWLEDGE = "knowledge_chunks"
COLLECTION_QUESTION = "question_bank"


def build_embeddings():
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)


def build_or_load_collection(collection_name: str) -> Chroma:
    embeddings = build_embeddings()
    db = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    return db


def detect_collection(jsonl_path: Path) -> str | None:
    """根据文件名判断应入库到哪个 collection。返回 None 表示跳过。"""
    if "_errors" in jsonl_path.parts or "_auto_test" in jsonl_path.parts:
        return None
    stem = jsonl_path.stem
    if "_chunks" in stem:
        return COLLECTION_KNOWLEDGE
    return COLLECTION_QUESTION


# ── 题目库 Document 构建 ──────────────────────────────────────────

def build_question_document(obj: dict, source_label: str) -> Document:
    topic = obj.get("topic", "Unknown")
    difficulty = obj.get("difficulty", "medium")
    question = obj.get("question", "")
    reference_points = obj.get("reference_points", [])
    follow_up_angles = obj.get("follow_up_angles", [])
    tags = obj.get("tags", [])

    page_content = (
        f"【知识点】: {topic}\n"
        f"【难度】: {difficulty}\n"
        f"【面试真题】: {question}\n"
        f"【参考要点】: {' | '.join(reference_points) if reference_points else '无'}\n"
        f"【追问角度】: {' | '.join(follow_up_angles) if follow_up_angles else '无'}\n"
        f"【标签】: {' '.join(tags) if tags else '无'}"
    )

    metadata = {
        "source": source_label,
        "keyword": topic,
        "level": difficulty,
        "role": obj.get("role", ""),
        "role_label": obj.get("role_label", ""),
        "category": obj.get("category", ""),
        "category_label": obj.get("category_label", ""),
        "question": question,
        "doc_type": "question",
    }

    return Document(page_content=page_content, metadata=metadata)


# ── 知识片段 Document 构建 ────────────────────────────────────────

def build_chunk_document(obj: dict, source_label: str) -> Document:
    topic = obj.get("topic", "Unknown")
    chunk_type = obj.get("chunk_type", "原理型")
    content = obj.get("content", "")
    key_points = obj.get("key_points", [])
    related_topics = obj.get("related_topics", [])
    tags = obj.get("tags", [])

    # page_content 使用干净的自然语言文本，便于 embedding 捕捉语义
    page_content = (
        f"【知识点】: {topic}\n"
        f"【片段类型】: {chunk_type}\n"
        f"【正文】: {content}\n"
        f"【核心要点】: {' | '.join(key_points) if key_points else '无'}\n"
        f"【关联知识点】: {' '.join(related_topics) if related_topics else '无'}"
    )

    metadata = {
        "source": source_label,
        "keyword": topic,
        "chunk_type": chunk_type,
        "role": obj.get("role", ""),
        "role_label": obj.get("role_label", ""),
        "category": obj.get("category", ""),
        "category_label": obj.get("category_label", ""),
        "content_length": len(content),
        "key_points_count": len(key_points),
        "doc_type": "chunk",
    }

    return Document(page_content=page_content, metadata=metadata)


# ── 文件扫描 ─────────────────────────────────────────────────────

def find_jsonl_files(
    qb_dir: Path,
    collection_filter: str | None = None,
    specific_files: list[Path] | None = None,
) -> list[tuple[Path, str]]:
    """返回 [(jsonl_path, collection_name)] 列表"""
    if specific_files:
        results = []
        for f in specific_files:
            col = detect_collection(f)
            if col is None:
                continue
            if collection_filter and col != collection_filter:
                continue
            results.append((f, col))
        return sorted(results, key=lambda x: str(x[0]))

    results = []
    for f in qb_dir.rglob("*.jsonl"):
        col = detect_collection(f)
        if col is None:
            continue
        if collection_filter and col != collection_filter:
            continue
        results.append((f, col))
    return sorted(results, key=lambda x: str(x[0]))


# ── 入库核心逻辑 ─────────────────────────────────────────────────

def ingest_jsonl(db: Chroma, jsonl_path: Path, doc_builder):
    """
    把一个jsonl文件导入数据库
    Args:
        db: Chroma 数据库实例
        jsonl_path: jsonl 文件路径
        doc_builder: 构建Documentd 函数
    Path("questions.jsonl").stem 获取文件名，不带扩展名

    enumerate(iterable, start) 返回 (索引, 元素)
    line = line.strip() 去掉前后空格 换行符
    json.loads 把json字符串 ---- Python对象
    """
    source_label = jsonl_path.stem
    docs = []
    skipped = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line: # 如果是空行 跳过
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"  第 {line_num} 行 JSON 解析失败，跳过")
                skipped += 1
                continue

            try:
                doc = doc_builder(obj, source_label)
                docs.append(doc)
            except Exception as exc:
                print(f"  第 {line_num} 行 Document 构建失败: {exc}，跳过")
                skipped += 1
                continue

    if not docs:
        print(f"  {jsonl_path.name} 中没有有效数据")
        return 0

    db.add_documents(docs)
    """
    for 每个 doc:
        embedding = model(doc.page_content)
        存入：
            id
            document
            metadata
            embedding
    """
    print(f"  [OK] {jsonl_path.name}: 导入 {len(docs)} 条（跳过 {skipped} 条）")
    return len(docs)


def ingest_collection(
    collection_name: str,
    jsonl_files: list[tuple[Path, str]],
) -> int:
    """
    把一个collection 下的所有jsonl文件导入数据库
    jsonl_files (文件路径, 该文件所属collection) find_json_files 的结果
    """
    if not jsonl_files:
        print(f"没有需要导入到 {collection_name} 的文件")
        return 0

    db = build_or_load_collection(collection_name)
    print(f"\n=== Collection: {collection_name} (当前记录数: {db._collection.count()}) ===")

    total = 0
    for jsonl_path, col in jsonl_files:
        """
        A if B else C
        B 为 True 返回 A, 否则返回 C
        """
        doc_builder = build_chunk_document if col == COLLECTION_KNOWLEDGE else build_question_document
        # 导入单个文件
        count = ingest_jsonl(db, jsonl_path, doc_builder)
        total += count

    print(f"导入完成，总计 {total} 条记录")
    print(f"Collection '{collection_name}' 当前总记录数: {db._collection.count()}")
    return total


# ── CLI ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """
    创建一个命令行参数解析器
    argparse是Python 标准库，用于解析命令行参数
    允许 python ingest.py --collection question --files a.jsonl b.jsonl运行程序
    parser.add_argument 就是增加参数 第一个参数 --collection
    """
    parser = argparse.ArgumentParser(
        description="将 jsonl 题库/知识片段导入 ChromaDB，支持双 Collection 模式。"
    )
    parser.add_argument(
        "--collection",
        choices=[COLLECTION_KNOWLEDGE, COLLECTION_QUESTION],
        default=None,
        help="只导入指定 collection，不传则导入全部",
    )
    parser.add_argument(
        "--files",
        nargs="+", # 控制参数个数，可以一个 可以多个
        type=Path, # 自动转换数据类型  --files a.jsonl 变成 Path("a.jsonl")
        default=None,
        help="指定要导入的 jsonl 文件路径",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args() 
    """
    解析命令行输入 转成Python对象
    python scripts\ingest_question_bank.py --collection question --files a.jsonl
    args.collection = "question"
    args.files = [Path("a.jsonl")]

    Path("a.jsonl") 创建一个路径对象
    Path("a.jsonl").resolve()
    把一个“相对路径”转换成“绝对路径”

    我在文件处理阶段使用 Path.resolve() 将用户输入路径统一转换为绝对路径，
    以避免相对路径带来的不确定性，提高系统稳定性。
    """
    files = [Path(f).resolve() for f in args.files] if args.files else None
    collection_filter = args.collection

    jsonl_files = find_jsonl_files(QB_DIR, collection_filter, files)
    """
    # 选出文件 result是 （f, col类型）的列表
    [
        (Path("a.jsonl"), "question"),
        (Path("b.jsonl"), "knowledge")
    ]
    """
    if not jsonl_files:
        print("[X] 未找到任何符合条件 jsonl 文件")
        return

    # 按 collection 分组
    by_collection: dict[str, list] = {}
    for path, col in jsonl_files:
        by_collection.setdefault(col, []).append((path, col))
        # 如果 key 不存在 → 创建并赋默认值
        # 如果存在 → 直接返回

    print(f"发现 {len(jsonl_files)} 个文件：")
    for path, col in jsonl_files:
        print(f"  [{col}] {path.relative_to(BASE_DIR)}")

    grand_total = 0
    for col, entries in sorted(by_collection.items()):
        grand_total += ingest_collection(col, entries)

    print(f"\n全部导入完成，总计 {grand_total} 条记录")


if __name__ == "__main__":
    main()