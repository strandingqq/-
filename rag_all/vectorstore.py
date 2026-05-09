from __future__ import annotations

import random
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import Settings

"""
加下划线前缀只是 Python 的命名惯例，表示"这是内部用的
"""
_db: Optional[Chroma] = None
_retriever = None

"""
需要详细学习embedding的获取方法 使用方法
这里做的是获取embedding模型
HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)   做加速 做缓存（非常关键） ？？


emb.embed_query("什么是RAG？")            返回的是 List[float]
emb.embed_documents([ "LangChain 是什么",
                    "Embedding 的作用"])  返回的是 List[List[float]]
"""
def _get_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
    # 等价于emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 


def get_vectorstore(settings: Settings) -> Chroma:
    """Lazy-load and cache Chroma vectorstore.
    懒加载：第一次调用时加载，之后直接返回缓存的结果 （if _db is None）
    db = Chroma(
            persist_directory=str(settings.chroma_persist_dir), 数据库文件存放的目录路径。
            collection_name=settings.collection_name,           要操作的集合名
            embedding_function=embeddings,                      指定用哪个 embedding 模型
        )

    retriever = VectorStoreRetriever(vectorstore=_db,search_kwargs={"k": 4}) 和
    _retriever = _db.as_retriever(search_kwargs={"k": 4}) 是等价的

    docs = _retriever.invoke("什么是RAG")
    """
    global _db, _retriever
    if _db is None: 
        embeddings = _get_embeddings(settings)
        _db = Chroma(
            persist_directory=str(settings.chroma_persist_dir),
            collection_name=settings.collection_name,
            embedding_function=embeddings,
        )
        # Create a retriever once; it’s relatively stable.
        _retriever = _db.as_retriever(search_kwargs={"k": 4})
        # retriever = VectorStoreRetriever(vectorstore=_db,search_kwargs={"k": 4})
    return _db


def get_retriever(settings: Settings):
    """Lazy-load and cache retriever.
    构建一个“Top-4 语义检索器”
    用于从 Chroma 中找最相关的4段文本
    """
    get_vectorstore(settings)
    return _retriever


def docs_to_context(docs: list[Document]) -> str:
    """Convert retrieved documents to a plain text context.
    把检索到的文档列表，拼成一段字符串，作为上下文喂给大模型。
    list[Document]：这是angchain 的一种结构
    Document(
        page_content="文本内容",
        metadata={...}
    )
    注意是\n\n 空一行 分割不同文档，提高可读性，帮助llm理解结构

    优化方法：
    加metedata
    限制长度 防止token超长
    排序
    """
    return "\n\n".join(d.page_content for d in docs if d.page_content)


def retrieve_context(settings: Settings, query: str) -> str:
    """Retrieve relevant context from vector store (RAG).
    完整的rag检索流程
    """
    retriever = get_retriever(settings)
    docs = retriever.invoke(query)
    return docs_to_context(docs) if docs else ""


def get_new_question_data( # 抽取新题目
    settings: Settings,
    db: Chroma,
    topics_list: list[str] | tuple[str, ...],
    level: str = "easy",
) -> tuple[Optional[str], Optional[str]]:
    """
    Pick a random topic, then pick one random question from Chroma by metadata.
    db.get(where={"$and": [...]}) 是 Chroma 的 metadata 过滤语法，
    """
    topics = list(topics_list)
    if not topics:
        return None, None

    # Random pick topic
    selected_topic = random.choice(topics)
    results = db.get(
        where={"$and": [{"keyword": selected_topic}, {"level": level}]}
    )

    ids = results.get("ids", [])
    if not ids:
        # Fallback: try remaining topics.
        remaining = [t for t in topics if t != selected_topic]
        if not remaining:
            return None, None
        return get_new_question_data(settings, db, remaining, level=level)

    idx = random.randint(0, len(ids) - 1)
    documents = results["documents"]
    metadatas = results["metadatas"]
    question_content = documents[idx]
    metadata = metadatas[idx] if metadatas else {}
    # metadata may contain the canonical keyword/level.
    topic = metadata.get("keyword", selected_topic)
    return question_content, topic

