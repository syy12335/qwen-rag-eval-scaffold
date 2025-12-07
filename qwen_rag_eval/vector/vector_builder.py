# qwen_rag_eval/vector/vector_builder.py

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from chromadb import PersistentClient
from langchain.schema import Document

from utils import YamlConfigReader
from qwen_rag_eval.dataset_tools.sampling import build_eval_samples
from qwen_rag_eval.dataset_tools.chunking import (
    make_chunks_from_samples,
    load_chunk_records,
)
from qwen_rag_eval.vector.vector_store_manager import VectorStoreManager


PathLike = Union[str, Path]


def _get_project_root(config: YamlConfigReader) -> Path:
    """
    约定：
      application.yaml 位于 project-root/config/application.yaml
      因此 project-root = config_path.parent.parent
    """
    return config.config_path.parent.parent


def _to_abs_path(path: PathLike, project_root: Path) -> Path:
    """
    将传入路径转换为绝对路径：
      若本身为绝对路径，直接返回；
      否则视为相对于 project_root。
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root / p


def convert_chunks_to_documents(records: List[Dict[str, Any]]) -> List[Document]:
    """
    将 chunk 记录转换为 LangChain Document 列表。

    数据标准：
      每条记录至少包含：
        text: str       chunk 文本内容
        doc_id: str     chunk 唯一标识
        sample_id: str  对应样本 id
    """
    docs: List[Document] = []
    for item in records:
        docs.append(
            Document(
                page_content=item["text"],
                metadata={
                    "doc_id": item.get("doc_id", ""),
                    "sample_id": item.get("sample_id", ""),
                },
            )
        )
    return docs


def deduplicate_documents(docs: List[Document]) -> List[Document]:
    """
    对一批 Document 做去重。

    策略：
      以 page_content 作为唯一键，文本完全相同视为重复。
      若多条 Document 拥有相同的 page_content，则保留遇到的第一条，
      丢弃后续相同文本的 Document。
    """
    seen_texts = set()
    deduped: List[Document] = []

    for doc in docs:
        text = doc.page_content
        if text in seen_texts:
            continue
        seen_texts.add(text)
        deduped.append(doc)

    if len(deduped) != len(docs):
        print(
            f"[vector_builder] 去重后文档数：{len(deduped)} "
            f"(原始 {len(docs)}，去掉 {len(docs) - len(deduped)} 条完全重复文本)"
        )

    return deduped


def ensure_cmrc_vector_store(
    config: Union[YamlConfigReader, str] = "config/application.yaml",
) -> None:
    """
    CMRC 专用入口（配置驱动）：

      1. 使用 application.yaml 中的 dataset.* 和 vector_store.* 配置
      2. 若目标 collection 已存在，则直接返回
      3. 若 samples 不存在，则调用 build_eval_samples 构建
      4. 若 chunks 不存在，则基于 samples 构建 JSONL chunks
      5. 读取 chunks → Document → 写入向量库（含去重）
    """
    # 统一成 YamlConfigReader 实例
    if isinstance(config, str):
        config = YamlConfigReader(config)

    project_root = _get_project_root(config)

    # collection 名称从 vector_store.collection_name 读取
    collection = config.get("vector_store.collection_name")
    if not collection:
        raise ValueError("配置缺少 vector_store.collection_name")

    # VectorStoreManager 内部负责读取 embedding_model 和 persist_directory
    manager = VectorStoreManager(config)
    persist_dir = Path(manager.persist_directory)

    # 使用 PersistentClient 检查 collection 是否已存在
    client = PersistentClient(path=str(persist_dir))
    existing_collections = [c.name for c in client.list_collections()]

    if collection in existing_collections:
        print(
            f"[vector_builder] 向量库已存在：{persist_dir} "
            f"(collection='{collection}')"
        )
        return

    # 确保 samples 存在
    samples_rel = config.get("dataset.samples_path")
    if not samples_rel:
        raise ValueError("配置缺少 dataset.samples_path")

    samples_path = project_root / samples_rel
    if not samples_path.exists():
        print(f"[vector_builder] 样本不存在，先构建 samples：{samples_path}")
        # build_eval_samples 会根据 config.raw_path 等重新生成 samples
        samples_path = build_eval_samples(config)

    # 确保 chunks 存在
    chunks_rel = config.get("dataset.chunks_path")
    if not chunks_rel:
        raise ValueError("配置缺少 dataset.chunks_path")

    chunks_path = project_root / chunks_rel
    if not chunks_path.exists():
        print(f"[vector_builder] 使用 samples 切割 chunks → {chunks_path}")
        make_chunks_from_samples(samples_path, chunks_path)

    # 读取 chunks → Document → 写入向量库（带去重）
    records = load_chunk_records(chunks_path)
    docs = convert_chunks_to_documents(records)
    docs = deduplicate_documents(docs)

    print(
        f"[vector_builder] 写入向量库（共 {len(docs)} 条 chunk），"
        f"collection='{collection}'"
    )
    manager.add_documents(docs, collection_name=collection)

    print("[vector_builder] 向量库构建完成")


def build_vector_store_from_samples(
    samples_path: PathLike,
    config_path: str = "config/application.yaml",
    collection_name: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    通用入口：传入一个“评估样本”文件路径，自动切 chunk 并构建向量库。

    约定：
      1. samples_path 指向的数据必须满足 data_standard_README 中定义的
         "评估样本（samples）数据格式"。
      2. 若未显式指定 collection_name，则从 vector_store.collection_name 读取。
      3. chunks 文件默认写在 samples 同一目录，命名为：
         <样本文件名去掉扩展名> + "_chunks.jsonl"
    """
    config = YamlConfigReader(config_path)
    project_root = _get_project_root(config)

    samples_path = _to_abs_path(samples_path, project_root)
    if not samples_path.exists():
        raise FileNotFoundError(f"未找到样本文件：{samples_path}")

    if collection_name is None:
        collection_name = config.get("vector_store.collection_name")
    if not collection_name:
        raise ValueError(
            "未指定 collection_name，且配置中缺少 vector_store.collection_name"
        )

    manager = VectorStoreManager(config)
    persist_dir = Path(manager.persist_directory)

    client = PersistentClient(path=str(persist_dir))
    existing_collections = [c.name for c in client.list_collections()]

    if collection_name in existing_collections:
        if not overwrite:
            print(
                f"[vector_builder] 向量库已存在，直接返回：{persist_dir} "
                f"(collection='{collection_name}')"
            )
            return
        else:
            print(
                f"[vector_builder] 覆盖模式：删除已有 collection='{collection_name}'"
            )
            client.delete_collection(name=collection_name)

    chunks_path = samples_path.with_name(samples_path.stem + "_chunks.jsonl")

    print(f"[vector_builder] 使用 samples 切割 chunks → {chunks_path}")
    make_chunks_from_samples(samples_path, chunks_path)

    records = load_chunk_records(chunks_path)
    docs = convert_chunks_to_documents(records)
    docs = deduplicate_documents(docs)

    print(
        f"[vector_builder] 写入向量库（共 {len(docs)} 条 chunk），"
        f"collection='{collection_name}'"
    )
    manager.add_documents(docs, collection_name=collection_name)

    print("[vector_builder] 向量库构建完成（path 模式）")


def build_cmrc_dataset_and_vector_store(
    config_path: str = "config/application.yaml",
) -> None:
    """
    一键执行 CMRC 全流程：
      1. raw → samples
      2. samples → chunks
      3. chunks → VectorStore

    等价于：
      build_eval_samples(...)
      ensure_cmrc_vector_store(...)
    """
    config = YamlConfigReader(config_path)
    build_eval_samples(config)
    ensure_cmrc_vector_store(config)
