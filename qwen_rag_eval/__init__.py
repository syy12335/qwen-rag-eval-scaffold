"""
qwen_rag_eval 顶层包：对外统一入口
"""

# ================================
# 1. 数据构建相关接口（dataset_tools）
# ================================

from qwen_rag_eval.dataset_tools.sampling import build_eval_samples
'''
build_eval_samples(config: YamlConfigReader | str = "config/application.yaml") -> Path

用途：
1. 从配置中的 dataset.raw_path 读取原始数据集（例如 CMRC2018）
2. 抽样并转换为标准化 samples（id / question / ground_truth / ground_truth_context）
3. 写入 dataset.samples_path，如文件已存在则直接复用
典型用法：
    from qwen_rag_eval import build_eval_samples
    samples_path = build_eval_samples("config/application.yaml")
'''

from qwen_rag_eval.dataset_tools.chunking import make_chunks_from_samples
'''
make_chunks_from_samples(
    samples_path: str | Path,
    chunks_path: str | Path,
    chunk_size: int = 300,
    chunk_overlap: int = 30,
) -> Path

用途：
1. 读取一份符合规范的 samples JSON 文件
2. 按 ground_truth_context 切片，生成 chunk 级 JSONL 文件
3. 每行包含 doc_id / sample_id / text，用于后续向量库构建
典型用法：
    from qwen_rag_eval import make_chunks_from_samples
    make_chunks_from_samples(".../samples.json", ".../samples_chunks.jsonl")
'''

from qwen_rag_eval.dataset_tools.loader import load_cmrc_samples
'''
load_cmrc_samples(config: YamlConfigReader | str = "config/application.yaml") -> list[dict]

用途：
1. 按配置中的 dataset.samples_path 加载 samples 文件
2. 返回 Python list[dict]，不做额外处理
典型用法：
    from qwen_rag_eval import load_cmrc_samples
    samples = load_cmrc_samples("config/application.yaml")
'''

# ================================
# 2. 向量库相关接口（vector）
# ================================

from qwen_rag_eval.vector.vector_store_manager import VectorStoreManager
'''
VectorStoreManager(config: YamlConfigReader | None = None)

用途：
1. 封装 DashScopeEmbeddings + Chroma 的向量库管理
2. 内部读取 application.yaml 中的 vector_store.embedding_model / persist_directory
   并兼容旧键 path.embedding-directory / embedding.model
3. 提供：
   load_or_create_collection(collection_name)
   add_documents(documents, collection_name, batch_size=...)
   get_retriever(collection_name, k=...)
   delete_collection(collection_name)
典型用法：
    from qwen_rag_eval import VectorStoreManager
    manager = VectorStoreManager()
    retriever = manager.get_retriever("cmrc2018", k=5)
'''

from qwen_rag_eval.vector.vector_builder import build_cmrc_dataset_and_vector_store
'''
build_cmrc_dataset_and_vector_store(
    config_path: str = "config/application.yaml",
) -> None

用途：
1. 从原始 CMRC 数据开始，一键完成：
   raw → samples → chunks → 向量库
2. 中间产物已存在时自动复用，避免重复构建
典型用法：
    from qwen_rag_eval import build_cmrc_dataset_and_vector_store
    build_cmrc_dataset_and_vector_store("config/application.yaml")
'''

from qwen_rag_eval.vector.vector_builder import ensure_cmrc_vector_store
'''
ensure_cmrc_vector_store(
    config: YamlConfigReader | str = "config/application.yaml",
) -> None

用途：
1. 按配置检查目标 Chroma collection 是否存在
2. 不存在时自动补齐：
   如 samples 缺失则调用 build_eval_samples
   如 chunks 缺失则调用 make_chunks_from_samples
   最终写入向量库
3. 已存在时不做改动
典型用法：
    from qwen_rag_eval import ensure_cmrc_vector_store
    ensure_cmrc_vector_store("config/application.yaml")
'''

from qwen_rag_eval.vector.vector_builder import build_vector_store_from_samples
'''
build_vector_store_from_samples(
    samples_path: str | Path,
    config_path: str = "config/application.yaml",
    collection_name: str | None = None,
    overwrite: bool = False,
) -> None

用途：
1. 在只给定一份 samples 的前提下，一键构建向量库
2. 自动在 samples 同目录生成 <name>_chunks.jsonl，并写入 Chroma
3. collection_name 未显式传入时，使用 vector_store.collection_name
4. 当 overwrite=False 且目标 collection 已存在时，直接返回
典型用法：
    from qwen_rag_eval import build_vector_store_from_samples
    build_vector_store_from_samples(
        "datasets/processed/my_samples.json",
        config_path="config/application.yaml",
        overwrite=False,
    )
'''

# ================================
# 3. 未来：评估相关接口（evaluation）
# ================================
'''
预留位：
后续在 qwen_rag_eval.evaluation 子包中实现 RagasEvaluator / EvalResult /
RagBatchRunner 后，可在此处挂载为包级接口，供外部直接导入使用。
'''
# from qwen_rag_eval.evaluation.ragas_eval import RagasEvaluator
# from qwen_rag_eval.evaluation.eval_result import EvalResult
# from qwen_rag_eval.evaluation.rag_batch_runner import RagBatchRunner

__all__ = [
    "build_eval_samples",
    "make_chunks_from_samples",
    "load_cmrc_samples",
    "VectorStoreManager",
    "build_cmrc_dataset_and_vector_store",
    "ensure_cmrc_vector_store",
    "build_vector_store_from_samples",
    # "RagasEvaluator",
    # "EvalResult",
    # "RagBatchRunner",
]
