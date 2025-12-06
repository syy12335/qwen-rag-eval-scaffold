# qwen_rag_eval/evaluation/ragas_eval.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import os

from datasets import Dataset, Features, Sequence as HFSequence, Value

from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

import pandas as pd

from utils import YamlConfigReader
from qwen_rag_eval.evaluation.rag_batch_runner import RagEvalRecord
from qwen_rag_eval.evaluation.eval_result import EvalResult


def _build_ragas_dataset(records: List[RagEvalRecord]) -> Dataset:
    """
    将 RagEvalRecord 列表转换为 RAGAS 期望的 Dataset 结构。

    字段约定（对应 RAGAS 传统接口）：
      question:      str                    # 问题文本
      answer:        str                    # RAG 生成的回答
      contexts:      List[str]              # 上下文文本列表
      ground_truth:  str                    # 标准答案（可以为空字符串）
    """
    if not records:
        raise ValueError("records 为空，无法构建 RAGAS 数据集。")

    # 1. 先把所有字段整理成最朴素的 Python 列表
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for r in records:
        questions.append(str(r.question) if r.question is not None else "")
        answers.append(str(r.answer) if r.answer is not None else "")

        # r.contexts 应该是 List[str] 或 List[Document]，这里统一转成 List[str]
        ctx_texts: List[str] = []
        for c in r.contexts or []:
            # 如果是 langchain 的 Document，就取 page_content；否则直接转字符串
            text = getattr(c, "page_content", c)
            ctx_texts.append(str(text))
        contexts_list.append(ctx_texts)

        # ground_truth 允许为空，统一转成 str
        gt = r.ground_truth if r.ground_truth is not None else ""
        ground_truths.append(str(gt))

    raw_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }

    # 2. 明确声明 HF Features：contexts 是 Sequence[string]
    features = Features(
        {
            "question": Value("string"),
            "answer": Value("string"),
            "contexts": HFSequence(Value("string")),
            "ground_truth": Value("string"),
        }
    )

    # 3. 先 from_dict，再用 cast 强制成上面的 Features
    ds = Dataset.from_dict(raw_data)
    ds = ds.cast(features)

    # 4. 调试输出（只在你排查问题这段时间保留，稳定后可以删掉）
    print("[debug] RAGAS Dataset:", ds)
    print("[debug] features:", ds.features)
    ctx_feat = ds.features["contexts"]
    print(
        "[debug] contexts feature -> type:",
        type(ctx_feat),
        "inner dtype:",
        getattr(ctx_feat, "feature", None).dtype if getattr(ctx_feat, "feature", None) else None,
    )

    return ds


def _build_ragas_components(config: YamlConfigReader):
    """
    构造 RAGAS 评估使用的 LLM 与 Embedding 封装。

    模型来源优先级：
      1. application.yaml 中 evaluation 段的配置：
           evaluation.llm_model
           evaluation.embedding_model
      2. 默认值：
           llm_model = "qwen-flash"
           embedding_model = "text-embedding-v4"
    """
    eval_llm_model = (
        config.get("evaluation.llm_model")
        or "qwen-flash"
    )
    eval_embedding_model = (
        config.get("evaluation.embedding_model")
        or "text-embedding-v4"
    )

    api_key = os.environ.get("API_KEY_Qwen")
    if not api_key:
        raise ValueError(
            "未在环境变量 API_KEY_Qwen 中找到千问 API Key，"
            "请先设置：set API_KEY_Qwen=你的key"
        )

    llm = LangchainLLMWrapper(
        ChatTongyi(
            dashscope_api_key=api_key,
            model=eval_llm_model,
            temperature=0,
        )
    )

    embeddings = LangchainEmbeddingsWrapper(
        DashScopeEmbeddings(
            model=eval_embedding_model,
            dashscope_api_key=api_key,
        )
    )

    return llm, embeddings


def run_ragas_evaluation(
    records: List[RagEvalRecord],
    config_path: str = "config/application.yaml",
    metrics: Optional[Sequence[Any]] = None,
) -> EvalResult:
    """
    以函数方式执行一次完整的 RAGAS 评估流程。

    参数：
      records:
        来自 RagBatchRunner 的 RagEvalRecord 列表。

      config_path:
        application.yaml 路径，默认 "config/application.yaml"。

      metrics:
        要使用的 RAGAS 指标列表。
        若为 None，则默认使用：
          faithfulness, answer_relevancy, context_precision, context_recall

    返回：
      EvalResult 对象，封装整体指标、逐样本结果、Dataset、CSV 路径等信息。
    """
    if not records:
        raise ValueError("records 为空，无法执行 RAGAS 评估。")

    config = YamlConfigReader(config_path)

    # 计算项目根路径（约定 application.yaml 位于 project-root/config 下）
    project_root = config.config_path.parent.parent

    # CSV 输出路径配置
    csv_rel_path = config.get("evaluation.output_csv") or "data/evaluation/ragas_result.csv"
    csv_path = (project_root / csv_rel_path).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 构造 Dataset
    dataset = _build_ragas_dataset(records)

    # 评估使用的 LLM 与 Embeddings
    eval_llm, eval_embeddings = _build_ragas_components(config)

    # 指标列表
    if metrics is None:
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    # 为需要 embeddings 的指标注入统一的 embedding 对象
    for m in metrics:
        if hasattr(m, "embeddings"):
            setattr(m, "embeddings", eval_embeddings)

    # 调用 RAGAS 评估
    result = ragas_evaluate(
        dataset=dataset,
        metrics=list(metrics),
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    # 转为 DataFrame
    df: pd.DataFrame = result.to_pandas()

    # 写出 CSV
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[ragas_eval] 已将评估结果写入：{csv_path}")

    # 汇总 overall 指标
    overall: Dict[str, float] = {}
    for m in metrics:
        name = getattr(m, "name", None)
        if name is None:
            continue
        try:
            overall[name] = float(result[name])
        except Exception:
            # 若某个指标不在 result 中或非标量，可以忽略或保留原值
            value = result.get(name)
            if value is not None:
                try:
                    overall[name] = float(value)
                except Exception:
                    pass

    eval_result = EvalResult(
        overall=overall,
        per_sample=df,
        dataset=dataset,
        csv_path=str(csv_path),
        raw_result=result,
    )
    return eval_result


class RagasEvaluator:
    """
    RagasEvaluator：面向对象的评估封装。

    用法示例：

      from qwen_rag_eval.evaluation import RagBatchRunner, RagasEvaluator

      runner = DefaultRunner()
      batch = RagBatchRunner(runner)

      records = batch.run_batch(eval_samples, limit=200)
      evaluator = RagasEvaluator()
      eval_result = evaluator.evaluate(records)

      eval_result.show_console()
    """

    def __init__(
        self,
        config_path: str = "config/application.yaml",
        metrics: Optional[Sequence[Any]] = None,
    ):
        self.config_path = config_path
        self.metrics = metrics

    def evaluate(self, records: List[RagEvalRecord]) -> EvalResult:
        """
        执行 RAGAS 评估。
        """
        return run_ragas_evaluation(
            records=records,
            config_path=self.config_path,
            metrics=self.metrics,
        )
