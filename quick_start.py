"""
quick_start.py

用一条命令跑通完整链路：
  1）构建 CMRC 数据集与向量库
  2）加载评估样本
  3）用 DefaultRunner 跑 RAG
  4）用 RAGAS 做评估并打印整体指标

运行方式（在项目根目录）：
    python quick_start.py
"""

from typing import List

from qwen_rag_eval import (
    build_cmrc_dataset_and_vector_store,
    load_cmrc_samples,
)
from qwen_rag_eval.runner.default_runner import DefaultRunner
from qwen_rag_eval.evaluation import (
    RagBatchRunner,
    RagasEvaluator,
)


def main():
    config_path = "config/application.yaml"

    print("[1] 构建 CMRC2018 数据与向量库…")
    build_cmrc_dataset_and_vector_store(config_path)

    print("[2] 加载评估样本…")
    samples: List[dict] = load_cmrc_samples(config_path)

    # 为了快速跑通，这里只取前 N 条样本
    eval_limit = 50
    eval_samples = samples[:eval_limit]
    print(f"[2] 本次评估样本数：{len(eval_samples)}")

    print("[3] 初始化 DefaultRunner 并批量执行 RAG…")
    runner = DefaultRunner(config_path=config_path)
    batch = RagBatchRunner(runner, mode="default")
    records = batch.run_batch(eval_samples, show_progress=True)

    print("[4] 调用 RAGAS 进行评估…")
    evaluator = RagasEvaluator(config_path=config_path)
    result = evaluator.evaluate(records)

    # 兼容两种实现：对象风格 EvalResult 或早期 dict 风格
    if hasattr(result, "overall"):
        overall = result.overall
        csv_path = getattr(result, "csv_path", None)
    else:
        overall = result.get("overall", {})
        csv_path = result.get("csv_path")

    print("\n[RAGAS Overall Metrics]")
    for name, value in overall.items():
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")

    if csv_path:
        print(f"\n详细逐样本结果已保存至: {csv_path}")


if __name__ == "__main__":
    main()
