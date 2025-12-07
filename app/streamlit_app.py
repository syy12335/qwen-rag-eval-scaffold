"""
极简版前端（严格基于 quick_start.py 的现有接口）：

功能：
  1. 在侧边栏指定 application.yaml，点击按钮构建 / 刷新 CMRC 向量库。
  2. 批量评估模式：
       - 完全走 quick_start.py 的四步流程：
           build_cmrc_dataset_and_vector_store
           load_cmrc_samples
           DefaultRunner + RagBatchRunner
           RagasEvaluator
       - 前端展示 overall 指标和逐样本评分表。
       - 将后台批量 Evaluating 进度通过回调同步到 Streamlit 进度条。
  3. 单条查看模式：
       - 在已经评估完成的前提下，从逐样本结果中选一条，
         展示该样本的 question / answer / contexts / 各项指标。
  4. 在线 RAG 问答模式：
       - 复用 DefaultRunner，输入任意问题直接走一条 RAG 链路，
         展示生成答案和检索上下文。
"""

from typing import List, Any, Dict

import streamlit as st
import pandas as pd
import time


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from qwen_rag_eval import (
    build_cmrc_dataset_and_vector_store,
    load_cmrc_samples,
)
from qwen_rag_eval.runner.default_runner import DefaultRunner
from qwen_rag_eval.evaluation import (
    RagBatchRunner,
    RagasEvaluator,
)


# ============================================================
# 复用 quick_start.py 的批量评估逻辑（增加进度回调）
# ============================================================

# app/streamlit_app.py 中，替换原来的 run_batch_evaluation 定义

def run_batch_evaluation(
    config_path: str,
    eval_limit: int,
) -> Any:
    """
    使用 quick_start.py 的同一条链路，增加前端展示：

      1. 前半段：批量生成答案（带真实进度条）
      2. 后半段：RAGAS 评分（转圈圈 + 提示大约耗时为生成答案的四倍）
    """
    # 1. 构建数据与向量库（如果已经存在会直接跳过或增量）
    build_cmrc_dataset_and_vector_store(config_path)

    # 2. 加载评估样本
    samples: List[dict] = load_cmrc_samples(config_path)
    eval_samples = samples[:eval_limit]
    total = len(eval_samples)

    # 3. 批量生成答案：DefaultRunner + RagBatchRunner
    runner = DefaultRunner(config_path=config_path)
    batch = RagBatchRunner(runner, mode="default")

    # 进度条：明确说明是“生成答案”
    rag_progress = st.progress(0, text="正在批量生成答案…")
    rag_status = st.empty()

    def rag_progress_callback(current: int, total_: int):
        ratio = current / max(total_, 1)
        rag_progress.progress(
            ratio,
            text=f"正在批量生成答案：{current}/{total_} ({ratio:.0%})"
        )
        rag_status.text(f"批量生成答案进度：{current}/{total_}")

    rag_start = time.perf_counter()
    records = batch.run_batch(
        eval_samples,
        show_progress=False,              # 关闭后端 tqdm，只用前端进度条
        progress_callback=rag_progress_callback,
    )
    rag_end = time.perf_counter()
    rag_duration = rag_end - rag_start

    rag_progress.progress(1.0, text="批量生成答案完成")
    rag_status.text(
        f"批量生成答案完成，共 {total} 条，实际耗时 {rag_duration:.1f} 秒"
    )

    # 4. RAGAS 评分阶段：只用转圈提示“正在评分中，大约是生成答案时间的四倍”
    #    做一个简单的时间估计，不参与任何真实逻辑
    estimated_eval_seconds = rag_duration * 4.0
    # 给一个合理的上下限，避免极端值
    estimated_eval_seconds = max(5.0, estimated_eval_seconds)

    evaluator = RagasEvaluator(config_path=config_path)

    msg = (
        f"正在进行 RAGAS 评分，大约需要 {estimated_eval_seconds:.0f} 秒"
        "（约为批量生成答案耗时的四倍）…"
    )

    eval_start = time.perf_counter()
    with st.spinner(msg):
        # 这里内部会打印它自己的 Casting / Evaluating tqdm 到终端，
        # 前端只看到这个“转圈圈”提示。
        result = evaluator.evaluate(records)
    eval_end = time.perf_counter()
    eval_duration = eval_end - eval_start

    st.info(f"评分完成，实际评分耗时约 {eval_duration:.1f} 秒")

    return result




# ============================================================
# Streamlit 主体
# ============================================================

def main():
    st.set_page_config(
        page_title="RAG Eval Console",
        layout="wide",
    )

    st.title("RAG 评估控制台")

    # 侧边栏全局配置
    st.sidebar.header("全局配置")

    default_config_path = "config/application.yaml"
    config_path = st.sidebar.text_input(
        "配置文件路径（application.yaml）",
        value=default_config_path,
    )

    eval_limit = st.sidebar.number_input(
        "评估样本数",
        min_value=1,
        max_value=200,
        value=20,
        step=1,
    )

    if st.sidebar.button("构建 / 刷新向量库"):
        with st.spinner("正在构建数据集与向量库…"):
            build_cmrc_dataset_and_vector_store(config_path)
        st.sidebar.success("向量库构建 / 刷新完成")

    mode = st.sidebar.radio(
        "模式选择",
        ("批量评估", "单条查看", "在线 RAG 问答"),
    )

    # 用于跨模式共享评估结果和 runner
    if "eval_result" not in st.session_state:
        st.session_state["eval_result"] = None
    if "eval_per_sample" not in st.session_state:
        st.session_state["eval_per_sample"] = None
    if "eval_overall" not in st.session_state:
        st.session_state["eval_overall"] = None
    if "eval_csv_path" not in st.session_state:
        st.session_state["eval_csv_path"] = None
    if "online_runner" not in st.session_state:
        st.session_state["online_runner"] = None

    # ========================================================
    # 模式一：批量评估（主入口）
    # ========================================================
    if mode == "批量评估":
        st.subheader("RAG 批量评估（RAGAS）")

        st.markdown(
            f"当前配置文件：`{config_path}`，本次评估样本数：{int(eval_limit)} 条"
        )

        if st.button("运行评估"):
            with st.spinner("正在执行 RAG 批量评估并计算 RAGAS 指标…"):
                result = run_batch_evaluation(
                    config_path=config_path,
                    eval_limit=int(eval_limit),
                )

            # 兼容 EvalResult 和 dict 两种返回形式
            if hasattr(result, "overall"):
                overall = result.overall
                csv_path = getattr(result, "csv_path", None)
                per_sample = getattr(result, "per_sample", None)
            else:
                overall = result.get("overall", {})
                csv_path = result.get("csv_path")
                per_sample = result.get("per_sample")

            st.session_state["eval_result"] = result
            st.session_state["eval_overall"] = overall
            st.session_state["eval_per_sample"] = per_sample
            st.session_state["eval_csv_path"] = csv_path

        # 如果已经有历史评估结果，则展示
        overall = st.session_state.get("eval_overall")
        per_sample = st.session_state.get("eval_per_sample")
        csv_path = st.session_state.get("eval_csv_path")

        if overall is not None:
            st.markdown("**Overall 指标**")

            # 只取数值型指标用于可视化
            numeric_overall = {
                k: float(v)
                for k, v in overall.items()
                if isinstance(v, (int, float))
            }

            col1, col2 = st.columns([2, 3])

            with col1:
                st.write(numeric_overall)

            if numeric_overall:
                with col2:
                    df_overall = pd.DataFrame.from_dict(
                        numeric_overall, orient="index", columns=["score"]
                    )
                    df_overall.index.name = "metric"
                    st.bar_chart(df_overall)

        if per_sample is not None:
            st.markdown("**逐样本评分表**")

            if isinstance(per_sample, pd.DataFrame):
                df_samples = per_sample
            else:
                try:
                    df_samples = pd.DataFrame(per_sample)
                except Exception:
                    df_samples = None

            if df_samples is not None:
                st.dataframe(df_samples, use_container_width=True)

        if csv_path:
            st.info(f"逐样本结果已写入 CSV：{csv_path}")

    # ========================================================
    # 模式二：单条查看
    # ========================================================
    elif mode == "单条查看":
        st.subheader("单条样本详情查看")

        per_sample = st.session_state.get("eval_per_sample")
        overall = st.session_state.get("eval_overall")

        if per_sample is None:
            st.warning("尚未有评估结果，请先在“批量评估”模式中运行一次评估。")
            return

        if isinstance(per_sample, pd.DataFrame):
            df_samples = per_sample
        else:
            try:
                df_samples = pd.DataFrame(per_sample)
            except Exception:
                st.error("逐样本结果格式异常，无法展示。")
                return

        st.markdown(f"当前已有样本数：{len(df_samples)} 条")

        idx = st.number_input(
            "选择样本索引（从 0 开始）",
            min_value=0,
            max_value=len(df_samples) - 1,
            value=0,
            step=1,
        )

        row = df_samples.iloc[int(idx)]

        # question / answer / ground_truth
        if "question" in df_samples.columns:
            st.markdown("**Question**")
            st.write(row.get("question", ""))

        if "answer" in df_samples.columns:
            st.markdown("**Answer（RAG 生成）**")
            st.write(row.get("answer", ""))

        if "ground_truth" in df_samples.columns:
            st.markdown("**Ground Truth**")
            st.write(row.get("ground_truth", ""))

        # contexts 展示
        if "contexts" in df_samples.columns:
            st.markdown("**Contexts（检索到的上下文）**")
            st.write(row.get("contexts", ""))

        # 指标列：排除原始字段，剩下的都视为评分
        metric_cols = [
            c
            for c in df_samples.columns
            if c not in ("question", "answer", "contexts", "ground_truth")
        ]
        if metric_cols:
            st.markdown("**该样本的各项指标**")
            st.write(row[metric_cols].to_dict())

    # ========================================================
    # 模式三：在线 RAG 问答
    # ========================================================
    else:
        st.subheader("在线 RAG 问答")

        # 懒加载一个 DefaultRunner，避免每次重建
        if st.session_state["online_runner"] is None:
            st.session_state["online_runner"] = DefaultRunner(
                config_path=config_path
            )

        runner: DefaultRunner = st.session_state["online_runner"]

        # 简单单轮问答窗口（保持和 DefaultRunner.invoke(question) 接口一致）
        question = st.text_area(
            "输入你的问题（将通过当前配置的 RAG 链路检索与生成）：",
            height=100,
            placeholder="例如：CMRC2018 数据集主要解决什么任务？",
        )

        if st.button("发送"):
            if not question.strip():
                st.warning("问题内容不能为空。")
                return

            with st.spinner("正在通过 RAG 链路生成答案…"):
                # 这里假定 DefaultRunner 暴露了 invoke(question: str) 接口
                out: Dict[str, Any] = runner.invoke(question)

            answer = out.get("answer") or out.get("generation", "")
            contexts = out.get("contexts", [])

            st.markdown("**Answer**")
            st.write(answer)

            if contexts:
                st.markdown("**检索到的上下文（前若干条）**")
                # 简单展开前 3 条
                for i, ctx in enumerate(contexts[:3]):
                    st.markdown(f"Context {i + 1}")
                    if isinstance(ctx, dict):
                        st.write(ctx)
                    else:
                        st.write(str(ctx))


if __name__ == "__main__":
    main()
