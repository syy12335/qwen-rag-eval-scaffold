# Evaluation Layer (RAG 评估层)

评估层是本项目的第三层功能模块，主要职责是：统一 RAG runner 的输出结构、批量化运行评估样本，并与 RAGAS 框架对接，生成可用于分析、展示与保存的结构化评估结果。

评估层不负责构建向量库、加载数据集，也不关心 RAG 工作流内部细节。其核心输入是一个实现了 `invoke(question)` 的 runner，以及一组标准化的评估样本。


## 1. 模块结构

项目中与评估相关的目录结构为：

```text
qwen_rag_eval/
  evaluation/
    rag_batch_runner.py     # 批量执行 RAG，构造 RagEvalRecord 列表
    ragas_eval.py           # 调用 RAGAS 评估，生成指标与 DataFrame
    eval_result.py          # 统一的评估结果对象 EvalResult
    __init__.py
```

三个核心模块的角色说明：

1. RagBatchRunner（rag_batch_runner.py）  
   输入：评估样本列表与 runner 实例。  
   输出：RagEvalRecord 列表（纯文本上下文 + 问题 + 回答 + ground truth）。

2. RagasEvaluator（ragas_eval.py）  
   输入：RagEvalRecord 列表。  
   输出：RAGAS 指标（faithfulness、answer_relevancy 等）和逐样本结果。

3. EvalResult（eval_result.py）  
   输入：RAGAS 结果对象。  
   输出：可展示、可保存、可进一步分析的统一结果对象。


## 2. Runner 协议（统一接口约定）

评估层依赖于 RAG runner 提供如下最小接口：

```python
out = runner.invoke(question)
```

返回值需要包含下列字段：

1. `question`：字符串，通常与传入问题一致。  
2. `generation`：字符串，RAG 的最终回答。  
3. `contexts`：列表，元素可以是：  
   3.1 纯字符串；  
   3.2 具有 `page_content` 属性的对象（例如 LangChain Document）。

评估层会自动将 `contexts` 规整为 `List[str]`，确保与 RAGAS 输入一致。

示例：

```python
{
  "question": "商鞅变法在何年实施？",
  "generation": "商鞅变法实施于前356年。",
  "contexts": [
      "……历史文献 A ……",
      Document(page_content="……文献 B ……")
  ]
}
```


## 3. 从样本到 RagEvalRecord 列表

典型调用流程如下：

```python
from qwen_rag_eval.runner.default_runner import DefaultRunner
from qwen_rag_eval.evaluation import RagBatchRunner

# 评估样本示意
samples = [
    {"question": "...", "ground_truth": "..."},
    {"question": "...", "ground_truth": "..."},
]

# 1. 初始化 runner
runner = DefaultRunner()

# 2. 批量执行
batch = RagBatchRunner(runner)
records = batch.run_batch(samples, limit=20)
```

其中 `records` 是 `List[RagEvalRecord]`，每条记录包含：问题、回答、上下文文本列表、ground truth 与元信息。


## 4. 执行 RAGAS 评估

在得到 `records` 后，可以直接调用 RAGAS 评估：

```python
from qwen_rag_eval.evaluation import RagasEvaluator

evaluator = RagasEvaluator("config/application.yaml")
result = evaluator.evaluate(records)
```

此时 `result` 为一个 `EvalResult` 对象，包含：

1. `overall`：总体指标（faithfulness、answer_relevancy、context_precision、context_recall 等）。  
2. `per_sample`：逐样本 `pandas.DataFrame`。  
3. `csv_path`：结果 CSV 保存路径（若已写出）。  
4. `dataset` 与 `raw_result`：用于高级分析或调试的底层对象。


## 5. 展示与导出评估结果

EvalResult 提供了若干便捷方法，辅助结果查看与持久化。

1. 在命令行中快速查看整体指标及部分样本：

```python
result.show_console(top_n=5)
```

2. 获取逐样本结果的 DataFrame：

```python
df = result.to_dataframe()
```

3. 导出为 CSV 文件：

```python
result.to_csv("data/evaluation/ragas_result.csv")
```

4. 使用 Streamlit 展示（可选）：

```python
result.show_streamlit()
```

该方法依赖 `streamlit`，需要预先安装依赖：

```bash
pip install streamlit
```


## 6. 自定义 Runner 的评估流程

评估层完全独立于默认 runner。只要自定义 runner 实现了前述的 `invoke(question)` 接口，即可直接复用评估能力。

自定义 runner 示例：

```python
class MyRunner:
    def invoke(self, question: str):
        answer = "...生成结果..."
        contexts = ["文本 A", "文本 B"]
        return {
            "question": question,
            "generation": answer,
            "contexts": contexts,
        }
```

与评估层衔接：

```python
from qwen_rag_eval.evaluation import RagBatchRunner, RagasEvaluator

runner = MyRunner()
batch = RagBatchRunner(runner, mode="my_runner_v1")

records = batch.run_batch(samples)
result = RagasEvaluator("config/application.yaml").evaluate(records)
```

对于评估层而言，只要保持 `invoke(question)` 约定不变，就可以替换不同的 RAG 实现进行对比实验。


## 7. 设计原则与使用建议

1. 解耦原则  
   评估层与 RAG 工作流解耦，只依赖统一的 runner 接口，不关心内部是普通检索、图结构、负记忆池还是其他增强机制。

2. 统一输入输出  
   输入统一为 `List[RagEvalRecord]`，输出统一封装在 `EvalResult` 中，便于脚本调用和可视化展示。

3. 逐步扩展  
   当前默认集成的指标为：faithfulness、answer_relevancy、context_precision、context_recall。  
   若后续需要扩展额外指标，可以在 `ragas_eval.py` 中修改 metrics 列表，并保持 EvalResult 接口不变。

4. 建议使用路径  
   在实践中，一条完整评估链路通常为：  
   4.1 使用数据层构建 samples；  
   4.2 使用 DefaultRunner 或自定义 runner；  
   4.3 使用 RagBatchRunner 产出 RagEvalRecord 列表；  
   4.4 使用 RagasEvaluator 获得 EvalResult；  
   4.5 使用 EvalResult 完成查看、导出或可视化。
