# Runner 层设计说明（DefaultRunner 与 NormalRag）

本文档说明本项目运行层（runner layer）的结构、使用方法，以及中阶与高阶用户的扩展方式。  
运行层位于 `qwen_rag_eval/runner/` 目录，核心文件包括：

1. `normal_rag.py`：定义默认 RAG 工作流（LangGraph），中阶用户在此文件中修改工作流。
2. `default_runner.py`：基于配置构造检索器与回答生成链，并封装 NormalRag，初学者直接使用。

运行层旨在提供一个稳定、轻量、可拆卸的执行接口，使不同能力水平的用户都能在不破坏底层数据结构与评估逻辑的前提下灵活构建自己的 RAG 流程。

---

# 一、DefaultRunner：默认运行方式（面向初学者）

初学者只需使用本组件，即可运行一个可用的 RAG demo。示例：

```python
from qwen_rag_eval.runner import DefaultRunner

runner = DefaultRunner()
result = runner.invoke("《战国无双3》是由哪两个公司合作开发的？")

print(result["generation"])
print(result["contexts"])
```

运行逻辑如下：

1. 从 `application.yaml` 读取向量库与检索配置。
2. 使用 `VectorStoreManager` 构造 retriever。
3. 从 `agents.yaml` 读取 `default_answer_generator` 的模型、解析方式与提示词。
4. 构造 NormalRag 工作流并执行检索加生成。
5. 返回统一格式：

```
{
  "question": str,
  "contexts": List[Document],
  "generation": str
}
```

该结构与评估层完全兼容，可直接用于 RAGAS 批量评估。

---

# 二、修改工作流：normal_rag.py（面向中阶用户）

中阶用户希望“保留默认检索与模型配置，但只修改 RAG 工作流”。  
为此，工作流被单独放在 `normal_rag.py` 中。

NormalRag 的核心结构：

```
entry → retrieve → generate → END
```

中阶用户可以在 `_build_graph` 方法中自由修改，例如：

1. 添加 rerank 节点。
2. 添加问题重写（rewrite）节点。
3. 添加多跳检索节点。
4. 修改边顺序或执行逻辑。

只需保持以下两点，该组件即可无缝接入 DefaultRunner 与评估层：

1. `__init__(self, retriever, answer_generator)` 不变。
2. `invoke(question)` 返回的结构包含  
   `question`、`contexts`、`generation` 三项。

示例修改方式（伪代码）：

```python
workflow.add_node("rerank", rerank_fn)
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "generate")
```

不需要改 default_runner，也不需要动向量库与评估层。

---

# 三、高阶用户：完全替换 Runner

高阶用户可完全实现一个新 runner，用更灵活的逻辑，如：

1. 自定义检索策略（如 rerank、dense + sparse hybrid）。
2. 使用不同的模型或链式推理方法。
3. 使用不同的 LangGraph 图结构或采用非 LangGraph 实现。
4. 引入 memory、tools、规划层等。

只需满足以下约定：

1. 提供一个 `invoke(self, question: str) -> dict` 方法。
2. 返回结构必须包含：

```
{
  "question": str,
  "generation": str,
  "contexts": List[str 或 Document]
}
```

3. 若 contexts 为 Document，则必须具备 `page_content` 字段。

最小可运行示例：

```python
class MyRunner:
    def __init__(self):
        self.retriever = ...
        self.generator = ...

    def invoke(self, question):
        ctx = self.retriever.get_relevant_documents(question)
        ans = self.generator.invoke({"question": question, "contexts": ctx})
        return {
            "question": question,
            "generation": ans,
            "contexts": ctx
        }
```

在评估层中替换 DefaultRunner：

```python
from qwen_rag_eval.evaluation.rag_batch_runner import RagBatchRunner

runner = MyRunner()
batch = RagBatchRunner(runner=runner)
records = batch.run_batch(eval_samples)
```

无需修改数据层、向量库层或评估层。

---

# 四、小结

1. DefaultRunner 提供最简单的 RAG 调用方式，适合初学者与评估链路。
2. NormalRag 专用于定义与修改工作流，适合中阶用户。
3. 高阶用户可完全替换 Runner，只需维持 invoke 的统一接口。
4. 数据层（samples/chunks）、向量库层（Chroma）和评估层（RAGAS）均保持稳定，不随工作流变化。

本设计使得运行层既可即插即用，也可按需扩展，同时不破坏项目整体的统一数据结构与评估接口。
