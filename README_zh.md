# qwen-rag-eval-scaffold

面向千问 Qwen API 的轻量级 RAG 评测脚手架，用一条命令跑通「数据集 → 向量库 → RAG 工作流 → RAGAS 评估」，并提供可视化控制台。适合作为在 Qwen 生态下搭建和评估 RAG 系统的基线工程。

## 项目特点

1. 面向 Qwen 生态  
   内置 DashScope embedding 与 Qwen 系列模型配置，通过 `agents.yaml` 管理回答生成等 Agent，只需设置环境变量 `API_KEY_Qwen` 即可运行。

2. 覆盖 RAG 全流程  
   数据侧：CMRC2018 原始数据 → 评估样本（samples）→ chunks → Chroma 向量库。  
   执行侧：通过 `DefaultRunner` 跑一条 RAG 链路（可替换为自定义 Runner）。  
   评估侧：使用 RAGAS 自动评分，输出整体指标和逐样本结果，并支持 CSV 导出。

3. 数据集与工作流可插拔  
   数据集：只要样本满足 `id / question / ground_truth / ground_truth_context` 四个字段，即可替换为自定义基准数据集。  
   工作流：只要 Runner 实现统一协议 `invoke(question) -> {"question": ..., "generation": ..., "contexts": [...]}`，即可无缝接入评估组件，对比不同 RAG 方案。

4. 自带中文基准与可视化控制台  
   默认提供 CMRC2018 的完整处理流程（raw → samples → chunks → 向量库），并内置 Streamlit 控制台，支持批量评估、单条结果查看和在线 RAG 问答，方便调试与展示。

## 安装与环境配置

1. Python 版本  
   推荐使用 Python 3.10 及以上版本。

2. 安装依赖  

   在项目根目录执行：

   ```bash
   pip install -r requirements.txt
   ```

3. 配置千问 API Key  

   Windows（当前会话）示例：

   ```bat
   set API_KEY_Qwen=your-dashscope-api-key
   python quick_start.py
   ```

   Linux / macOS 示例：

   ```bash
   export API_KEY_Qwen="your-dashscope-api-key"
   python quick_start.py
   ```

   也可以将环境变量配置到系统或 shell 配置文件中，避免每次手动输入。

## Quick Start：一条命令跑通

在项目根目录执行：

```bash
python quick_start.py
```

该命令会自动完成：

1. 构建 CMRC 基准数据与向量库  
   从原始 CMRC2018 构建评估样本（samples），切分 `ground_truth_context` 为 chunks，并写入 Chroma 向量库。

2. 加载评估样本  
   从 `dataset.samples_path` 读取 `question / ground_truth / ground_truth_context`。

3. 使用 `DefaultRunner` 批量运行 RAG  
   内部使用约定好的 NormalRag 工作流（可替换为自定义实现）。

4. 使用 RAGAS 评估  
   输出整体指标，并根据 `evaluation.output_csv` 写出逐样本评分 CSV，方便进一步分析。

只要环境变量和依赖安装正确，这一命令即可完成从数据到评估结果的完整闭环。

## 使用方式概览

### 1. 改写默认 RAG workflow

1. 在 `qwen_rag_eval/runner/normal_rag.py` 中修改 `_build_graph`，增删 LangGraph 节点（例如重写、重排、多跳检索），控制整条 RAG 工作流结构。  
2. 保持以下约定不变，即可复用全部评估组件：  
   构造函数签名：`__init__(retriever, answer_generator)`  
   输出字段：`question / contexts / generation`  
   只要 `invoke` 的返回结构满足协议，评估代码无需改动。

### 2. 接入自定义 Runner 并评估

1. 实现满足协议的 Runner：

   ```python
   def invoke(self, question: str) -> dict:
       return {
           "question": question,
           "generation": "...",        # 模型最终回答
           "contexts": [...],          # 可以是 str 或带 page_content 的对象
       }
   ```

2. 使用统一评估接口：

   ```python
   from qwen_rag_eval.evaluation import RagBatchRunner, RagasEvaluator

   runner = MyRunner()
   records = RagBatchRunner(runner, mode="my_runner").run_batch(eval_samples)
   result = RagasEvaluator("config/application.yaml").evaluate(records)
   ```

   其中 `records` 为规范化后的 `RagEvalRecord` 列表，直接传入 RAGAS 进行评分。

### 3. 替换数据集并重建向量库

1. 按 `datasets/DATASET_README.md` 准备 `samples.json`，至少包含 `id / question / ground_truth / ground_truth_context`。  
2. 基于样本直接构建向量库：

   ```python
   from qwen_rag_eval.vector import build_vector_store_from_samples

   build_vector_store_from_samples(
       "path/to/samples.json",
       collection_name="my_collection",
       overwrite=True,
   )
   ```

3. 使用配置驱动构建：

   ```python
   from qwen_rag_eval import build_cmrc_dataset_and_vector_store

   build_cmrc_dataset_and_vector_store("config/application.yaml")
   ```

## 配置与目录结构（简要）

1. 关键配置项（`config/application.yaml`）  

   1）数据相关 `dataset.*`：`raw_path`、`samples_path`、`chunks_path`、`sample_limit`  
   2）向量库 `vector_store.*`：`embedding_model`、`persist_directory`、`collection_name`  
   3）检索相关：`retrieval.top_k`  
   4）评估 `evaluation.*`：如 `output_csv` 等  
   5）Agents（`config/agents.yaml`）：`default_answer_generator` 及其他自定义 agent

2. 目录结构示意

   ```text
   datasets/                 # 原始与处理后数据，以及数据格式说明
   qwen_rag_eval/
     dataset_tools/          # 抽样、切分、加载等数据处理工具
     vector/                 # 向量库管理与构建逻辑
     runner/                 # DefaultRunner、NormalRag 及扩展 Runner
     evaluation/             # RagBatchRunner、RagasEvaluator 等
   app/streamlit_app.py      # 本地调试与展示用的控制台入口
   ```
