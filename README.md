# qwen-rag-eval-scaffold

A lightweight evaluation scaffold for Qwen-based RAG workflows. It runs the full pipeline — **Dataset → Vector Store → RAG Workflow → RAGAS Evaluation** — with a single command, and provides an optional Streamlit console for debugging and visualization.  
Designed as a clean baseline for building and evaluating RAG systems in the Qwen ecosystem.

## Features

1. **Built for the Qwen ecosystem**  
   Includes DashScope embeddings and Qwen model configuration out-of-the-box.  
   All agents are managed via `agents.yaml`.  
   Simply set the environment variable `API_KEY_Qwen` to run.

2. **Covers the entire RAG pipeline**  
   Data: CMRC2018 raw data → evaluation samples → chunks → Chroma vector store  
   Execution: run a full RAG workflow with `DefaultRunner` (or replace with your own)  
   Evaluation: powered by RAGAS, with both aggregated metrics and per-sample scoring (CSV export supported)

3. **Pluggable datasets and workflows**  
   Dataset: any collection satisfying the four fields  
   `id / question / ground_truth / ground_truth_context`  
   can be used as a custom evaluation set.  
   Workflow: any Runner implementing  
   `invoke(question) -> {"question": ..., "generation": ..., "contexts": [...]}`  
   can be plugged into the evaluation loop.

4. **Built‑in Chinese benchmark + visualization console**  
   Complete CMRC2018 processing pipeline (raw → samples → chunks → vector store).  
   Streamlit console supports batch evaluation, per-sample inspection, and interactive RAG QA.

## Installation & Environment Setup

1. **Python version**  
   Python 3.10+ is recommended.

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Configure Qwen API Key**

Windows example:

```bat
set API_KEY_Qwen=your-dashscope-api-key
python quick_start.py
```

Linux / macOS:

```bash
export API_KEY_Qwen="your-dashscope-api-key"
python quick_start.py
```

You may also add the environment variable to your shell config for convenience.

## Quick Start: Run the Full Pipeline with One Command

In the project root directory:

```bash
python quick_start.py
```

This command performs:

1. **Construct the CMRC benchmark dataset and vector store**  
   Converts raw CMRC2018 into evaluation samples, splits `ground_truth_context` into chunks, and builds the Chroma index.

2. **Load evaluation samples**  
   Reads `question / ground_truth / ground_truth_context` from `dataset.samples_path`.

3. **Run RAG in batch via `DefaultRunner`**  
   Uses the built‑in NormalRag workflow (you may replace it with your own).

4. **Evaluate with RAGAS**  
   Prints overall metrics and exports per-sample scores to CSV (path defined in `evaluation.output_csv`).  

If dependencies and environment variables are set correctly, this command completes the full loop from raw data to evaluation results.

## Usage Overview

### 1. Modify the default RAG workflow

1. Edit `_build_graph` in `qwen_rag_eval/runner/normal_rag.py`, adding/removing LangGraph nodes (e.g., rewriting, re‑ranking, multi‑hop retrieval).  
2. Keep these conventions unchanged to ensure full compatibility with the evaluation pipeline:  
   Constructor: `__init__(retriever, answer_generator)`  
   Output fields: `question / contexts / generation`  
   As long as `invoke()` returns the required structure, evaluation works out‑of‑the‑box.

### 2. Plug in your own Runner and evaluate

1. Implement the Runner interface:

```python
def invoke(self, question: str) -> dict:
    return {
        "question": question,
        "generation": "...",    # model’s final answer
        "contexts": [...],      # list of strings or objects with page_content
    }
```

2. Use the unified evaluation interface:

```python
from qwen_rag_eval.evaluation import RagBatchRunner, RagasEvaluator

runner = MyRunner()
records = RagBatchRunner(runner, mode="my_runner").run_batch(eval_samples)
result = RagasEvaluator("config/application.yaml").evaluate(records)
```

`records` will be normalized into a list of `RagEvalRecord` objects, suitable for direct input to RAGAS.

### 3. Replace the dataset and rebuild the vector store

1. Prepare your own `samples.json` following `datasets/DATASET_README.md`, ensuring at least the fields:  
   `id / question / ground_truth / ground_truth_context`.

2. Build a vector store from your samples:

```python
from qwen_rag_eval.vector import build_vector_store_from_samples

build_vector_store_from_samples(
    "path/to/samples.json",
    collection_name="my_collection",
    overwrite=True,
)
```

3. Or use configuration-driven building:

```python
from qwen_rag_eval import build_cmrc_dataset_and_vector_store

build_cmrc_dataset_and_vector_store("config/application.yaml")
```

## Configuration & Directory Structure (Brief)

1. **Key configurations (`config/application.yaml`)**

   Dataset (`dataset.*`):  
   `raw_path`, `samples_path`, `chunks_path`, `sample_limit`  

   Vector store (`vector_store.*`):  
   `embedding_model`, `persist_directory`, `collection_name`  

   Retrieval:  
   `retrieval.top_k`  

   Evaluation (`evaluation.*`):  
   e.g., `output_csv`  

   Agents (`config/agents.yaml`):  
   `default_answer_generator`, plus custom agents

2. **Directory layout**

```
datasets/                 # raw and processed data + data format docs
qwen_rag_eval/
  dataset_tools/          # sampling, chunking, loading utilities
  vector/                 # vector store building and management
  runner/                 # DefaultRunner, NormalRag, custom runners
  evaluation/             # RagBatchRunner, RagasEvaluator, etc.
app/streamlit_app.py      # Streamlit console for debugging & visualization
```
