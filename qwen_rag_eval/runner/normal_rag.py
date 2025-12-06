# qwen_rag_eval/runner/normal_rag.py

import logging
from typing import List

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain.schema import Document

logger = logging.getLogger(__name__)


class RagState(TypedDict):
    """
    LangGraph 中的状态结构：
      question: 当前问题
      contexts: 检索得到的文档列表
      generation: 最终回答
    """
    question: str
    contexts: List[Document]
    generation: str


class NormalRag:
    """
    NormalRag：基于 LangGraph 的“检索 + 生成”流程。

    对外约定：
      1. 构造函数签名保持不变：
         __init__(self, retriever, answer_generator)
      2. 主方法：
         invoke(self, question: str) -> dict
         返回结构：
           {
               "question": str,
               "contexts": List[Document],
               "generation": str,
           }

    中阶用户如需自定义工作流，建议只在 _build_graph 内部修改图结构，
    保持 __init__ 与 invoke 的接口不变。
    """

    def __init__(self, retriever, answer_generator):
        self.retriever = retriever
        self.answer_generator = answer_generator
        self.app = self._build_graph()

    def _build_graph(self):
        """
        默认图结构：
          entry → retrieve → generate → END
        """

        def retrieve(state: RagState) -> dict:
            question = state["question"]
            logger.info(f"[NormalRag] question = {question}")

            contexts = self.retriever.get_relevant_documents(question)
            logger.info(f"[NormalRag] retrieved {len(contexts)} contexts")

            return {
                "contexts": contexts
            }

        def generate(state: RagState) -> dict:
            question = state["question"]
            contexts = state["contexts"]

            generation = self.answer_generator.invoke(
                {
                    "question": question,
                    "contexts": contexts,
                    "memory_text": "",
                }
            )

            return {
                "generation": generation
            }

        workflow = StateGraph(RagState)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        app = workflow.compile()
        return app

    def invoke(self, question: str):
        """
        传入 question，返回字典：
          {
              "question": ...,
              "contexts": ...,
              "generation": ...
          }
        """
        init_state: RagState = {
            "question": question,
            "contexts": [],
            "generation": "",
        }

        final_state: RagState = self.app.invoke(init_state)

        return {
            "question": final_state["question"],
            "contexts": final_state["contexts"],
            "generation": final_state["generation"],
        }
