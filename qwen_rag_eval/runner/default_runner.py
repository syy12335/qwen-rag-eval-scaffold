# qwen_rag_eval/runner/default_runner.py

import logging
import os
from typing import List

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain.schema import Document

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_models import ChatTongyi

from utils import YamlConfigReader
from qwen_rag_eval.vector.vector_store_manager import VectorStoreManager

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
    NormalRag：基于 LangGraph 的“检索 + 生成”流程

    Inputs:
        retriever: 具有 get_relevant_documents(question) 方法的检索器
        answer_generator: 具有 invoke(dict) 方法的回答生成链/Agent
    """

    def __init__(self, retriever, answer_generator):
        self.retriever = retriever
        self.answer_generator = answer_generator
        self.app = self._build_graph()

    def _build_graph(self):
        """
        图结构：
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


class DefaultRunner:
    """
    DefaultRunner：当前项目默认的 RAG 运行组件。

    职责：
      1. 读取 application.yaml 与 agents.yaml 中的必要配置。
      2. 使用 VectorStoreManager 构造 retriever。
      3. 使用 agents.default_answer_generator 构造回答生成链。
      4. 组装 NormalRag 工作流，对外只暴露 invoke(question)。
    """

    def __init__(
        self,
        config_path: str = "config/application.yaml",
        agent_config_path: str = "config/agents.yaml",
    ):
        self.config_path = config_path
        self.agent_config_path = agent_config_path

        # 1. 加载系统配置
        self.config = YamlConfigReader(config_path)

        # 2. 初始化向量库管理器（内部根据配置决定 embedding / 持久化目录）
        self.vector_manager = VectorStoreManager(self.config)

        # 3. 构造 retriever（可在子类中重写 _build_retriever 达到“可拆卸”目的）
        self.retriever = self._build_retriever()

        # 4. 构造回答生成链（从 agents.yaml 读取 default_answer_generator）
        self.answer_generator = self._build_answer_generator()

        # 5. 组装工作流（默认使用 NormalRag，亦可在子类中重写 _build_workflow）
        self.rag = self._build_workflow()

    # ============================================================
    # 受保护构造方法：方便将来在子类中重写
    # ============================================================

    def _build_retriever(self):
        """
        默认 retriever 构造逻辑：
          1. 从 vector_store.collection_name 读取集合名；
          2. 从 retrieval.top_k 读取 top_k（默认 3）；
          3. 调用 VectorStoreManager.get_retriever。
        """
        collection_name = self.config.get("vector_store.collection_name")
        if not collection_name:
            raise ValueError("配置缺少 vector_store.collection_name")

        top_k = self.config.get("retrieval.top_k", 3)

        retriever = self.vector_manager.get_retriever(
            collection_name=collection_name,
            k=top_k,
        )
        return retriever

    def _build_answer_generator(self):
        """
        从 agents.yaml 中读取 agents.default_answer_generator 配置，
        构造 PromptTemplate → ChatTongyi → Parser 的标准链。
        """
        agents_cfg = YamlConfigReader(self.agent_config_path)
        cfg = agents_cfg.get("default_answer_generator")
        if cfg is None:
            raise ValueError(
                "agents.yaml 中缺少 default_answer_generator 配置"
            )

        model_name = cfg.get("model")
        if not model_name:
            raise ValueError("default_answer_generator 缺少 model 配置")

        temperature = float(cfg.get("temperature", 0.0))
        parser_type = cfg.get("parser", "str")
        inputs = cfg.get("inputs", ["question", "contexts", "memory_text"])
        prompt_template = cfg.get("prompt")
        if not prompt_template:
            raise ValueError("default_answer_generator 缺少 prompt 配置")

        api_key = os.environ.get("API_KEY_Qwen")
        if not api_key:
            raise ValueError(
                "未在环境变量 API_KEY_Qwen 中找到千问 API Key，"
                "请先设置：set API_KEY_Qwen=你的key"
            )

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=inputs,
        )

        # parser
        if parser_type == "json":
            parser = JsonOutputParser()
            llm_format = "json"
        else:
            parser = StrOutputParser()
            llm_format = None

        llm = ChatTongyi(
            model_name=model_name,
            dashscope_api_key=api_key,
            temperature=temperature,
            format=llm_format,
            enable_safety_check=False,
            extra_body={"disable_safety_check": True},
        )

        chain = prompt | llm | parser
        return chain

    def _build_workflow(self):
        """
        默认工作流：NormalRag。
        若需自定义工作流，可在子类中重写为其它 LangGraph 编排。
        """
        return NormalRag(
            retriever=self.retriever,
            answer_generator=self.answer_generator,
        )

    # ============================================================
    # 对外主接口
    # ============================================================

    def invoke(self, question: str):
        """
        对外唯一主接口：执行一条默认 RAG 流程。

        返回结构：
          {
              "question": str,
              "contexts": List[Document],
              "generation": str
          }
        """
        return self.rag.invoke(question)
