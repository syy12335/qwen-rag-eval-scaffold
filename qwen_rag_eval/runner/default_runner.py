# qwen_rag_eval/runner/default_runner.py

import os

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_models import ChatTongyi

from utils import YamlConfigReader
from qwen_rag_eval.vector.vector_store_manager import VectorStoreManager
from qwen_rag_eval.runner.normal_rag import NormalRag


class DefaultRunner:
    """
    DefaultRunner：当前项目默认的 RAG 运行组件。

    职责：
      1. 读取 application.yaml 与 agents.yaml 中的必要配置。
      2. 使用 VectorStoreManager 构造 retriever。
      3. 使用 default_answer_generator 配置构造回答生成链。
      4. 组装 NormalRag 工作流，对外只暴露 invoke(question)。

    不负责：
      1. 构建或检查向量库（由 vector_builder 负责）。
      2. 修改 RAG 工作流结构（由 NormalRag 负责）。

    使用方式：
      runner = DefaultRunner()
      out = runner.invoke("问题")
    """

    def __init__(
        self,
        config_path: str = "config/application.yaml",
        agent_config_path: str = "config/agents.yaml",
    ):
        self.config_path = config_path
        self.agent_config_path = agent_config_path

        self.config = YamlConfigReader(config_path)
        self.vector_manager = VectorStoreManager(self.config)

        self.retriever = self._build_retriever()
        self.answer_generator = self._build_answer_generator()
        self.rag = self._build_workflow()

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
        从 agents.yaml 中读取 default_answer_generator 配置，
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
        中阶用户如果想换成自己的工作流，可以在高阶 runner 中重写这一部分。
        """
        return NormalRag(
            retriever=self.retriever,
            answer_generator=self.answer_generator,
        )

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
