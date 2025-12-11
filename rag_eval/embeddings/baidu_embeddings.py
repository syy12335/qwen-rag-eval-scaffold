# rag_eval/embeddings/baidu_embeddings.py

from __future__ import annotations
import os
from typing import List

from langchain_core.embeddings import Embeddings
from openai import OpenAI


class BaiduAIStudioEmbeddings(Embeddings):
    """
    通过 OpenAI 客户端调用百度 AI Studio embedding 接口的 LangChain 封装。
    """

    def __init__(
        self,
        model: str = "embedding-v1",
        api_key: str | None = None,
        base_url: str = "https://aistudio.baidu.com/llm/lmapi/v3",
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("AI_STUDIO_API_KEY")
        if not self.api_key:
            raise ValueError("缺少 AI Studio API Key，请设置环境变量 AI_STUDIO_API_KEY")

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]
