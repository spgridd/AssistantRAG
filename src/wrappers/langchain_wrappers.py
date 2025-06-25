import os
from dotenv import load_dotenv
from pydantic import BaseModel, PrivateAttr

import google.genai as genai
from google.genai import types
from google.genai.types import Content, Part
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
import torch
from sentence_transformers import CrossEncoder
from typing import List, Callable, Any, Tuple

from genai_client.client import get_client


load_dotenv()


PROJECT_ID = os.getenv("GEMINI_PROJECT")
LOCATION = os.getenv("GEMINI_LOCATION")


CLIENT = get_client()

# Embbedding wrapper
class VertexAIEmbedding(Embeddings):
    def __init__(self, client: genai.Client = CLIENT, model: str = "models/text-embedding-004"):
        self.client = client
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            self._embed_text(text)
            for text in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> list[float]:
        contents = Content(parts=[Part(text=text)])
        response = self.client.models.embed_content(
            model=f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/text-embedding-004",
            contents=contents,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        return response.embeddings[0].values


# LLM wrapper
class VertexAIChat(BaseChatModel, BaseModel):
    model: str
    temperature: float = 0.0
    _client: genai.Client = PrivateAttr()

    def __init__(self, client: genai.Client = CLIENT, **kwargs):
        super().__init__(**kwargs)
        self._client = client

    def _generate(self, messages: list[BaseMessage], stop=None, **kwargs) -> ChatResult:
        contents = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "model"
            contents.append(Content(parts=[Part(text=msg.content)], role=role))

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config={"temperature": self.temperature}
        )
        return ChatResult(
            generations=[
                {
                    "text": response.text,
                    "message": AIMessage(content=response.text)
                }
            ]
        )

    def _create_chat_result(self, text: str):
        return self._to_chat_result(AIMessage(content=text))

    def _to_chat_result(self, message: AIMessage):
        from langchain_core.outputs import ChatResult, ChatGeneration
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "vertexai-chat"


#Re-ranking Retriever Wrapper
class CrossEncoderReRanker(BaseRetriever):
    retriever: BaseRetriever
    model: CrossEncoder
    top_n: int

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        initial_docs = self.retriever.get_relevant_documents(query, callbacks=run_manager.get_child())
        
        if not initial_docs:
            return []
        
        doc_pairs = [[query, doc.page_content] for doc in initial_docs]

        with torch.no_grad():
            scores = self.model.predict(doc_pairs)
        
        docs_with_scores = list(zip(initial_docs, scores))
        sorted_docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
        
        reranked_docs = [doc for doc, score in sorted_docs_with_scores[:self.top_n]]
        
        return reranked_docs
    

class HybridRetriever(BaseRetriever):
    vector_store: FAISS
    get_faiss_filter_func: Callable[[str, str], Tuple[Any, bool]]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        _, any_flag = self.get_faiss_filter_func(query)

        if any_flag:
            print("INFO: HybridRetriever performing hybrid search for general query.")
            
            retrieval_plan = [
                {'content_type': 'text', 'k': 10},
                {'content_type': 'table', 'k': 10},
                {'content_type': 'image', 'k': 5},
            ]
            
            all_retrieved_docs = []
            for plan in retrieval_plan:
                content_type, k = plan['content_type'], plan['k']
                
                specific_filter, _ = self.get_faiss_filter_func(query, force_content_type=content_type)
                
                temp_retriever = self.vector_store.as_retriever(
                    search_kwargs={'k': k, 'filter': specific_filter}
                )
                
                all_retrieved_docs.extend(temp_retriever.invoke(query))

            unique_docs = {}
            for doc in all_retrieved_docs:
                if 'chunk_id' in doc.metadata:
                    unique_docs[doc.metadata['chunk_id']] = doc
            
            return list(unique_docs.values())

        else:
            print("INFO: HybridRetriever performing search for specific content type.")
            
            specific_filter, _ = self.get_faiss_filter_func(query)
            
            retriever = self.vector_store.as_retriever(
                search_kwargs={'k': 25, 'filter': specific_filter}
            )
            return retriever.invoke(query)
