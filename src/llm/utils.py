from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.vectorstores import Chroma
from pydantic import BaseModel, Field
from typing import Literal, Dict, Optional
from google.genai import types
import json

from wrappers.langchain_wrappers import VertexAIEmbedding
from genai_client.client import get_client
from llm.prompts import get_filter_prompt

load_dotenv()
FAISS_INDEX_DIR = "data/faiss_index_with_images"
CHROMA_DIR = "data/chroma_with_images"


class FilterSchema(BaseModel):
    content_type: Literal["table", "text", "image", "any"]
    greater_than: Optional[int]
    less_than: Optional[int]


def get_vector_store(documents):
    embeddings = VertexAIEmbedding()
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    vectorstore.save_local(FAISS_INDEX_DIR)
    return vectorstore


def get_chroma(documents):
    embeddings = VertexAIEmbedding()
    vectorstore =Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()
    return vectorstore


def extract_filters(prompt):
    client = get_client()

    contents = []
    contents.append(types.Content(parts=[types.Part(text=prompt)], role='user'))

    config = {
        "system_instruction": get_filter_prompt(),
        "temperature": 0.0,
        "response_schema": FilterSchema,
        "response_mime_type": "application/json"
    }

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=contents,
        config=config
    )

    return response.candidates[0].content.parts[0].text.strip()


def get_faiss_filter(user_prompt):
    filters = extract_filters(user_prompt)

    try:
        filters_dict = json.loads(filters)
    except json.JSONDecodeError:
        filters_dict = {}

    faiss_filter = {}

    if filters_dict.get("content_type") and filters_dict["content_type"] != "any":
        faiss_filter["content_type"] = filters_dict["content_type"]

    greater_than = filters_dict.get("greater_than")
    less_than = filters_dict.get("less_than")

    if greater_than is not None or less_than is not None:
        def page_number_filter(metadata):
            page = metadata.get("page_number", 0)
            return (greater_than is None or page > greater_than) and \
                   (less_than is None or page < less_than)
        return page_number_filter

    return faiss_filter
