from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from sentence_transformers import CrossEncoder
import json

from wrappers.langchain_wrappers import VertexAIChat, CrossEncoderReRanker
from llm.prompts import get_doc_prompt, get_qa_prompt
from llm.utils import get_faiss_filter


GEMINI_MODEL_NAME = "gemini-2.0-flash"


def get_conversation_chain(vector_store, re_ranker, faiss, user_prompt):
    llm = VertexAIChat(model=GEMINI_MODEL_NAME, temperature=0.0)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    document_prompt = get_doc_prompt()
    qa_prompt = get_qa_prompt()

    # For Chroma only
    metadata_field_info = [
        AttributeInfo(
            name='content_type',
            description='Type of the retrieved content: text or table',
            type='string'
        ),
        AttributeInfo(
            name='title',
            description='Title of the document.',
            type='string'
        ),
        AttributeInfo(
            name='author',
            description='Author of the document.',
            type='string'
        ),
        AttributeInfo(
            name='page_number',
            description='Number of the page from which chunk was retrieved.',
            type='integer'
        )
    ]

    # Enable/Disable reranker
    if re_ranker:
        if faiss:
            faiss_filter = get_faiss_filter(user_prompt)
            base_retriever = vector_store.as_retriever(search_kwargs={'k': 10, 'filter': faiss_filter})
        else:
            base_retriever = SelfQueryRetriever.from_llm(
                llm=llm,
                vectorstore=vector_store,
                document_contents='This is a passage from a technical document or table, which includes information about the topic, author, and page number.',
                metadata_field_info=metadata_field_info,
                search_kwargs={'k': 50}
            )
        cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        retriever = CrossEncoderReRanker(
            retriever=base_retriever, 
            model=cross_encoder_model, 
            top_n=5
        )
    else:
        if faiss:
            faiss_filter = get_faiss_filter(user_prompt)
            retriever = vector_store.as_retriever(search_kwargs={'k': 10, 'filter': faiss_filter})
        else:
            retriever = SelfQueryRetriever.from_llm(
                llm=llm,
                vectorstore=vector_store,
                document_contents='A chunk of text from a document.',
                metadata_field_info=metadata_field_info,
                search_kwargs={'k': 50}
            )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True,
        combine_docs_chain_kwargs={
            "prompt": qa_prompt,
            "document_prompt": document_prompt
        }
    )

    return conversation_chain, retriever

