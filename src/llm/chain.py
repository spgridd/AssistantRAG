from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from sentence_transformers import CrossEncoder

from wrappers.langchain_wrappers import VertexAIChat, CrossEncoderReRanker
from llm.prompts import get_doc_prompt, get_qa_prompt


GEMINI_MODEL_NAME = "gemini-2.0-flash"


def get_conversation_chain(vector_store, re_ranker):
    llm = VertexAIChat(model=GEMINI_MODEL_NAME, temperature=0.0)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    document_prompt = get_doc_prompt()
    qa_prompt = get_qa_prompt()

    # Enable/Disable reranker
    if re_ranker:
        base_retriever = vector_store.as_retriever(search_kwargs={'k': 20})
        cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        retreiver = CrossEncoderReRanker(
            retriever=base_retriever, 
            model=cross_encoder_model, 
            top_n=5
        )
    else:
        retreiver = vector_store.as_retriever(search_kwargs={'k': 10})

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retreiver,
        memory=memory,
        verbose=True,
        combine_docs_chain_kwargs={
            "prompt": qa_prompt,
            "document_prompt": document_prompt
        }
    )

    return conversation_chain

