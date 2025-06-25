import os
import pickle
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.vectorstores import Chroma

from wrappers.langchain_wrappers import VertexAIEmbedding
from llm.utils import get_vector_store, get_chroma
from llm.chain import get_conversation_chain


with open("data/chunks_metadata_with_images.pkl", 'rb') as file:
    DOCUMENTS = pickle.load(file)

FAISS_INDEX_DIR = "data/faiss_index_with_images"
CHROMA_DIR = "data/chroma_with_images"


def handle_user_input(question):
    chain, retriever = get_conversation_chain(
        vector_store=st.session_state.vector_store,
        re_ranker=st.session_state.reranker,
        faiss=st.session_state.use_faiss,
        user_prompt=question
    )
    st.session_state.conversation = chain

    response = chain({"question": question})
    st.session_state.chat_history = response["chat_history"]

    try:
        retrieved_docs = retriever.get_relevant_documents(question)
        st.session_state.retrieved = retrieved_docs
    except Exception as e:
        st.warning(f"Failed to retrieve chunks: {e}")
        st.session_state.retrieved = []



def display_chat_history():
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg.type == "human":
                with st.chat_message("user"):
                    st.markdown(msg.content)
            elif msg.type == "ai":
                with st.chat_message("assistant"):
                    st.markdown(msg.content)


def main():
    st.set_page_config(page_title="PDF Assistant")
    st.header("PDF Assistant")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "add_info_prompt" not in st.session_state:
        st.session_state.add_info_prompt = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "retrieved" not in st.session_state:
        st.session_state.retrieved = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "reranker" not in st.session_state:
        st.session_state.reranker = False
    if "use_faiss" not in st.session_state:
        st.session_state.use_faiss = True
    if "last_vector_store_type" not in st.session_state:
        st.session_state.last_vector_store_type = None

    # SIDEBAR OPTIONS
    with st.sidebar:
        st.header("Additional Options")
        reranker_enabled = st.toggle("Enable Re-Ranker", value=st.session_state.reranker)
        faiss_enabled = st.toggle("Use FAISS", value=st.session_state.use_faiss)

    st.session_state.reranker = reranker_enabled
    st.session_state.use_faiss = faiss_enabled

    current_vector_type = "faiss" if st.session_state.use_faiss else "chroma"
    if (
        st.session_state.vector_store is None
        or st.session_state.last_vector_store_type != current_vector_type
    ):
        try:
            with st.spinner("Loading vector store..."):
                embeddings = VertexAIEmbedding()
                if st.session_state.use_faiss:
                    if os.path.exists(FAISS_INDEX_DIR):
                        st.session_state.vector_store = FAISS.load_local(
                            FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
                        )
                        st.success("FAISS vector store loaded.")
                    else:
                        st.session_state.vector_store = get_vector_store(DOCUMENTS)
                        st.success("FAISS vector store created.")
                else:
                    if os.path.exists(CHROMA_DIR):
                        st.session_state.vector_store = Chroma(
                            embedding_function=embeddings, persist_directory=CHROMA_DIR
                        )
                        st.success("Chroma vector store loaded.")
                    else:
                        st.session_state.vector_store = get_chroma(DOCUMENTS)
                        st.success("Chroma vector store created.")

                st.session_state.last_vector_store_type = current_vector_type

        except Exception as e:
            st.error(f"An error occurred while loading vector store: {e}")
            return

    display_chat_history()

    if prompt := st.chat_input("Ask anything about your PDF:"):
        with st.chat_message("user"):
            st.markdown(prompt)
        handle_user_input(prompt)
        if st.session_state.chat_history:
            last_ai_response = [
                msg for msg in st.session_state.chat_history if msg.type == "ai"
            ][-1].content
            with st.chat_message("assistant"):
                st.markdown(last_ai_response)
            if st.session_state.retrieved:
                with st.expander("Retrieved Chunks", expanded=False):
                    for i, doc in enumerate(st.session_state.retrieved, 1):
                        metadata = doc.metadata
                        chunk = doc.page_content
                        st.markdown(f"**Chunk {i}** (page: {metadata.get('page_number', '?')}, type: {metadata.get('content_type', '?')}):")
                        st.code(chunk.strip(), language='markdown')



if __name__ == "__main__":
    main()
