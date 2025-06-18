import os
import pickle
import streamlit as st
from langchain_community.vectorstores import FAISS

from wrappers.langchain_wrappers import VertexAIEmbedding
from llm.utils import get_vector_store
from llm.chain import get_conversation_chain


with open("data/chunks_metadata_with_type.pkl", 'rb') as file:
    DOCUMENTS = pickle.load(file)

FAISS_INDEX_DIR = "data/faiss_index_with_type"


def handle_user_input(question):
    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history = response["chat_history"]


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
        try:
            with st.spinner(text="In progress...", show_time=True):
                if os.path.exists(FAISS_INDEX_DIR):
                    embeddings = VertexAIEmbedding()
                    st.session_state.vector_store = FAISS.load_local(
                        FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
                    )
                    st.success("Vector store loaded from disk.")
                else:
                    st.session_state.vector_store = get_vector_store(DOCUMENTS)
                    st.success("Vector store created and saved.")
        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {e}")
            return

    if st.session_state.conversation is None and st.session_state.vector_store:
        st.session_state.conversation = get_conversation_chain(
            st.session_state.vector_store
        )

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

    with st.sidebar:
        st.header("Session Details")

        with st.expander("Prompt Sent to LLM"):
            st.code(st.session_state.add_info_prompt, language="markdown")

        with st.expander("Retrieved Context"):
            st.markdown("Chunks retrieved by the vector search:")
            st.code(st.session_state.retrieved, language="markdown")



if __name__ == "__main__":
    main()
