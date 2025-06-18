from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from wrappers.langchain_wrappers import VertexAIChat, CrossEncoderReRanker


GEMINI_MODEL_NAME = "gemini-2.0-flash"


def get_conversation_chain(vector_store):
    llm = VertexAIChat(model=GEMINI_MODEL_NAME, temperature=0.0)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    document_prompt_template = """
        Content: {page_content}
        
        Source Metadata:
            - Content Type: {content_type}
            - Title: {title}
            - Author: {author}
            - Page: {page_number}
    """

    DOCUMENT_PROMPT = PromptTemplate(
        template=document_prompt_template,
        input_variables=["page_content", "content_type", "title", "author", "page_number"]
    )

    qa_template = """
        You are a helpful AI assistant. Use the following pieces of context and the chat history to answer the question at the end.
        Specify page from which you took the informations and tell whether it was a text or the table.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        CHAT HISTORY:
        {chat_history}

        CONTEXT:
        {context}

        QUESTION:
        {question}

        Helpful answer in markdown format:
    """

    QA_PROMPT = PromptTemplate(
        template=qa_template,
        input_variables=["context", "question", "chat_history"]
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        verbose=True,
        combine_docs_chain_kwargs={
            "prompt": QA_PROMPT,
            "document_prompt": DOCUMENT_PROMPT
        }
    )

    return conversation_chain

