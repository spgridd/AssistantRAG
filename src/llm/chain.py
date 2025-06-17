from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from wrappers.langchain_wrappers import VertexAIChat


GEMINI_MODEL_NAME = "gemini-2.0-flash"


def get_conversation_chain(vector_store):
    llm = VertexAIChat(model=GEMINI_MODEL_NAME, temperature=0.0)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    system_template = """
        Use the following context and chat history to answer the question.
        If the answer is found in the context, cite where (e.g. page or paragraph).
        If the answer isn't found in the context, say you don't know.

        Context: {context}

        Chat history: {chat_history}

        Question: {question}

        Helpful Answer:
    """

    prompt = PromptTemplate(
        template=system_template,
        input_variables=["context", "question", "chat_history"],
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        verbose=True,
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

    return conversation_chain

