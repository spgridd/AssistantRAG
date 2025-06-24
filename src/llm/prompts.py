from langchain.prompts import PromptTemplate


def get_doc_prompt():
    document_prompt_template = """
        Content: {page_content}
        
        Source Metadata:
            - Content Type: {content_type}
            - Title: {title}
            - Author: {author}
            - Page: {page_number}
    """

    document_prompt = PromptTemplate(
        template=document_prompt_template,
        input_variables=["page_content", "content_type", "title", "author", "page_number"]
    )

    return document_prompt


def get_qa_prompt():
    qa_prompt_template = """
        You are a helpful AI assistant. Use the following pieces of context and the chat history to answer the question at the end.
        Specify page from which you took the informations and tell whether it was a text or the table.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        CHAT HISTORY:
        {chat_history}

        CONTEXT:
        {context}

        QUESTION:
        {question}

        Comprehensive answer in correct markdown format:
    """

    qa_prompt = PromptTemplate(
        template=qa_prompt_template,
        input_variables=["context", "question", "chat_history"]
    )

    return qa_prompt


def get_filter_prompt():
    filter_prompt = """
        From given prompt retrieve three parameters:
        - content_type: 'text' or 'table' or 'any' (if not specified)
        - greater_than: lower bound for pages range (if not specified don't return this field)
        - less_than: upper bound for pages range (if not specified don't return this field)
    """

    return filter_prompt
