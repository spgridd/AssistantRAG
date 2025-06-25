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

        Comprehensive answer in well-formated, readible markdown format:
    """

    qa_prompt = PromptTemplate(
        template=qa_prompt_template,
        input_variables=["context", "question", "chat_history"]
    )

    return qa_prompt


def get_filter_prompt():
    filter_prompt = """
        Your task is to analyze a user's prompt and extract specific filtering criteria in a strict JSON format. Follow these rules precisely.

        **1. Rules for `content_type`**

        - The default value is ALWAYS 'any'.
        - You MUST NOT change the `content_type` to 'table', 'image', or 'text' based on the topic of the query. For example, if the user asks about financial data, you must not assume it's a table.
        - You should ONLY set the `content_type` to a specific value if the user's prompt contains explicit keywords for that type.

        - **Keywords for 'table'**: "table", "tables", "tabulated", "tabular data", "in the data".
        - **Keywords for 'image'**: "image", "images", "figure", "picture", "diagram", "graph", "chart".
        - **Keywords for 'text'**: "paragraph", "sentence", "text", "prose", "in the text".

        **2. Rules for Page Ranges (`greater_than`, `less_than`)**

        - Extract a lower bound for a page range if the user specifies one (e.g., "after page 10", "from page 7 onwards").
        - Extract an upper bound for a page range if the user specifies one (e.g., "before page 20", "up to page 15").
        - If the user specifies an exact page (e.g., "on page 12"), do not return a range.

        **3. Examples**

        Here are some examples of how to apply these rules:

        ---
        User: "What was the change in the Net Income Loss from FY22 to FY24?"
        Assistant:
        {
        "content_type": "any"
        }
        ---
        User: "Show me the table with the net income loss from FY22 to FY24."
        Assistant:
        {
        "content_type": "table"
        }
        ---
        User: "Summarize the document after page 30."
        Assistant:
        {
        "content_type": "any",
        "greater_than": 30
        }
        ---
        User: "Find the main diagram of the system architecture before page 15."
        Assistant:
        {
        "content_type": "image",
        "less_than": 15
        }
        ---
        User: "What does the text say about risk factors between page 5 and 10?"
        Assistant:
        {
        "content_type": "text",
        "greater_than": 5,
        "less_than": 10
        }
        ---

        Now, analyze the following user prompt and provide only the JSON output.
    """
    return filter_prompt
