from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from wrappers.langchain_wrappers import VertexAIEmbedding

load_dotenv()
FAISS_INDEX_DIR = "data/faiss_index"


def get_vector_store(documents):
    embeddings = VertexAIEmbedding()
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    vectorstore.save_local(FAISS_INDEX_DIR)
    return vectorstore