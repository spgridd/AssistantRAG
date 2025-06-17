from dotenv import load_dotenv
import google.genai as genai
import os


load_dotenv()

PROJECT_ID = os.getenv("GEMINI_PROJECT")
LOCATION = os.getenv("GEMINI_LOCATION")
USE_VERTEXAI = True
GEMINI_MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
FAISS_INDEX_DIR = "data/faiss_index"


def get_client():
    client = genai.Client(
        vertexai=USE_VERTEXAI,
        project=PROJECT_ID,
        location=LOCATION,
    )
    return client