from dotenv import load_dotenv
import google.genai as genai
import os


load_dotenv()

PROJECT_ID = os.getenv("GEMINI_PROJECT")
LOCATION = os.getenv("GEMINI_LOCATION")
USE_VERTEXAI = True


def get_client():
    client = genai.Client(
        vertexai=USE_VERTEXAI,
        project=PROJECT_ID,
        location=LOCATION,
    )
    return client