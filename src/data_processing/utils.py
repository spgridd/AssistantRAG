from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
import io
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types
import os

load_dotenv()

PROJECT_ID = os.getenv("GEMINI_PROJECT")
LOCATION = os.getenv("GEMINI_LOCATION")
USE_VERTEXAI = True


def chunking(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_text(text)


def is_overlap(word_bbox, table_bbox):
    wx0, wtop, wx1, wbot = word_bbox
    tx0, ttop, tx1, tbot = table_bbox

    horizontal_overlap = wx1 > tx0 and wx0 < tx1
    vertical_overlap = wbot > ttop and wtop < tbot
    return horizontal_overlap and vertical_overlap


def check_row(row):
    return [cell if cell is not None else " " for cell in row]


def escape_pipes(text):
    return text.replace('|', '\\|')


def clean_cell(cell):
    if cell is None:
        return "-"
    cell = str(cell).strip()
    if not cell:
        return "-"
    return escape_pipes(cell)


def table_to_markdown(table):
    if not table or not any(table):
        return "*(empty table)*"

    max_cols = max(len(row) for row in table)
    normalized_rows = []
    for row in table:
        row_extended = list(row) + ["-"] * (max_cols - len(row))
        normalized_rows.append([clean_cell(cell) for cell in row_extended])

    header = normalized_rows[0]
    separator = ["---"] * max_cols
    body = normalized_rows[1:] if len(normalized_rows) > 1 else []

    markdown = []
    markdown.append("| " + " | ".join(header) + " |")
    markdown.append("| " + " | ".join(separator) + " |")
    for row in body:
        markdown.append("| " + " | ".join(row) + " |")

    return "\n".join(markdown)


def get_client():
    client = genai.Client(
        vertexai=USE_VERTEXAI,
        project=PROJECT_ID,
        location=LOCATION,
    )
    return client


def describe_image(image_path):
    client = get_client()

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()

        parts = [
            types.Part(
                inline_data=types.Blob(
                    mime_type="image/png",
                    data=image_data
                )
            ),
            types.Part(text="Describe this image in detail.")
        ]

        contents = [types.Content(parts=parts, role="user")]

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )

        description = response.candidates[0].content.parts[0].text.strip()

    except Exception as e:
        description = f"Error analyzing image: {e}"

    return description
