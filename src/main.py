import os
from dotenv import load_dotenv

from pdf_chunker import extract_with_metadata

PDF_PATH = os.getenv("DOC_PATH")

chunked_pdf = extract_with_metadata(PDF_PATH)
