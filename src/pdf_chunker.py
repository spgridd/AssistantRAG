from pdfminer.high_level import extract_text_to_fp
from pdfminer.pdfpage import PDFPage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import StringIO
import re
import fitz
from dotenv import load_dotenv
import os

load_dotenv()

TITLE = os.getenv("DOC_TITLE")

# Extraction

def extract_pages(pdf_path):
    pages = []
    with open(pdf_path, 'rb') as f:
        for i, _ in enumerate(PDFPage.get_pages(f)):
            output_string = StringIO()
            extract_text_to_fp(f, output_string, page_numbers=[i])
            text = output_string.getvalue()
            pages.append({'page_number': i, 'text': text})
    return pages


# Chunking
## Cleaning

def clean_text(text):
    text = re.sub(r'\n\n\n+', '\n\n\n', text)          # multiple newlines
    text = re.sub(r'\s+', ' ', text)                   # whitespace
    text = text.strip()
    return text


def remove_footer(text, page_num, title):
    escaped_title = re.escape(title.strip())

    pattern1 = rf"\s*{page_num}\s+{escaped_title}\s*$"
    pattern2 = rf"\s*{escaped_title}\s+{page_num}\s*$"

    text = re.sub(pattern1, '', text)
    text = re.sub(pattern2, '', text)
    
    return text.rstrip()


## Actual chunking (with metadata)

def chunk_with_metadata(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    all_chunks = []
    for page in pages:
        text_no_footer = remove_footer(page['text'], str(page['page_number']), TITLE)
        clean = clean_text(text_no_footer)
        chunks = splitter.split_text(clean)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'chunk_index': len(all_chunks),
                'chunk_id': f"{page['page_number']}_{i}",
                'text': chunk,
                'page_number': page['page_number']
            })
    return all_chunks


def extract_doc_metadata(pdf_path):
    doc = fitz.open(pdf_path)
    metadata = doc.metadata
    return {
        "title": metadata.get("title"),
        "author": metadata.get("author"),
        "page_count": doc.page_count,
    }


def enrich_chunks_with_doc_metadata(chunks, doc_meta):
    for chunk in chunks:
        chunk.update(doc_meta)
    return chunks


def extract_with_metadata(pdf_path):
    extracted = extract_pages(pdf_path)
    meta_chunks = chunk_with_metadata(extracted)
    metadata = extract_doc_metadata(pdf_path)

    return enrich_chunks_with_doc_metadata(meta_chunks, metadata)