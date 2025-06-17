from langchain.schema import Document
from dotenv import load_dotenv
import pdfplumber
import fitz
import os
import pickle

from cleaning import clean_text, remove_footer
from utils import chunking, is_overlap, table_to_markdown

load_dotenv()

PDF_PATH = os.getenv("DOC_PATH")


def extract_text_and_tables(pdf_path):
    result = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.find_tables()
            table_bboxes = [table.bbox for table in tables]

            non_table_words = []
            for word in page.extract_words():
                # word bbox: (x0, top, x1, bottom)
                word_bbox = (float(word["x0"]), float(word["top"]), float(word["x1"]), float(word["bottom"]))
                
                # Check if word bbox overlaps with any table bbox
                is_table_word = any(is_overlap(word_bbox, bbox) for bbox in table_bboxes)
                if not is_table_word:
                    non_table_words.append(word["text"])

            paragraph_text = " ".join(non_table_words)

            extracted_tables = [table.extract() for table in tables]

            id_counter = 0
            if extracted_tables:
                for table in extracted_tables:
                    table_str = table_to_markdown(table)
                    result.append({
                        "chunk_index": len(result),
                        "chunk_id": f"{page_num}_{id_counter}",
                        "page_number": page_num,
                        "page_content": table_str,
                        "content_type": "table"
                    })
                    id_counter += 1
            
            txt = paragraph_text.strip()
            txt = clean_text(txt)
            txt = remove_footer(txt, page_num)

            chunked = chunking(txt)

            for chunk in chunked:
                result.append({
                    "chunk_index": len(result),
                    "chunk_id": f"{page_num}_{id_counter}",
                    "page_number": page_num,
                    "page_content": chunk,
                    "content_type": "text"
                })
                id_counter += 1

    return result


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
    extracted = extract_text_and_tables(pdf_path)
    metadata = extract_doc_metadata(pdf_path)

    return enrich_chunks_with_doc_metadata(extracted, metadata)


def save_as_documents(pdf_path):
    enriched_chunks = extract_with_metadata(pdf_path)

    documents = [
        Document(
            page_content=chunk['page_content'],
            metadata={
                'chunk_index': chunk['chunk_index'],
                'chunk_id': chunk['chunk_id'],
                'page_number': chunk['page_number'],
                'content_type': chunk['content_type'],
                'title': chunk['title'],
                'author': chunk['author'],
                'page_count': chunk['page_count']
            }
        )
        for chunk in enriched_chunks
    ]

    with open("data/chunks_metadata_with_type.pkl", 'wb') as file:
        pickle.dump(documents, file)



if __name__ == '__main__':
    save_as_documents(PDF_PATH)
