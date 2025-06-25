from langchain.schema import Document
from dotenv import load_dotenv
import google.genai as genai
import pdfplumber
import fitz
import os
import pickle

from cleaning import clean_text, remove_footer
from utils import chunking, is_overlap, table_to_markdown, describe_image
from mapping import get_image_pages

load_dotenv()

PDF_PATH = os.getenv("DOC_PATH")
IMG_PATH = "data/images"


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
                        "chunk_id": f"{page_num}_tab_{id_counter}",
                        "page_number": page_num,
                        "page_content": table_str,
                        "content_type": "table"
                    })
                    id_counter += 1
            
            txt = paragraph_text.strip()
            txt = clean_text(txt)
            txt = remove_footer(txt, page_num)

            chunked = chunking(txt)

            id_counter = 0
            for chunk in chunked:
                result.append({
                    "chunk_index": len(result),
                    "chunk_id": f"{page_num}_txt_{id_counter}",
                    "page_number": page_num,
                    "page_content": chunk,
                    "content_type": "text"
                })
                id_counter += 1

    return result


def extract_images(pdf_path, result, pages, dpi=150):
    output_dir = IMG_PATH
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_document = fitz.open(pdf_path)
    
    chunk_index = len(result)

    for fig_num, page_num in pages.items():
        image_path = os.path.join(output_dir, f"page_{page_num}.png")

        if not os.path.exists(image_path):
            print(f"INFO: Image '{image_path}' not found. Rendering page {page_num}...")
            if 0 < page_num <= pdf_document.page_count:
                page = pdf_document.load_page(page_num)
                
                zoom = dpi / 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                pix.save(image_path)
            else:
                print(f"ERROR: Page number {page_num} is out of range. Skipping figure {fig_num}.")
                continue
        else:
            print(f"INFO: Found existing image for page {page_num} at '{image_path}'.")

        description = describe_image(image_path, fig_num)

        chunk_index += 1
        result.append({
            "chunk_index": chunk_index,
            "chunk_id": f"{chunk_index}_img_{fig_num}",
            "page_number": page_num,
            "page_content": description,
            "content_type": "image"
        })

    pdf_document.close()
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
    pages = get_image_pages()
    extracted = extract_images(pdf_path, extracted, pages)
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

    with open("data/chunks_metadata_with_images.pkl", 'wb') as file:
        pickle.dump(documents, file)



if __name__ == '__main__':
    save_as_documents(PDF_PATH)
