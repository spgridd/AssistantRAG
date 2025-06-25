from langchain.schema import Document
from dotenv import load_dotenv
import google.genai as genai
import pdfplumber
import fitz
import os
import pickle

from cleaning import clean_text, remove_footer
from utils import chunking, is_overlap, table_to_markdown, describe_image

load_dotenv()

PDF_PATH = os.getenv("DOC_PATH")
IMG_PATH = "data/images"


def extract_pdf_content(pdf_path):
    result = []

    os.makedirs(IMG_PATH, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        chunk_index = 0

        for page_num, page in enumerate(pdf.pages):
            tables = page.find_tables()
            table_bboxes = [table.bbox for table in tables]

            non_table_words = []
            for word in page.extract_words():
                word_bbox = (float(word["x0"]), float(word["top"]), float(word["x1"]), float(word["bottom"]))
                if not any(is_overlap(word_bbox, bbox) for bbox in table_bboxes):
                    non_table_words.append(word["text"])

            paragraph_text = " ".join(non_table_words)
            extracted_tables = [table.extract() for table in tables]

            # Add tables
            for table_index, table in enumerate(extracted_tables):
                table_str = table_to_markdown(table)
                result.append({
                    "chunk_index": chunk_index,
                    "chunk_id": f"{page_num}_table_{table_index}",
                    "page_number": page_num,
                    "page_content": table_str,
                    "content_type": "table"
                })
                chunk_index += 1

            # Add text chunks
            txt = clean_text(paragraph_text.strip())
            txt = remove_footer(txt, page_num)
            chunked = chunking(txt)

            for text_index, chunk in enumerate(chunked):
                result.append({
                    "chunk_index": chunk_index,
                    "chunk_id": f"{page_num}_text_{text_index}",
                    "page_number": page_num,
                    "page_content": chunk,
                    "content_type": "text"
                })
                chunk_index += 1

    # Extract images
    pdf_document = fitz.open(pdf_path)
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list, start=1):
            base_image = pdf_document.extract_image(img[0])
            image_data = base_image["image"]
            image_ext = base_image["ext"]

            image_name = f"{file_name}_page_{page_num + 1}_image_{img_index}.{image_ext}"
            image_path = os.path.join(IMG_PATH, image_name)

            with open(image_path, "wb") as f:
                f.write(image_data)

            description = describe_image(image_path)

            result.append({
                "chunk_index": chunk_index,
                "chunk_id": f"{page_num}_img_{img_index}",
                "page_number": page_num,
                "page_content": description,
                "content_type": "image"
            })
            chunk_index += 1

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
    extracted = extract_pdf_content(pdf_path)
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
