import pymupdf

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def load_and_split_document():

    document = pymupdf.open(PDF_PATH)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []

    for page_number, page in enumerate(document):

        text = page.get_text().strip()

        if not text:
            continue

        page_chunks = text_splitter.split_text(text)

        for chunk in page_chunks:

            chunks.append({
                "text": chunk,
                "page": page_number + 1,
                "source": "network_protocols.pdf"
            })

    document.close()

    return chunks