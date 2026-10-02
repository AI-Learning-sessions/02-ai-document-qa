import os
import pymupdf

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def load_and_split_document():

    """
    Load the PDF and split its text into chunks.

    Returns:
        A list of document chunks with source and page metadata.
    """

    if not os.path.exists(PDF_PATH):
          raise FileNotFoundError(
              f"PDF file not found: {PDF_PATH}"
          )

    document = pymupdf.open(PDF_PATH)

    if len(document) == 0:
        document.close()

        raise ValueError(
            "The PDF document contains no pages."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []

    for page_number, page in enumerate(document):

        text = page.get_text().strip()

        if not text:
            continue

        page_chunks = text_splitter.split_text(
            text
        )

        for chunk in page_chunks:

            chunks.append({
                "text": chunk,
                "page": page_number + 1,
                "source": os.path.basename(PDF_PATH)
            })

    document.close()

    if not chunks:
        raise ValueError(
            "No text chunks were extracted from the PDF."
        )

    return chunks