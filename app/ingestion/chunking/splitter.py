import logfire
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 1500 ) -> List[str]:
    """Split a text into chunks of a specified size."""

    with logfire.span("Text Chunking", text_length=len(text), chunk_size=chunk_size):
        if not text.strip():
            logfire.info("Empty text provided for chunking")
            return []

    splitter = RecursiveCharacterTextSplitter(\
        chunk_size=chunk_size,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""])

    chunks = splitter.split_text(text)

    valid_chunks = [chunk for chunk in chunks if chunk.strip()]

    logfire.info(f"Chunked text into {len(valid_chunks)} valid chunks.")

    return valid_chunks


if __name__ == "__main__":
    sample_text = "This is a sample text that will be chunked into smaller pieces. " * 100
    chunks = chunk_text(sample_text, chunk_size=1500)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:\n{chunk}\n")