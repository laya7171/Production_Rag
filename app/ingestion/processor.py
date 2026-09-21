import os
import sys
import uuid
import json
import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models
from torch import chunk

from app.config import settings
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.loaders.office import parse_office
from app.ingestion.chunking.splitter import chunk_text
from app.services.retrieval.embedding import embed_texts, get_embedding_dim

logfire.configure(service_name="enterprise-ingestion-service")

PROCESSED_DATA_DIR = "processed_data"

qdrant_client = QdrantClient(
    url = settings.QDRANT_URL,
    api_key = settings.QDRANT_API_KEY
)

def save_processed_locally(data:dict, source_type: str, filename: str)-> str:
    """Save processed data to a local Jsonfile"""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return dest


def process_file(file_path: str, filename: str, source_type: str):
    """Parse -> chunk -> embed -> store in Qdrant"""

    with logfire.span(
        "File Processing",
        file_path=file_path,
        filename=filename,
        source_type=source_type
    ):
        try:
            ext = filename.split(".")[-1].lower()

            if ext == "pdf":
                full_text = parse_pdf(file_path)

            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)

            elif ext in ("txt", "md"):
                full_text = parse_text(file_path)

            elif ext in ("docx", "doc", "xlsx", "xls", "pptx", "ppt"):
                full_text = parse_office(file_path)

            else:
                logfire.error(f"Unsupported file type: {ext}")
                return

            chunks = chunk_text(full_text)

            if not chunks:
                logfire.warning(
                    f"No valid chunks generated for file: {filename}"
                )
                return

            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks
            }

            local_path = save_processed_locally(
                processed_data,
                source_type,
                filename
            )

            logfire.info(
                f"Processed data saved locally at: {local_path}"
            )

            with logfire.span("Vectorizing & Indexing"):
                embeddings = embed_texts(chunks)

                points = [
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type
                        },
                    )
                    for chunk, vector in zip(chunks, embeddings)
                ]

            qdrant_client.upsert(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points=points
            )

            logfire.info(
                f"Indexed {len(points)} chunks into Qdrant collection: "
                f"{settings.QDRANT_COLLECTION_NAME}"
            )

        except Exception as e:
            logfire.error(
                f"Error processing file {filename}: {e}"
            )


def process_directory(directory_path: str, source_type: str):
    """Process all files in a directory."""
    with logfire.span("Scanning directory", path=directory_path, source_type=source_type):
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        logfire.info(f"Found {len(files)} files in directory: {directory_path}")
        for filename in files:
            process_file(os.path.join(directory_path, filename), filename, source_type)


def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False):
    """
    Scan base_dir for subdirectories, each representing a course types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion
    """

    with logfire.span("Universal Ingestion", base_dir=base_dir, explicit_source_type=explicit_source_type, wipe=wipe):
        if wipe:
            logfire.info(f"Wiping Qdrant collection: {settings.QDRANT_COLLECTION_NAME}")
            qdrant_client.delete_collection(settings.QDRANT_COLLECTION_NAME)
            qdrant_client.recreate_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(size=embed_texts(["test"])[0].__len__(), distance=models.Distance.COSINE)
            )
            logfire.info(f"Recreated Qdrant collection: {settings.QDRANT_COLLECTION_NAME}")

        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION_NAMED):
            dim = get_embedding_dim()
            qdrant_client.recreate_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )
            logfire.info(
                f"Created Qdrant collection: {settings.QDRANT_COLLECTION_NAME} with embedding dimension: {dim}"
            )
            
            subdirs = [
                d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))
            ]

            if not subdirs:
                if explicit_source_type:
                    source_type = explicit_source_type
                else:
                    base_name = os.path.basename(os.path.normpath(base_dir))
                    source_type = (
                        "true" if "true" in base_name.lower() else
                        "noisy" if "noisy" in base_name.lower() else
                        "general"
                    )
                    logfire.info(f"Detected source type: {source_type}")
                    process_directory(base_dir, source_type)

            else:
                for subdir in subdirs:
                    subdir_path = os.path.join(base_dir, subdir)
                    if explicit_source_type:
                        source_type = explicit_source_type
                    else:
                        source_type = (
                            "true" if "true" in subdir.lower() else
                            "noisy" if "noisy" in subdir.lower() else
                            "general"
                        )
                        logfire.info(f"Detected source type for {subdir}: {source_type}")
                    process_directory(subdir_path, source_type)


if __name__ == "__main__":
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]

    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_source_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        logfire.error(f"Target directory does not exist: {target_dir}")
        sys.exit(1)
    run_universal_ingestion(target_dir, explicit_source_type, wipe_requested)   
    logfire.info("Ingestion process completed.")