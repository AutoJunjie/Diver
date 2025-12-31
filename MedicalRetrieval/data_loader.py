"""
Data loader for medical literature CSV from S3.
"""
import os
import ast
import csv
import boto3
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class Document:
    """Represents a medical literature document."""
    doc_id: int
    title: str
    abstract: str
    keywords: List[str]

    @property
    def full_text(self) -> str:
        """Concatenate title, abstract, and keywords for embedding/prompts."""
        keywords_str = ", ".join(self.keywords) if self.keywords else ""
        return f"{self.title}\n\n{self.abstract}\n\n关键词: {keywords_str}"

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "abstract": self.abstract,
            "keywords": self.keywords
        }


def parse_keywords(keywords_str: str) -> List[str]:
    """Parse keywords string which is stored as a Python list representation."""
    if not keywords_str or keywords_str.strip() == "":
        return []
    try:
        # Keywords are stored as string representation of Python list
        return ast.literal_eval(keywords_str)
    except (ValueError, SyntaxError):
        # Fallback: split by comma if not a valid list
        return [k.strip() for k in keywords_str.split(",") if k.strip()]


def download_from_s3(s3_path: str, local_path: Optional[str] = None) -> str:
    """
    Download a file from S3.

    Args:
        s3_path: S3 URI (s3://bucket/key)
        local_path: Optional local path. If None, uses a temp file.

    Returns:
        Local file path
    """
    # Parse S3 URI
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}")

    path_parts = s3_path[5:].split("/", 1)
    bucket = path_parts[0]
    key = path_parts[1] if len(path_parts) > 1 else ""

    if local_path is None:
        # Create temp file with same extension
        ext = os.path.splitext(key)[1]
        fd, local_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)

    print(f"Downloading from s3://{bucket}/{key} to {local_path}")
    s3_client = boto3.client("s3")
    s3_client.download_file(bucket, key, local_path)
    print(f"Download complete: {local_path}")

    return local_path


def load_medical_documents(
    s3_path: str,
    local_cache_path: Optional[str] = None,
    max_documents: Optional[int] = None
) -> List[Document]:
    """
    Load medical literature documents from S3 CSV.

    Args:
        s3_path: S3 URI to the CSV file
        local_cache_path: Optional local path to cache the CSV
        max_documents: Optional limit on number of documents to load

    Returns:
        List of Document objects
    """
    # Download if needed
    if local_cache_path and os.path.exists(local_cache_path):
        print(f"Using cached file: {local_cache_path}")
        csv_path = local_cache_path
    else:
        csv_path = download_from_s3(s3_path, local_cache_path)

    documents = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for idx, row in enumerate(reader):
            if max_documents and idx >= max_documents:
                break

            doc = Document(
                doc_id=idx,
                title=row.get("title", ""),
                abstract=row.get("abstract", ""),
                keywords=parse_keywords(row.get("keywords", ""))
            )
            documents.append(doc)

    print(f"Loaded {len(documents)} documents")
    return documents


def get_documents_by_ids(documents: List[Document], doc_ids: List[int]) -> List[Document]:
    """Get documents by their IDs."""
    doc_map = {d.doc_id: d for d in documents}
    return [doc_map[did] for did in doc_ids if did in doc_map]


def create_document_index(documents: List[Document]) -> Dict[int, Document]:
    """Create a document ID to document mapping."""
    return {d.doc_id: d for d in documents}


if __name__ == "__main__":
    # Test loading
    import json

    with open("config.json") as f:
        config = json.load(f)

    docs = load_medical_documents(
        config["s3_input_path"],
        local_cache_path="cache/medical_literature.csv",
        max_documents=10
    )

    for doc in docs[:3]:
        print(f"ID: {doc.doc_id}")
        print(f"Title: {doc.title}")
        print(f"Keywords: {doc.keywords}")
        print(f"Abstract: {doc.abstract[:200]}...")
        print("-" * 50)
