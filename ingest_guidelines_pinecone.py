"""
Pinecone Guidelines Ingestion Script

This script ingests medical guidelines from markdown files into Pinecone
for enterprise-mode guideline retrieval.

IMPORTANT: This is for PUBLIC medical guidelines only.
Patient data (PHI) should NEVER be sent to Pinecone.

Usage:
    python ingest_guidelines_pinecone.py

Requirements:
    - PINECONE_API_KEY environment variable
    - OPENAI_API_KEY environment variable

The script will:
1. Read guideline markdown files from data/medical_kb/
2. Generate embeddings using text-embedding-3-small
3. Create a Pinecone serverless index if it doesn't exist
4. Upsert all guidelines with metadata
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

load_dotenv()


# Configuration
INDEX_NAME = "careflow-guidelines"
NAMESPACE = "medical-kb"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
BATCH_SIZE = 100  # Pinecone recommends batches of 100 vectors


def load_guidelines_from_markdown(kb_dir: str = "data/medical_kb") -> List[Dict]:
    """Load guideline documents from markdown files.

    Args:
        kb_dir: Directory containing guideline markdown files

    Returns:
        List of document dicts with id, text, metadata
    """
    kb_path = Path(kb_dir)
    documents = []

    if not kb_path.exists():
        print(f"Error: {kb_dir} does not exist")
        return documents

    for md_file in sorted(kb_path.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")

        # Parse metadata from front matter if present
        metadata = {"source_file": md_file.name}

        # Extract guideline ID from filename
        guideline_id = md_file.stem  # e.g., "guideline_001_a1c_threshold"

        # Parse content for title and body
        lines = content.strip().split("\n")
        title = ""
        body_lines = []

        for line in lines:
            if line.startswith("# ") and not title:
                title = line[2:].strip()
                metadata["title"] = title
            elif line.startswith("## Category:"):
                metadata["category"] = line.replace("## Category:", "").strip()
            elif line.startswith("## Condition:"):
                metadata["condition"] = line.replace("## Condition:", "").strip()
            elif line.startswith("## Source:"):
                metadata["source"] = line.replace("## Source:", "").strip()
            elif not line.startswith("#"):
                body_lines.append(line)

        # Combine for searchable text
        text = f"{title}\n\n" + "\n".join(body_lines).strip()

        documents.append({
            "id": guideline_id,
            "text": text,
            "metadata": metadata
        })

    print(f"Loaded {len(documents)} guidelines from {kb_dir}")
    return documents


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using OpenAI.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    return [data.embedding for data in response.data]


def create_pinecone_index():
    """Create the Pinecone index if it doesn't exist.

    Returns:
        Pinecone Index object
    """
    from pinecone import Pinecone, ServerlessSpec

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")

    pc = Pinecone(api_key=api_key)

    # Check if index exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        # Wait for index to be ready
        print("Waiting for index to be ready...")
        while not pc.describe_index(INDEX_NAME).status.ready:
            time.sleep(1)
        print("Index ready!")
    else:
        print(f"Index '{INDEX_NAME}' already exists")

    return pc.Index(INDEX_NAME)


def ingest_guidelines(documents: List[Dict], index) -> int:
    """Ingest guidelines into Pinecone.

    Args:
        documents: List of guideline documents
        index: Pinecone Index object

    Returns:
        Number of documents ingested
    """
    if not documents:
        print("No documents to ingest")
        return 0

    print(f"Generating embeddings for {len(documents)} documents...")
    texts = [doc["text"] for doc in documents]
    embeddings = get_embeddings(texts)

    # Prepare vectors for upsert
    vectors = []
    for doc, embedding in zip(documents, embeddings):
        # Include text in metadata for retrieval
        metadata = doc["metadata"].copy()
        metadata["text"] = doc["text"]

        vectors.append({
            "id": doc["id"],
            "values": embedding,
            "metadata": metadata
        })

    # Upsert in batches
    print(f"Upserting to Pinecone namespace '{NAMESPACE}'...")
    total_upserted = 0

    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace=NAMESPACE)
        total_upserted += len(batch)
        print(f"  Upserted {total_upserted}/{len(vectors)} vectors")

    return total_upserted


def verify_ingestion(index) -> Dict:
    """Verify the ingestion by checking index stats.

    Args:
        index: Pinecone Index object

    Returns:
        Index statistics
    """
    stats = index.describe_index_stats()
    return {
        "total_vectors": stats.total_vector_count,
        "namespaces": {
            ns: data.vector_count
            for ns, data in stats.namespaces.items()
        }
    }


def main():
    """Main ingestion workflow."""
    print("=" * 60)
    print("CAREFLOW GUIDELINE INGESTION TO PINECONE")
    print("=" * 60)
    print()

    # Check environment variables
    missing_vars = []
    if not os.getenv("PINECONE_API_KEY"):
        missing_vars.append("PINECONE_API_KEY")
    if not os.getenv("OPENAI_API_KEY"):
        missing_vars.append("OPENAI_API_KEY")

    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nSet these in your .env file or environment.")
        sys.exit(1)

    # Load guidelines
    print("Loading guidelines from markdown files...")
    documents = load_guidelines_from_markdown()

    if not documents:
        print("No guidelines found. Exiting.")
        sys.exit(1)

    print()
    for doc in documents:
        print(f"  - {doc['id']}: {doc['metadata'].get('title', 'Untitled')}")
    print()

    # Create/connect to index
    print("Connecting to Pinecone...")
    try:
        index = create_pinecone_index()
    except Exception as e:
        print(f"Error connecting to Pinecone: {e}")
        sys.exit(1)

    print()

    # Ingest guidelines
    try:
        count = ingest_guidelines(documents, index)
        print(f"\nSuccessfully ingested {count} guidelines")
    except Exception as e:
        print(f"Error during ingestion: {e}")
        sys.exit(1)

    # Verify
    print("\nVerifying ingestion...")
    stats = verify_ingestion(index)
    print(f"  Total vectors in index: {stats['total_vectors']}")
    print(f"  Namespaces: {stats['namespaces']}")

    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print()
    print("To use enterprise mode in CareFlow:")
    print("  python care_app.py --mode=enterprise")
    print()
    print("Or set mode in code:")
    print("  from guideline_retriever import GuidelineRetriever, RetrievalMode")
    print("  retriever = GuidelineRetriever(mode=RetrievalMode.ENTERPRISE)")


if __name__ == "__main__":
    main()
