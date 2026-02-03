"""Ingest medical knowledge base into the vector store."""

# TODO: Implement medical KB ingestion
# - Read guidelines from data/medical_kb/
# - Chunk and embed guidelines
# - Store in FAISS index


def ingest_medical_kb(kb_dir: str = "data/medical_kb"):
    """Ingest medical knowledge base into vector store.

    Args:
        kb_dir: Directory containing medical knowledge base files
    """
    raise NotImplementedError("Medical KB ingestion not yet implemented")


if __name__ == "__main__":
    print("Medical KB ingestion - not yet implemented")
