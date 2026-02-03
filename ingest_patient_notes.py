"""Ingest patient notes into the vector store."""

# TODO: Implement patient notes ingestion
# - Read patient notes from data/patients/
# - Chunk and embed notes
# - Store in FAISS index


def ingest_patient_notes(notes_dir: str = "data/patients"):
    """Ingest patient notes into vector store.

    Args:
        notes_dir: Directory containing patient notes
    """
    raise NotImplementedError("Patient notes ingestion not yet implemented")


if __name__ == "__main__":
    print("Patient notes ingestion - not yet implemented")
