"""FAISS vector store for CareFlow semantic search."""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class FAISSIndex:
    """FAISS-based vector index for semantic search."""

    def __init__(
        self,
        index_path: str,
        embedding_model: str = "text-embedding-3-small"
    ):
        """Initialize the FAISS index.

        Args:
            index_path: Directory path to store/load index files
            embedding_model: OpenAI embedding model to use
        """
        self.index_path = Path(index_path)
        self.embedding_model = embedding_model
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: List[Dict] = []
        self.dimension: int = 1536  # text-embedding-3-small dimension

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Try Streamlit secrets
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
                    api_key = st.secrets["OPENAI_API_KEY"]
            except:
                pass

        self.client = OpenAI(api_key=api_key) if api_key else None

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string.

        Args:
            text: Text to embed

        Returns:
            Numpy array of embedding
        """
        if not self.client:
            raise ValueError("OpenAI API key not configured")

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            Numpy array of embeddings (n_texts x dimension)
        """
        if not self.client:
            raise ValueError("OpenAI API key not configured")

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )

        embeddings = [data.embedding for data in response.data]
        return np.array(embeddings, dtype=np.float32)

    def build_index(self, documents: List[Dict]) -> None:
        """Build FAISS index from documents.

        Args:
            documents: List of dicts with "id", "text", and optional "metadata"
        """
        if not faiss:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

        if not documents:
            raise ValueError("No documents provided")

        self.documents = documents

        # Get embeddings for all documents
        texts = [doc["text"] for doc in documents]
        embeddings = self._get_embeddings_batch(texts)

        # Normalize embeddings for cosine similarity (using inner product)
        faiss.normalize_L2(embeddings)

        # Create index
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product = cosine sim after normalization
        self.index.add(embeddings)

        print(f"Built index with {len(documents)} documents")

    def query(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """Query the index for similar documents.

        Args:
            query_text: Query string
            top_k: Number of results to return

        Returns:
            List of dicts with "id", "text", "metadata", "score"
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Get query embedding
        query_embedding = self._get_embedding(query_text)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        # Build results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    "id": doc.get("id", str(idx)),
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "score": float(score)
                })

        return results

    def save(self) -> None:
        """Save index and documents to disk."""
        if self.index is None:
            raise ValueError("No index to save")

        self.index_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_file = self.index_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))

        # Save documents
        docs_file = self.index_path / "documents.json"
        with open(docs_file, "w") as f:
            json.dump(self.documents, f, indent=2)

        # Save metadata
        meta_file = self.index_path / "metadata.json"
        with open(meta_file, "w") as f:
            json.dump({
                "embedding_model": self.embedding_model,
                "dimension": self.dimension,
                "num_documents": len(self.documents)
            }, f, indent=2)

        print(f"Saved index to {self.index_path}")

    def load(self) -> bool:
        """Load index and documents from disk.

        Returns:
            True if loaded successfully, False if files don't exist
        """
        index_file = self.index_path / "index.faiss"
        docs_file = self.index_path / "documents.json"
        meta_file = self.index_path / "metadata.json"

        if not all(f.exists() for f in [index_file, docs_file, meta_file]):
            return False

        # Load FAISS index
        self.index = faiss.read_index(str(index_file))

        # Load documents
        with open(docs_file, "r") as f:
            self.documents = json.load(f)

        # Load metadata
        with open(meta_file, "r") as f:
            meta = json.load(f)
            self.embedding_model = meta.get("embedding_model", self.embedding_model)
            self.dimension = meta.get("dimension", self.dimension)

        print(f"Loaded index with {len(self.documents)} documents")
        return True


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
        print(f"Warning: {kb_dir} does not exist")
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


# Convenience functions for global indexes

_patient_index: Optional[FAISSIndex] = None
_guidelines_index: Optional[FAISSIndex] = None


def get_patient_index() -> FAISSIndex:
    """Get or create the patient notes index."""
    global _patient_index
    if _patient_index is None:
        _patient_index = FAISSIndex("indexes/patients")
    return _patient_index


def get_guidelines_index() -> FAISSIndex:
    """Get or create the medical guidelines index."""
    global _guidelines_index
    if _guidelines_index is None:
        _guidelines_index = FAISSIndex("indexes/guidelines")
    return _guidelines_index
