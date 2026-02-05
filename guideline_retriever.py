"""
Guideline Retriever - PHI-Aware Hybrid Vector Strategy

This module implements the dual-mode guideline retrieval system:
- LOCAL mode: Uses FAISS for all retrieval (default, zero external dependencies)
- ENTERPRISE mode: Uses Pinecone for guidelines, FAISS for patient notes

Architecture:
    Patient Notes (PHI) → Always FAISS (local, never leaves machine)
    Medical Guidelines (public) → Pinecone (enterprise) or FAISS (local)

This demonstrates "PHI-Aware Data Residency" - a compliance-informed design pattern
that keeps protected health information on-premises while allowing public knowledge
bases to leverage cloud infrastructure.

NOTE: This is a portfolio reference implementation with synthetic data.
It is NOT a production medical device or HIPAA-certified system.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv

from concept_query import ConceptQueryBuilder, ConceptQuery, validate_phi_safety
from vector_store_faiss import FAISSIndex, get_guidelines_index, load_guidelines_from_markdown

load_dotenv()


class RetrievalMode(str, Enum):
    """Operating mode for guideline retrieval."""
    LOCAL = "local"          # FAISS only (default)
    ENTERPRISE = "enterprise"  # Pinecone for guidelines


@dataclass
class RetrievalResult:
    """Result from a guideline retrieval operation."""
    guidelines: List[Dict]
    mode_used: RetrievalMode
    query_used: str
    phi_safe: bool
    source: str  # "faiss" or "pinecone"
    fallback_used: bool = False
    fallback_reason: Optional[str] = None


class BaseGuidelineRetriever(ABC):
    """Abstract base class for guideline retrievers."""

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for guidelines matching the query."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this retriever is available and configured."""
        pass


class FAISSGuidelineRetriever(BaseGuidelineRetriever):
    """Local FAISS-based guideline retriever."""

    def __init__(self, index_path: str = "indexes/guidelines"):
        self.index_path = index_path
        self._index: Optional[FAISSIndex] = None

    def _get_index(self) -> FAISSIndex:
        """Get or load the FAISS index."""
        if self._index is None:
            self._index = FAISSIndex(self.index_path)
            if not self._index.load():
                # Build index from markdown files if not exists
                print("Building guidelines index from markdown files...")
                documents = load_guidelines_from_markdown()
                if documents:
                    self._index.build_index(documents)
                    self._index.save()
                else:
                    raise ValueError("No guideline documents found to index")
        return self._index

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search guidelines using FAISS."""
        index = self._get_index()
        return index.query(query, top_k=top_k)

    def is_available(self) -> bool:
        """FAISS is always available (local)."""
        try:
            import faiss
            return True
        except ImportError:
            return False


class PineconeGuidelineRetriever(BaseGuidelineRetriever):
    """
    Pinecone-based guideline retriever for enterprise mode.

    This retriever ONLY handles guidelines (public data).
    Patient data NEVER goes through Pinecone.
    """

    def __init__(
        self,
        index_name: str = "careflow-guidelines",
        namespace: str = "medical-kb"
    ):
        self.index_name = index_name
        self.namespace = namespace
        self._client = None
        self._index = None
        self._openai_client = None

    def _init_clients(self):
        """Initialize Pinecone and OpenAI clients."""
        if self._client is not None:
            return

        try:
            from pinecone import Pinecone
            from openai import OpenAI

            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY not set")

            self._client = Pinecone(api_key=api_key)
            self._index = self._client.Index(self.index_name)

            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                raise ValueError("OPENAI_API_KEY not set")

            self._openai_client = OpenAI(api_key=openai_key)

        except ImportError:
            raise ImportError("pinecone-client not installed. Run: pip install pinecone-client")

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for query text."""
        self._init_clients()

        response = self._openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search guidelines using Pinecone.

        IMPORTANT: This should only be called with de-identified queries
        from the ConceptQueryBuilder. Never send patient-specific data.
        """
        self._init_clients()

        # Validate PHI safety before sending to external service
        is_safe, violations = validate_phi_safety(query)
        if not is_safe:
            raise ValueError(
                f"PHI detected in query - refusing to send to Pinecone. "
                f"Violations: {violations}"
            )

        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Query Pinecone
        results = self._index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True
        )

        # Format results to match FAISS format
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "id": match.id,
                "text": match.metadata.get("text", ""),
                "metadata": {
                    k: v for k, v in match.metadata.items()
                    if k != "text"
                },
                "score": float(match.score)
            })

        return formatted_results

    def is_available(self) -> bool:
        """Check if Pinecone is available and configured."""
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            return False

        try:
            from pinecone import Pinecone
            client = Pinecone(api_key=api_key)
            # Check if index exists
            indexes = client.list_indexes()
            return any(idx.name == self.index_name for idx in indexes)
        except Exception:
            return False


class GuidelineRetriever:
    """
    Main guideline retriever with PHI-aware hybrid strategy.

    Usage:
        # Local mode (default) - FAISS only
        retriever = GuidelineRetriever(mode=RetrievalMode.LOCAL)

        # Enterprise mode - Pinecone for guidelines
        retriever = GuidelineRetriever(mode=RetrievalMode.ENTERPRISE)

    The mode can also be set via command line:
        python care_app.py --mode=local
        python care_app.py --mode=enterprise
    """

    def __init__(
        self,
        mode: RetrievalMode = RetrievalMode.LOCAL,
        fallback_to_local: bool = True
    ):
        """
        Initialize the guideline retriever.

        Args:
            mode: Operating mode (LOCAL or ENTERPRISE)
            fallback_to_local: If True, fall back to FAISS when Pinecone fails
        """
        self.mode = mode
        self.fallback_to_local = fallback_to_local
        self.concept_builder = ConceptQueryBuilder()

        # Initialize retrievers
        self._faiss_retriever = FAISSGuidelineRetriever()
        self._pinecone_retriever: Optional[PineconeGuidelineRetriever] = None

        if mode == RetrievalMode.ENTERPRISE:
            self._pinecone_retriever = PineconeGuidelineRetriever()

    def search_with_facts(
        self,
        facts,
        top_k: int = 3
    ) -> RetrievalResult:
        """
        Search guidelines using extracted patient facts.

        This is the PHI-safe entry point. It:
        1. Converts patient facts to de-identified concepts
        2. Queries the appropriate backend
        3. Returns guidelines without ever sending PHI externally

        Args:
            facts: ExtractedFacts from extraction.py
            top_k: Number of results to return

        Returns:
            RetrievalResult with guidelines and metadata
        """
        # Build de-identified concept query
        concept_query = self.concept_builder.build_from_extracted_facts(facts)

        return self._search_with_concept_query(concept_query, top_k)

    def search_with_gaps(
        self,
        gaps: list,
        top_k: int = 3
    ) -> RetrievalResult:
        """
        Search guidelines based on detected care gaps.

        Args:
            gaps: List of GapResult from reasoning_engine.py
            top_k: Number of results to return

        Returns:
            RetrievalResult with guidelines targeting the gaps
        """
        # Build concept query from gaps
        concept_query = self.concept_builder.build_from_gap_results(gaps)

        return self._search_with_concept_query(concept_query, top_k)

    def search_raw(
        self,
        query: str,
        top_k: int = 3,
        skip_phi_check: bool = False
    ) -> RetrievalResult:
        """
        Search guidelines with a raw query string.

        WARNING: Use with caution. This bypasses the ConceptQueryBuilder.
        Only use for pre-validated queries or testing.

        Args:
            query: Raw query string
            top_k: Number of results to return
            skip_phi_check: If True, skip PHI validation (use only in tests)

        Returns:
            RetrievalResult with guidelines
        """
        if not skip_phi_check and self.mode == RetrievalMode.ENTERPRISE:
            is_safe, violations = validate_phi_safety(query)
            if not is_safe:
                raise ValueError(
                    f"PHI detected in raw query. Use search_with_facts() instead. "
                    f"Violations: {violations}"
                )

        return self._execute_search(query, top_k)

    def _search_with_concept_query(
        self,
        concept_query: ConceptQuery,
        top_k: int
    ) -> RetrievalResult:
        """Execute search with a validated concept query."""
        return self._execute_search(concept_query.query_text, top_k)

    def _execute_search(self, query: str, top_k: int) -> RetrievalResult:
        """Execute the actual search based on mode."""
        fallback_used = False
        fallback_reason = None

        if self.mode == RetrievalMode.ENTERPRISE:
            # Try Pinecone first
            if self._pinecone_retriever and self._pinecone_retriever.is_available():
                try:
                    guidelines = self._pinecone_retriever.search(query, top_k)
                    return RetrievalResult(
                        guidelines=guidelines,
                        mode_used=RetrievalMode.ENTERPRISE,
                        query_used=query,
                        phi_safe=True,
                        source="pinecone",
                        fallback_used=False
                    )
                except Exception as e:
                    if self.fallback_to_local:
                        fallback_used = True
                        fallback_reason = f"Pinecone error: {str(e)}"
                    else:
                        raise
            else:
                if self.fallback_to_local:
                    fallback_used = True
                    fallback_reason = "Pinecone not available or not configured"
                else:
                    raise ValueError("Pinecone not available and fallback disabled")

        # Use FAISS (either as primary in LOCAL mode or as fallback)
        guidelines = self._faiss_retriever.search(query, top_k)

        return RetrievalResult(
            guidelines=guidelines,
            mode_used=self.mode,
            query_used=query,
            phi_safe=True,
            source="faiss",
            fallback_used=fallback_used,
            fallback_reason=fallback_reason
        )

    def get_status(self) -> Dict:
        """Get retriever status for diagnostics."""
        pinecone_available = (
            self._pinecone_retriever.is_available()
            if self._pinecone_retriever else False
        )

        return {
            "mode": self.mode.value,
            "faiss_available": self._faiss_retriever.is_available(),
            "pinecone_available": pinecone_available,
            "pinecone_configured": os.getenv("PINECONE_API_KEY") is not None,
            "fallback_enabled": self.fallback_to_local
        }


def get_retrieval_mode_from_args() -> RetrievalMode:
    """
    Parse retrieval mode from command line arguments.

    Usage:
        python care_app.py --mode=local
        python care_app.py --mode=enterprise
    """
    import sys

    for arg in sys.argv[1:]:
        if arg.startswith("--mode="):
            mode_str = arg.split("=")[1].lower()
            if mode_str == "enterprise":
                return RetrievalMode.ENTERPRISE
            elif mode_str == "local":
                return RetrievalMode.LOCAL
            else:
                print(f"Warning: Unknown mode '{mode_str}', using LOCAL")

    # Default to LOCAL mode
    return RetrievalMode.LOCAL


# Module-level retriever instance
_retriever: Optional[GuidelineRetriever] = None


def get_guideline_retriever(
    mode: Optional[RetrievalMode] = None
) -> GuidelineRetriever:
    """
    Get or create the global guideline retriever.

    Args:
        mode: If provided, creates new retriever with this mode.
              If None, returns existing or creates with default (LOCAL).
    """
    global _retriever

    if mode is not None or _retriever is None:
        actual_mode = mode if mode is not None else get_retrieval_mode_from_args()
        _retriever = GuidelineRetriever(mode=actual_mode)

    return _retriever
