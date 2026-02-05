"""Tests for the Guideline Retriever - PHI-Aware Hybrid Vector Strategy.

Tests the retrieval system with mode switching and PHI protection.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guideline_retriever import (
    GuidelineRetriever,
    RetrievalMode,
    RetrievalResult,
    FAISSGuidelineRetriever,
    get_retrieval_mode_from_args,
)
from concept_query import ConceptQueryBuilder


class TestRetrieval:
    """Test suite for GuidelineRetriever."""

    def __init__(self):
        self.results = []
        # Use LOCAL mode for tests (doesn't require Pinecone)
        self.retriever = GuidelineRetriever(mode=RetrievalMode.LOCAL)

    def _assert(self, condition: bool, test_name: str, details: str = ""):
        """Record test result."""
        self.results.append({
            "test": test_name,
            "passed": condition,
            "details": details
        })
        return condition

    def test_local_mode_initialization(self):
        """Test: Retriever initializes in LOCAL mode by default."""
        retriever = GuidelineRetriever()

        return self._assert(
            retriever.mode == RetrievalMode.LOCAL,
            "Local mode initialization",
            f"Mode: {retriever.mode.value}"
        )

    def test_enterprise_mode_initialization(self):
        """Test: Retriever can initialize in ENTERPRISE mode."""
        retriever = GuidelineRetriever(mode=RetrievalMode.ENTERPRISE)

        return self._assert(
            retriever.mode == RetrievalMode.ENTERPRISE,
            "Enterprise mode initialization",
            f"Mode: {retriever.mode.value}"
        )

    def test_faiss_retriever_available(self):
        """Test: FAISS retriever is available (local dependency)."""
        faiss_retriever = FAISSGuidelineRetriever()

        # Check if FAISS can be imported
        is_available = faiss_retriever.is_available()

        return self._assert(
            is_available,
            "FAISS retriever available",
            f"Available: {is_available}"
        )

    def test_search_raw_returns_result(self):
        """Test: search_raw returns RetrievalResult with guidelines."""
        result = self.retriever.search_raw(
            "diabetes a1c glycemic control guidelines",
            top_k=3,
            skip_phi_check=True
        )

        has_guidelines = len(result.guidelines) > 0
        correct_type = isinstance(result, RetrievalResult)
        is_faiss = result.source == "faiss"

        return self._assert(
            has_guidelines and correct_type and is_faiss,
            "Search raw returns results",
            f"Guidelines: {len(result.guidelines)}, Source: {result.source}"
        )

    def test_retrieval_result_structure(self):
        """Test: RetrievalResult has all required fields."""
        result = self.retriever.search_raw(
            "diabetes guidelines",
            top_k=1,
            skip_phi_check=True
        )

        has_guidelines = hasattr(result, "guidelines")
        has_mode = hasattr(result, "mode_used")
        has_query = hasattr(result, "query_used")
        has_phi_safe = hasattr(result, "phi_safe")
        has_source = hasattr(result, "source")

        all_fields = has_guidelines and has_mode and has_query and has_phi_safe and has_source

        return self._assert(
            all_fields,
            "RetrievalResult structure",
            f"Has all required fields: {all_fields}"
        )

    def test_result_marked_phi_safe(self):
        """Test: Results are marked as PHI-safe."""
        result = self.retriever.search_raw(
            "hypertension blood pressure guidelines",
            skip_phi_check=True
        )

        return self._assert(
            result.phi_safe == True,
            "Result marked PHI-safe",
            f"PHI safe: {result.phi_safe}"
        )

    def test_guideline_has_text_and_metadata(self):
        """Test: Retrieved guidelines have text and metadata."""
        result = self.retriever.search_raw(
            "diabetes a1c guidelines",
            top_k=1,
            skip_phi_check=True
        )

        if not result.guidelines:
            return self._assert(False, "Guideline structure", "No guidelines returned")

        guideline = result.guidelines[0]
        has_text = "text" in guideline and len(guideline["text"]) > 0
        has_metadata = "metadata" in guideline
        has_score = "score" in guideline

        return self._assert(
            has_text and has_metadata and has_score,
            "Guideline has text and metadata",
            f"Text length: {len(guideline.get('text', ''))}, Has metadata: {has_metadata}"
        )

    def test_top_k_limits_results(self):
        """Test: top_k parameter limits number of results."""
        result_1 = self.retriever.search_raw("diabetes", top_k=1, skip_phi_check=True)
        result_3 = self.retriever.search_raw("diabetes", top_k=3, skip_phi_check=True)

        return self._assert(
            len(result_1.guidelines) <= 1 and len(result_3.guidelines) <= 3,
            "Top K limits results",
            f"top_k=1: {len(result_1.guidelines)}, top_k=3: {len(result_3.guidelines)}"
        )

    def test_phi_check_blocks_unsafe_query_enterprise(self):
        """Test: PHI check blocks unsafe queries in enterprise mode."""
        # Create retriever but with fallback disabled so we get the error
        retriever = GuidelineRetriever(
            mode=RetrievalMode.ENTERPRISE,
            fallback_to_local=False
        )

        try:
            # This should raise an error due to PHI in query
            retriever.search_raw("patient PT001 A1C 8.2 diabetes", skip_phi_check=False)
            blocked = False
        except ValueError as e:
            blocked = "PHI detected" in str(e)

        return self._assert(
            blocked,
            "PHI check blocks unsafe query",
            f"Query with PHI was blocked: {blocked}"
        )

    def test_local_mode_skips_phi_check_for_raw(self):
        """Test: LOCAL mode with skip_phi_check=True allows any query."""
        # In local mode, search_raw with skip_phi_check=True should work
        result = self.retriever.search_raw(
            "diabetes guidelines management",
            skip_phi_check=True
        )

        return self._assert(
            result is not None and result.source == "faiss",
            "Local mode raw search works",
            f"Source: {result.source}"
        )

    def test_get_status_returns_info(self):
        """Test: get_status returns diagnostic information."""
        status = self.retriever.get_status()

        has_mode = "mode" in status
        has_faiss = "faiss_available" in status
        has_fallback = "fallback_enabled" in status

        return self._assert(
            has_mode and has_faiss and has_fallback,
            "Get status returns info",
            f"Status keys: {list(status.keys())}"
        )

    def test_fallback_enabled_by_default(self):
        """Test: Fallback to local is enabled by default."""
        retriever = GuidelineRetriever(mode=RetrievalMode.ENTERPRISE)
        status = retriever.get_status()

        return self._assert(
            status["fallback_enabled"] == True,
            "Fallback enabled by default",
            f"Fallback: {status['fallback_enabled']}"
        )

    def test_concept_builder_integration(self):
        """Test: Retriever has ConceptQueryBuilder for PHI protection."""
        has_builder = hasattr(self.retriever, 'concept_builder')
        builder_correct_type = isinstance(
            self.retriever.concept_builder,
            ConceptQueryBuilder
        )

        return self._assert(
            has_builder and builder_correct_type,
            "Concept builder integration",
            f"Has builder: {has_builder}, Correct type: {builder_correct_type}"
        )

    def test_search_returns_relevant_results(self):
        """Test: Search returns semantically relevant results."""
        result = self.retriever.search_raw(
            "a1c glycemic control diabetes target",
            top_k=3,
            skip_phi_check=True
        )

        if not result.guidelines:
            return self._assert(False, "Semantic relevance", "No results")

        # Check if A1C-related guideline is in results
        texts = [g["text"].lower() for g in result.guidelines]
        has_a1c_content = any("a1c" in t or "glycemic" in t for t in texts)

        return self._assert(
            has_a1c_content,
            "Search returns relevant results",
            f"Found A1C-related content: {has_a1c_content}"
        )

    def test_retrieval_mode_enum_values(self):
        """Test: RetrievalMode enum has correct values."""
        local_value = RetrievalMode.LOCAL.value == "local"
        enterprise_value = RetrievalMode.ENTERPRISE.value == "enterprise"

        return self._assert(
            local_value and enterprise_value,
            "RetrievalMode enum values",
            f"LOCAL={RetrievalMode.LOCAL.value}, ENTERPRISE={RetrievalMode.ENTERPRISE.value}"
        )

    def run_all(self) -> dict:
        """Run all retrieval tests."""
        self.results = []

        # Run all test methods
        self.test_local_mode_initialization()
        self.test_enterprise_mode_initialization()
        self.test_faiss_retriever_available()
        self.test_search_raw_returns_result()
        self.test_retrieval_result_structure()
        self.test_result_marked_phi_safe()
        self.test_guideline_has_text_and_metadata()
        self.test_top_k_limits_results()
        self.test_phi_check_blocks_unsafe_query_enterprise()
        self.test_local_mode_skips_phi_check_for_raw()
        self.test_get_status_returns_info()
        self.test_fallback_enabled_by_default()
        self.test_concept_builder_integration()
        self.test_search_returns_relevant_results()
        self.test_retrieval_mode_enum_values()

        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)

        return {
            "suite": "Retrieval",
            "passed": passed,
            "total": total,
            "results": self.results
        }


def run_tests():
    """Run retrieval tests and print results."""
    suite = TestRetrieval()
    results = suite.run_all()

    print("=" * 60)
    print("RETRIEVAL TESTS")
    print("=" * 60)

    for r in results["results"]:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['test']}")
        if r["details"]:
            print(f"         {r['details']}")

    print("-" * 60)
    print(f"Results: {results['passed']}/{results['total']} passed")

    return results


if __name__ == "__main__":
    run_tests()
