"""Tests for the Concept Query Builder - PHI-Aware De-identification Layer.

Tests that patient-specific data is properly de-identified before external queries.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concept_query import (
    ConceptQueryBuilder,
    ConceptQuery,
    validate_phi_safety,
    DIAGNOSIS_CONCEPTS,
    MEDICATION_CLASS_CONCEPTS,
)


class TestConceptQuery:
    """Test suite for ConceptQueryBuilder."""

    def __init__(self):
        self.builder = ConceptQueryBuilder()
        self.results = []

    def _assert(self, condition: bool, test_name: str, details: str = ""):
        """Record test result."""
        self.results.append({
            "test": test_name,
            "passed": condition,
            "details": details
        })
        return condition

    def test_diabetes_concepts_extracted(self):
        """Test: Diabetes diagnosis maps to correct concepts."""
        query = self.builder.build_query(diagnoses=["Type 2 Diabetes"])

        has_diabetes = "diabetes" in query.concepts
        has_a1c = "a1c" in query.concepts
        has_glycemic = "glycemic" in query.concepts

        return self._assert(
            has_diabetes and has_a1c and has_glycemic,
            "Diabetes concepts extracted",
            f"Concepts: {query.concepts[:5]}..."
        )

    def test_hypertension_concepts_extracted(self):
        """Test: Hypertension diagnosis maps to correct concepts."""
        query = self.builder.build_query(diagnoses=["Essential Hypertension"])

        has_htn = "hypertension" in query.concepts
        has_bp = "blood pressure" in query.concepts
        has_cv = "cardiovascular" in query.concepts

        return self._assert(
            has_htn and has_bp and has_cv,
            "Hypertension concepts extracted",
            f"Concepts: {query.concepts[:5]}..."
        )

    def test_no_numeric_values_in_query(self):
        """Test: Query text does not contain specific numeric values."""
        # Build query with flags indicating metrics exist (but NOT their values)
        query = self.builder.build_query(
            diagnoses=["Type 2 Diabetes", "Hypertension"],
            has_a1c=True,
            has_blood_pressure=True
        )

        # Check that query doesn't contain numbers
        import re
        has_decimals = bool(re.search(r'\d+\.\d+', query.query_text))
        has_bp_pattern = bool(re.search(r'\d{2,3}/\d{2,3}', query.query_text))

        return self._assert(
            not has_decimals and not has_bp_pattern,
            "No numeric values in query",
            f"Query: {query.query_text[:100]}..."
        )

    def test_missing_med_class_concepts(self):
        """Test: Missing medication class adds appropriate concepts."""
        query = self.builder.build_query(
            diagnoses=["Type 2 Diabetes"],
            missing_med_classes=["ace_arb", "statin"]
        )

        has_ace = "ace inhibitor" in query.concepts
        has_statin = "statin" in query.concepts

        return self._assert(
            has_ace and has_statin,
            "Missing medication class concepts",
            f"Has ACE: {has_ace}, Has Statin: {has_statin}"
        )

    def test_phi_safety_passes_clean_query(self):
        """Test: PHI-safe query passes validation."""
        query = self.builder.build_query(diagnoses=["Type 2 Diabetes"])

        is_safe, violations = validate_phi_safety(query.query_text)

        return self._assert(
            is_safe and len(violations) == 0,
            "PHI safety passes clean query",
            f"Safe: {is_safe}, Violations: {violations}"
        )

    def test_phi_safety_catches_numeric_values(self):
        """Test: PHI validator catches numeric values."""
        unsafe_query = "diabetes patient with A1C 8.2 needs treatment"

        is_safe, violations = validate_phi_safety(unsafe_query)

        return self._assert(
            not is_safe and len(violations) > 0,
            "PHI safety catches numeric values",
            f"Violations: {violations}"
        )

    def test_phi_safety_catches_bp_pattern(self):
        """Test: PHI validator catches blood pressure patterns."""
        unsafe_query = "hypertension patient with BP 142/94"

        is_safe, violations = validate_phi_safety(unsafe_query)

        return self._assert(
            not is_safe and len(violations) > 0,
            "PHI safety catches BP pattern",
            f"Violations: {violations}"
        )

    def test_phi_safety_catches_date_pattern(self):
        """Test: PHI validator catches date patterns."""
        unsafe_query = "diabetes treatment started 01/15/2024"

        is_safe, violations = validate_phi_safety(unsafe_query)

        return self._assert(
            not is_safe and len(violations) > 0,
            "PHI safety catches date pattern",
            f"Violations: {violations}"
        )

    def test_phi_safety_catches_patient_id(self):
        """Test: PHI validator catches patient ID patterns."""
        unsafe_query = "guidelines for PT001 diabetes management"

        is_safe, violations = validate_phi_safety(unsafe_query)

        return self._assert(
            not is_safe and len(violations) > 0,
            "PHI safety catches patient ID",
            f"Violations: {violations}"
        )

    def test_query_has_clinical_suffix(self):
        """Test: Query text includes clinical/recommendations suffix."""
        query = self.builder.build_query(diagnoses=["Type 2 Diabetes"])

        has_guidelines = "guidelines" in query.query_text.lower()
        has_recommendations = "recommendations" in query.query_text.lower()

        return self._assert(
            has_guidelines or has_recommendations,
            "Query has clinical suffix",
            f"Query ends with: ...{query.query_text[-50:]}"
        )

    def test_source_conditions_tracked(self):
        """Test: Source conditions are tracked for debugging."""
        query = self.builder.build_query(
            diagnoses=["Type 2 Diabetes"],
            has_a1c=True
        )

        has_diagnosis_source = any("diagnosis:" in s for s in query.source_conditions)
        has_metric_source = any("metric:" in s for s in query.source_conditions)

        return self._assert(
            has_diagnosis_source and has_metric_source,
            "Source conditions tracked",
            f"Sources: {query.source_conditions}"
        )

    def test_concept_query_is_phi_safe_flag(self):
        """Test: ConceptQuery object has is_phi_safe=True."""
        query = self.builder.build_query(diagnoses=["Hypertension"])

        return self._assert(
            query.is_phi_safe == True,
            "ConceptQuery has is_phi_safe flag",
            f"is_phi_safe: {query.is_phi_safe}"
        )

    def test_multiple_diagnoses_combined(self):
        """Test: Multiple diagnoses combine concepts correctly."""
        query = self.builder.build_query(
            diagnoses=["Type 2 Diabetes", "Hypertension", "Hyperlipidemia"]
        )

        has_diabetes = "diabetes" in query.concepts
        has_htn = "hypertension" in query.concepts
        has_lipids = "lipids" in query.concepts or "cholesterol" in query.concepts

        return self._assert(
            has_diabetes and has_htn and has_lipids,
            "Multiple diagnoses combined",
            f"Total concepts: {len(query.concepts)}"
        )

    def test_gap_type_concepts(self):
        """Test: Gap types add appropriate concepts."""
        query = self.builder.build_query(
            gap_types=["A1C_THRESHOLD", "HTN_ACE_ARB"]
        )

        has_a1c = "a1c" in query.concepts
        has_ace = "ace inhibitor" in query.concepts

        return self._assert(
            has_a1c and has_ace,
            "Gap type concepts extracted",
            f"Concepts: {query.concepts[:6]}..."
        )

    def test_safe_term_extraction(self):
        """Test: _extract_safe_terms filters out numbers and identifiers."""
        # Use the private method through the builder
        safe_terms = self.builder._extract_safe_terms(
            "Patient PT001 has diabetes type 2 since 2020"
        )

        # Should have clinical terms, not IDs or years
        has_diabetes = "diabetes" in safe_terms
        has_type = "type" in safe_terms
        has_patient_id = any("pt001" in t.lower() for t in safe_terms)
        has_year = any("2020" in t for t in safe_terms)

        return self._assert(
            has_diabetes and has_type and not has_patient_id and not has_year,
            "Safe term extraction filters PHI",
            f"Safe terms: {safe_terms}"
        )

    def run_all(self) -> dict:
        """Run all concept query tests."""
        self.results = []

        # Run all test methods
        self.test_diabetes_concepts_extracted()
        self.test_hypertension_concepts_extracted()
        self.test_no_numeric_values_in_query()
        self.test_missing_med_class_concepts()
        self.test_phi_safety_passes_clean_query()
        self.test_phi_safety_catches_numeric_values()
        self.test_phi_safety_catches_bp_pattern()
        self.test_phi_safety_catches_date_pattern()
        self.test_phi_safety_catches_patient_id()
        self.test_query_has_clinical_suffix()
        self.test_source_conditions_tracked()
        self.test_concept_query_is_phi_safe_flag()
        self.test_multiple_diagnoses_combined()
        self.test_gap_type_concepts()
        self.test_safe_term_extraction()

        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)

        return {
            "suite": "Concept Query",
            "passed": passed,
            "total": total,
            "results": self.results
        }


def run_tests():
    """Run concept query tests and print results."""
    suite = TestConceptQuery()
    results = suite.run_all()

    print("=" * 60)
    print("CONCEPT QUERY TESTS")
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
