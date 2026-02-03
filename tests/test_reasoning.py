"""Tests for the reasoning engine.

Tests deterministic gap detection rules.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction import PatientFactExtractor, ExtractedFacts
from reasoning_engine import ReasoningEngine, CareGapRules, GapResult
from care_database import get_database


class TestReasoning:
    """Test suite for ReasoningEngine."""

    def __init__(self):
        self.engine = ReasoningEngine()
        self.extractor = PatientFactExtractor()
        self.db = get_database()
        self.results = []

    def _assert(self, condition: bool, test_name: str, details: str = ""):
        """Record test result."""
        status = "PASS" if condition else "FAIL"
        self.results.append({
            "test": test_name,
            "passed": condition,
            "details": details
        })
        return condition

    def test_a1c_gap_above_threshold(self):
        """Test: A1C 8.2 > 7.0 -> gap_detected = True."""
        facts = ExtractedFacts(
            a1c=8.2,
            diagnoses=["Type 2 Diabetes Mellitus"],
            medications=["Metformin 1000mg"]
        )

        gap = CareGapRules.check_a1c_threshold(facts, "TEST001")

        return self._assert(
            gap.gap_detected is True,
            "A1C 8.2 > 7.0 gap detected",
            f"gap_detected={gap.gap_detected}, comparison='{gap.comparison}'"
        )

    def test_a1c_gap_below_threshold(self):
        """Test: A1C 6.8 < 7.0 -> gap_detected = False."""
        facts = ExtractedFacts(
            a1c=6.8,
            diagnoses=["Type 2 Diabetes Mellitus"],
            medications=["Metformin 1000mg"]
        )

        gap = CareGapRules.check_a1c_threshold(facts, "TEST002")

        return self._assert(
            gap.gap_detected is False,
            "A1C 6.8 < 7.0 no gap",
            f"gap_detected={gap.gap_detected}, therefore='{gap.therefore}'"
        )

    def test_a1c_gap_at_threshold(self):
        """Test: A1C 7.0 = 7.0 -> gap_detected = True (at or above)."""
        facts = ExtractedFacts(
            a1c=7.0,
            diagnoses=["Type 2 Diabetes Mellitus"],
            medications=["Metformin 1000mg"]
        )

        gap = CareGapRules.check_a1c_threshold(facts, "TEST003")

        return self._assert(
            gap.gap_detected is True,
            "A1C 7.0 at threshold is a gap",
            f"gap_detected={gap.gap_detected}"
        )

    def test_htn_gap_no_ace_arb(self):
        """Test: HTN + Diabetes + no ACE/ARB -> gap_detected = True."""
        facts = ExtractedFacts(
            a1c=7.5,
            diagnoses=["Type 2 Diabetes Mellitus", "Hypertension"],
            medications=["Metformin 1000mg", "Amlodipine 5mg"]  # No ACE/ARB
        )

        gap = CareGapRules.check_htn_ace_arb(facts, "TEST004")

        return self._assert(
            gap.gap_detected is True,
            "HTN + DM + no ACE/ARB gap detected",
            f"gap_detected={gap.gap_detected}, recommendation='{gap.recommendation[:50]}...'"
        )

    def test_htn_gap_with_lisinopril(self):
        """Test: HTN + Diabetes + Lisinopril -> gap_detected = False."""
        facts = ExtractedFacts(
            a1c=7.5,
            diagnoses=["Type 2 Diabetes Mellitus", "Hypertension"],
            medications=["Metformin 1000mg", "Lisinopril 10mg"]  # Has ACE inhibitor
        )

        gap = CareGapRules.check_htn_ace_arb(facts, "TEST005")

        return self._assert(
            gap.gap_detected is False,
            "HTN + DM + Lisinopril no gap",
            f"gap_detected={gap.gap_detected}, therefore='{gap.therefore}'"
        )

    def test_htn_gap_with_losartan(self):
        """Test: HTN + Diabetes + Losartan (ARB) -> gap_detected = False."""
        facts = ExtractedFacts(
            a1c=7.5,
            diagnoses=["Type 2 Diabetes Mellitus", "Hypertension"],
            medications=["Metformin 1000mg", "Losartan 50mg"]  # Has ARB
        )

        gap = CareGapRules.check_htn_ace_arb(facts, "TEST006")

        return self._assert(
            gap.gap_detected is False,
            "HTN + DM + Losartan (ARB) no gap",
            f"gap_detected={gap.gap_detected}"
        )

    def test_bp_control_gap_elevated(self):
        """Test: BP 142/94 > 140/90 -> gap_detected = True."""
        facts = ExtractedFacts(
            a1c=7.5,
            blood_pressure={"systolic": 142, "diastolic": 94},
            diagnoses=["Type 2 Diabetes Mellitus", "Hypertension"],
            medications=["Metformin 1000mg"]
        )

        gap = CareGapRules.check_bp_control(facts, "TEST007")

        return self._assert(
            gap.gap_detected is True,
            "BP 142/94 elevated gap detected",
            f"gap_detected={gap.gap_detected}, comparison='{gap.comparison}'"
        )

    def test_bp_control_gap_normal(self):
        """Test: BP 128/82 < 140/90 -> gap_detected = False."""
        facts = ExtractedFacts(
            a1c=7.5,
            blood_pressure={"systolic": 128, "diastolic": 82},
            diagnoses=["Type 2 Diabetes Mellitus", "Hypertension"],
            medications=["Metformin 1000mg", "Lisinopril 10mg"]
        )

        gap = CareGapRules.check_bp_control(facts, "TEST008")

        return self._assert(
            gap.gap_detected is False,
            "BP 128/82 normal no gap",
            f"gap_detected={gap.gap_detected}, therefore='{gap.therefore}'"
        )

    def test_maria_garcia_has_3_gaps(self):
        """Test: Maria Garcia (PT001) has 3 gaps."""
        note = self.db.get_latest_note("PT001")
        facts = self.extractor.extract(note["note_text"])
        result = self.engine.evaluate_patient(facts, "PT001")

        detected = [g for g in result.gaps if g.gap_detected]

        return self._assert(
            result.gaps_found == 3,
            "Maria Garcia has 3 gaps",
            f"Found {result.gaps_found} gaps: {[g.gap_type for g in detected]}"
        )

    def test_james_wilson_minimal_gaps(self):
        """Test: James Wilson (PT002) is well-controlled with minimal gaps."""
        note = self.db.get_latest_note("PT002")
        facts = self.extractor.extract(note["note_text"])
        result = self.engine.evaluate_patient(facts, "PT002")

        # James has A1C 7.4 (borderline), so should have 1 gap (A1C threshold)
        # But good BP and on Lisinopril, so no other gaps
        return self._assert(
            result.gaps_found <= 1,
            "James Wilson minimal gaps (well-controlled)",
            f"Found {result.gaps_found} gaps, status: {result.overall_status}"
        )

    def test_robert_johnson_has_2_gaps(self):
        """Test: Robert Johnson (PT004) has 2 gaps (ACE/ARB + BP)."""
        note = self.db.get_latest_note("PT004")
        facts = self.extractor.extract(note["note_text"])
        result = self.engine.evaluate_patient(facts, "PT004")

        # Robert has good A1C but HTN without ACE/ARB and elevated BP
        return self._assert(
            result.gaps_found == 2,
            "Robert Johnson has 2 gaps",
            f"Found {result.gaps_found} gaps: {[g.gap_type for g in result.gaps if g.gap_detected]}"
        )

    def test_gap_result_has_citations(self):
        """Test: Gap results include proper citations."""
        facts = ExtractedFacts(
            a1c=8.2,
            diagnoses=["Type 2 Diabetes Mellitus"],
            medications=["Metformin 1000mg"]
        )

        gap = CareGapRules.check_a1c_threshold(facts, "PT001")

        has_patient_citation = "PATIENT:" in gap.patient_fact.get("source", "")
        has_guideline_citation = gap.guideline_id != ""

        return self._assert(
            has_patient_citation and has_guideline_citation,
            "Gap results have citations",
            f"Patient source: {gap.patient_fact.get('source')}, Guideline: {gap.guideline_id}"
        )

    def test_gap_severity_high_for_a1c_above_9(self):
        """Test: A1C > 9.0 results in high severity."""
        facts = ExtractedFacts(
            a1c=9.5,
            diagnoses=["Type 2 Diabetes Mellitus"],
            medications=["Metformin 1000mg"]
        )

        gap = CareGapRules.check_a1c_threshold(facts, "TEST009")

        return self._assert(
            gap.severity == "high",
            "A1C > 9.0 is high severity",
            f"A1C={facts.a1c}, severity={gap.severity}"
        )

    def test_therefore_statement_format(self):
        """Test: Gap results have 'Therefore' statement."""
        facts = ExtractedFacts(
            a1c=8.2,
            diagnoses=["Type 2 Diabetes Mellitus"],
            medications=["Metformin 1000mg"]
        )

        gap = CareGapRules.check_a1c_threshold(facts, "TEST010")

        has_therefore = gap.therefore.startswith("Therefore")

        return self._assert(
            has_therefore,
            "Gap has 'Therefore' statement",
            f"therefore='{gap.therefore}'"
        )

    def run_all(self) -> dict:
        """Run all reasoning tests."""
        self.results = []

        # Run all test methods
        self.test_a1c_gap_above_threshold()
        self.test_a1c_gap_below_threshold()
        self.test_a1c_gap_at_threshold()
        self.test_htn_gap_no_ace_arb()
        self.test_htn_gap_with_lisinopril()
        self.test_htn_gap_with_losartan()
        self.test_bp_control_gap_elevated()
        self.test_bp_control_gap_normal()
        self.test_maria_garcia_has_3_gaps()
        self.test_james_wilson_minimal_gaps()
        self.test_robert_johnson_has_2_gaps()
        self.test_gap_result_has_citations()
        self.test_gap_severity_high_for_a1c_above_9()
        self.test_therefore_statement_format()

        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)

        return {
            "suite": "Reasoning",
            "passed": passed,
            "total": total,
            "results": self.results
        }


def run_tests():
    """Run reasoning tests and print results."""
    suite = TestReasoning()
    results = suite.run_all()

    print("=" * 60)
    print("REASONING TESTS")
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
