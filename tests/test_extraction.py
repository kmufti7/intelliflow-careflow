"""Tests for the extraction layer.

Tests regex-first extraction of clinical facts from patient notes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction import PatientFactExtractor, ExtractedFacts
from care_database import get_database


class TestExtraction:
    """Test suite for PatientFactExtractor."""

    def __init__(self):
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

    def test_maria_garcia_a1c(self):
        """Test: Regex extracts A1C = 8.2 from Maria Garcia's note."""
        note = self.db.get_latest_note("PT001")
        facts = self.extractor.extract(note["note_text"])

        return self._assert(
            facts.a1c == 8.2,
            "Maria Garcia A1C extraction",
            f"Expected 8.2, got {facts.a1c}"
        )

    def test_maria_garcia_diagnoses(self):
        """Test: Regex extracts diagnoses including Hypertension."""
        note = self.db.get_latest_note("PT001")
        facts = self.extractor.extract(note["note_text"])

        # Check for hypertension (case-insensitive)
        has_htn = any("hypertension" in dx.lower() for dx in facts.diagnoses)

        return self._assert(
            has_htn,
            "Maria Garcia HTN diagnosis",
            f"Diagnoses found: {facts.diagnoses}"
        )

    def test_maria_garcia_medications(self):
        """Test: Maria Garcia is NOT on Lisinopril (care gap exists)."""
        note = self.db.get_latest_note("PT001")
        facts = self.extractor.extract(note["note_text"])

        # Check that Lisinopril is NOT in medications
        has_lisinopril = any("lisinopril" in med.lower() for med in facts.medications)

        return self._assert(
            not has_lisinopril,
            "Maria Garcia no Lisinopril",
            f"Medications found: {facts.medications}"
        )

    def test_maria_garcia_blood_pressure(self):
        """Test: Regex extracts BP = 142/94 from Maria Garcia."""
        note = self.db.get_latest_note("PT001")
        facts = self.extractor.extract(note["note_text"])

        bp_correct = (
            facts.blood_pressure is not None and
            facts.blood_pressure.get("systolic") == 142 and
            facts.blood_pressure.get("diastolic") == 94
        )

        return self._assert(
            bp_correct,
            "Maria Garcia BP extraction",
            f"Expected 142/94, got {facts.blood_pressure}"
        )

    def test_maria_garcia_extraction_method(self):
        """Test: Extraction uses regex (not LLM fallback)."""
        note = self.db.get_latest_note("PT001")
        facts = self.extractor.extract(note["note_text"])

        return self._assert(
            facts.extraction_method == "regex",
            "Maria Garcia regex extraction",
            f"Method: {facts.extraction_method}, Confidence: {facts.confidence}"
        )

    def test_james_wilson_extraction(self):
        """Test: James Wilson (PT002) extracts correctly."""
        note = self.db.get_latest_note("PT002")
        facts = self.extractor.extract(note["note_text"])

        # James has A1C 7.4, BP 128/82, on Lisinopril
        a1c_ok = facts.a1c == 7.4
        bp_ok = facts.blood_pressure and facts.blood_pressure.get("systolic") == 128
        has_lisinopril = any("lisinopril" in med.lower() for med in facts.medications)

        all_ok = a1c_ok and bp_ok and has_lisinopril

        return self._assert(
            all_ok,
            "James Wilson extraction",
            f"A1C={facts.a1c}, BP={facts.blood_pressure}, Lisinopril={has_lisinopril}"
        )

    def test_sarah_chen_extraction(self):
        """Test: Sarah Chen (PT003) extracts correctly."""
        note = self.db.get_latest_note("PT003")
        facts = self.extractor.extract(note["note_text"])

        # Sarah has A1C 9.1, newly diagnosed
        a1c_ok = facts.a1c == 9.1

        return self._assert(
            a1c_ok,
            "Sarah Chen extraction",
            f"A1C={facts.a1c}, Method={facts.extraction_method}"
        )

    def test_robert_johnson_extraction(self):
        """Test: Robert Johnson (PT004) extracts correctly."""
        note = self.db.get_latest_note("PT004")
        facts = self.extractor.extract(note["note_text"])

        # Robert has A1C 6.8, BP 148/94, HTN but no ACE/ARB
        a1c_ok = facts.a1c == 6.8
        bp_ok = facts.blood_pressure and facts.blood_pressure.get("systolic") == 148

        return self._assert(
            a1c_ok and bp_ok,
            "Robert Johnson extraction",
            f"A1C={facts.a1c}, BP={facts.blood_pressure}"
        )

    def test_linda_martinez_extraction(self):
        """Test: Linda Martinez (PT005) extracts correctly."""
        note = self.db.get_latest_note("PT005")
        facts = self.extractor.extract(note["note_text"])

        # Linda has A1C 7.0, at goal
        a1c_ok = facts.a1c == 7.0

        return self._assert(
            a1c_ok,
            "Linda Martinez extraction",
            f"A1C={facts.a1c}, Method={facts.extraction_method}"
        )

    def test_all_patients_use_regex(self):
        """Test: All 5 patients extract with regex (no LLM fallback needed)."""
        patients = ["PT001", "PT002", "PT003", "PT004", "PT005"]
        all_regex = True

        for patient_id in patients:
            note = self.db.get_latest_note(patient_id)
            if note:
                facts = self.extractor.extract(note["note_text"])
                if facts.extraction_method != "regex":
                    all_regex = False
                    break

        return self._assert(
            all_regex,
            "All patients use regex extraction",
            f"All 5 patients extracted without LLM fallback"
        )

    def test_negation_handling(self):
        """Test: Negated diagnoses are not extracted as positive diagnoses."""
        # Test using a real patient note (PT005) which has "No hypertension"
        note = self.db.get_latest_note("PT005")
        facts = self.extractor.extract(note["note_text"])

        # PT005's note says "No hypertension" - should NOT appear as a diagnosis
        # But should have diabetes
        has_diabetes = any("diabetes" in dx.lower() for dx in facts.diagnoses)

        # Check that "hypertension" is NOT in the diagnosis list
        # (since it's negated in the note)
        has_hypertension = any("hypertension" in dx.lower() for dx in facts.diagnoses)

        return self._assert(
            has_diabetes and not has_hypertension,
            "Negation handling",
            f"Has DM: {has_diabetes}, Has HTN: {has_hypertension}, Diagnoses: {facts.diagnoses}"
        )

    def run_all(self) -> dict:
        """Run all extraction tests."""
        self.results = []

        # Run all test methods
        self.test_maria_garcia_a1c()
        self.test_maria_garcia_diagnoses()
        self.test_maria_garcia_medications()
        self.test_maria_garcia_blood_pressure()
        self.test_maria_garcia_extraction_method()
        self.test_james_wilson_extraction()
        self.test_sarah_chen_extraction()
        self.test_robert_johnson_extraction()
        self.test_linda_martinez_extraction()
        self.test_all_patients_use_regex()
        self.test_negation_handling()

        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)

        return {
            "suite": "Extraction",
            "passed": passed,
            "total": total,
            "results": self.results
        }


def run_tests():
    """Run extraction tests and print results."""
    suite = TestExtraction()
    results = suite.run_all()

    print("=" * 60)
    print("EXTRACTION TESTS")
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
