"""Test extraction on all 5 patients."""

from care_database import get_database
from extraction import PatientFactExtractor


def test_all_patients():
    """Test extraction on all patients in the database."""
    db = get_database()
    extractor = PatientFactExtractor()

    patients = db.get_all_patients()

    print("=" * 70)
    print("EXTRACTION TEST - ALL PATIENTS")
    print("=" * 70)

    # Expected values for verification
    expected = {
        "PT001": {"a1c": 8.2, "bp_sys": 142, "has_htn": True, "has_ace": False},
        "PT002": {"a1c": 7.4, "bp_sys": 128, "has_htn": True, "has_ace": True},
        "PT003": {"a1c": 9.1, "bp_sys": 118, "has_htn": False, "has_ace": False},
        "PT004": {"a1c": 6.8, "bp_sys": 148, "has_htn": True, "has_ace": False},
        "PT005": {"a1c": 7.0, "bp_sys": 122, "has_htn": False, "has_ace": True},
    }

    all_passed = True

    for patient in patients:
        patient_id = patient["patient_id"]
        note = db.get_latest_note(patient_id)

        if not note:
            print(f"\n{patient_id}: No note found!")
            continue

        facts = extractor.extract(note["note_text"])
        exp = expected.get(patient_id, {})

        print(f"\n{'-' * 70}")
        print(f"Patient: {patient['name']} ({patient_id})")
        print(f"{'-' * 70}")

        # Check A1C
        a1c_match = facts.a1c == exp.get("a1c")
        a1c_status = "PASS" if a1c_match else "FAIL"
        print(f"  A1C: {facts.a1c}% (expected: {exp.get('a1c')}%) [{a1c_status}]")
        if not a1c_match:
            all_passed = False

        # Check BP
        if facts.blood_pressure:
            bp_match = facts.blood_pressure["systolic"] == exp.get("bp_sys")
            bp_status = "PASS" if bp_match else "FAIL"
            print(f"  BP: {facts.blood_pressure['systolic']}/{facts.blood_pressure['diastolic']} mmHg (expected systolic: {exp.get('bp_sys')}) [{bp_status}]")
            if not bp_match:
                all_passed = False
        else:
            print(f"  BP: Not found [FAIL]")
            all_passed = False

        # Check diagnoses
        dx_lower = [d.lower() for d in facts.diagnoses]
        has_htn = any("hypertension" in d for d in dx_lower)
        has_diabetes = any("diabetes" in d for d in dx_lower)

        htn_expected = exp.get("has_htn", False)
        htn_match = has_htn == htn_expected
        htn_status = "PASS" if htn_match else "FAIL"
        print(f"  Has HTN: {has_htn} (expected: {htn_expected}) [{htn_status}]")
        if not htn_match:
            all_passed = False

        print(f"  Has Diabetes: {has_diabetes}")
        print(f"  Diagnoses: {facts.diagnoses}")

        # Check medications
        med_lower = [m.lower() for m in facts.medications]
        has_ace = any("lisinopril" in m or "enalapril" in m or "ramipril" in m for m in med_lower)
        has_arb = any("losartan" in m or "valsartan" in m for m in med_lower)
        has_ace_or_arb = has_ace or has_arb

        ace_expected = exp.get("has_ace", False)
        ace_match = has_ace_or_arb == ace_expected
        ace_status = "PASS" if ace_match else "FAIL"
        print(f"  Has ACE/ARB: {has_ace_or_arb} (expected: {ace_expected}) [{ace_status}]")
        if not ace_match:
            all_passed = False

        print(f"  Medications: {facts.medications}")

        # Extraction metadata
        print(f"\n  Extraction Method: {facts.extraction_method}")
        print(f"  Confidence: {facts.confidence:.0%}")
        print(f"  Complete: {facts.is_complete()}")

    print(f"\n{'=' * 70}")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    test_all_patients()
