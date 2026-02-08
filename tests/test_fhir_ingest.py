"""Tests for FHIR dual-mode ingestion."""

import json
import os
import tempfile
import unittest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fhir_ingest import parse_fhir_bundle


SAMPLE_BUNDLE = {
    "resourceType": "Bundle",
    "id": "test-bundle",
    "type": "collection",
    "entry": [
        {
            "resource": {
                "resourceType": "Patient",
                "id": "pat-001",
                "name": [{"use": "official", "family": "Smith", "given": ["John"]}],
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "id": "obs-a1c-001",
                "status": "final",
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "4548-4",
                            "display": "Hemoglobin A1c/Hemoglobin.total in Blood",
                        }
                    ]
                },
                "subject": {"reference": "Patient/pat-001"},
                "valueQuantity": {"value": 8.2, "unit": "%"},
            }
        },
    ],
}

BUNDLE_NO_A1C = {
    "resourceType": "Bundle",
    "id": "test-bundle-no-a1c",
    "type": "collection",
    "entry": [
        {
            "resource": {
                "resourceType": "Patient",
                "id": "pat-002",
                "name": [{"use": "official", "family": "Doe", "given": ["Jane"]}],
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "id": "obs-bp-001",
                "status": "final",
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "85354-9",
                            "display": "Blood pressure panel",
                        }
                    ]
                },
                "subject": {"reference": "Patient/pat-002"},
                "valueQuantity": {"value": 120, "unit": "mmHg"},
            }
        },
    ],
}


def _write_bundle(bundle: dict) -> str:
    """Write a bundle dict to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(bundle, f)
    return path


class TestFHIRIngest(unittest.TestCase):

    def test_extract_patient_name(self):
        path = _write_bundle(SAMPLE_BUNDLE)
        try:
            result = parse_fhir_bundle(path)
            self.assertEqual(result["patient_name"], "John Smith")
        finally:
            os.unlink(path)

    def test_extract_a1c_value(self):
        path = _write_bundle(SAMPLE_BUNDLE)
        try:
            result = parse_fhir_bundle(path)
            self.assertAlmostEqual(result["a1c_value"], 8.2)
        finally:
            os.unlink(path)

    def test_missing_a1c_returns_none(self):
        path = _write_bundle(BUNDLE_NO_A1C)
        try:
            result = parse_fhir_bundle(path)
            self.assertEqual(result["patient_name"], "Jane Doe")
            self.assertIsNone(result["a1c_value"])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
