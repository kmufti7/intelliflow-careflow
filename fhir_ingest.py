"""
FHIR Dual-Mode Ingestion — Demo-grade HL7 FHIR R4 parser.

Parses FHIR Bundles (JSON) to extract patient demographics and lab results,
enabling structured data ingestion alongside the existing regex/LLM pipeline
for unstructured clinical notes.
"""

import json


LOINC_A1C = "4548-4"


def parse_fhir_bundle(file_path: str) -> dict:
    """Parse a FHIR R4 Bundle and extract patient name + A1C value.

    Args:
        file_path: Path to a FHIR Bundle JSON file.

    Returns:
        dict with keys:
            - patient_name (str or None): "Given Family" format
            - a1c_value (float or None): A1C percentage from Observation
    """
    with open(file_path, "r") as f:
        bundle = json.load(f)

    patient_name = None
    a1c_value = None

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")

        if resource_type == "Patient":
            patient_name = _extract_patient_name(resource)

        elif resource_type == "Observation":
            a1c = _extract_a1c(resource)
            if a1c is not None:
                a1c_value = a1c

    return {"patient_name": patient_name, "a1c_value": a1c_value}


def _extract_patient_name(patient: dict) -> str | None:
    """Extract display name from Patient resource."""
    names = patient.get("name", [])
    if not names:
        return None
    name = names[0]
    given = " ".join(name.get("given", []))
    family = name.get("family", "")
    full = f"{given} {family}".strip()
    return full if full else None


def _extract_a1c(observation: dict) -> float | None:
    """Extract A1C value if Observation has LOINC code 4548-4."""
    codings = observation.get("code", {}).get("coding", [])
    for coding in codings:
        if coding.get("code") == LOINC_A1C:
            quantity = observation.get("valueQuantity", {})
            return quantity.get("value")
    return None
