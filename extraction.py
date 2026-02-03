"""Extraction module - extracts structured data from patient notes.

Uses regex-first approach with LLM fallback for robustness.
"""

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ExtractedFacts:
    """Structured clinical facts extracted from a patient note."""

    a1c: Optional[float] = None  # e.g., 8.2
    blood_pressure: Optional[dict] = None  # e.g., {"systolic": 140, "diastolic": 90}
    diagnoses: list[str] = field(default_factory=list)  # e.g., ["Type 2 Diabetes", "Hypertension"]
    medications: list[str] = field(default_factory=list)  # e.g., ["Metformin 500mg", "Lisinopril 10mg"]
    extraction_method: str = "regex"  # "regex" or "llm"
    confidence: float = 1.0  # 0.0-1.0
    raw_extractions: dict = field(default_factory=dict)  # For debugging

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def is_complete(self) -> bool:
        """Check if all critical fields were extracted."""
        return (
            self.a1c is not None and
            self.blood_pressure is not None and
            len(self.diagnoses) > 0 and
            len(self.medications) > 0
        )

    def missing_fields(self) -> list[str]:
        """Return list of fields that couldn't be extracted."""
        missing = []
        if self.a1c is None:
            missing.append("a1c")
        if self.blood_pressure is None:
            missing.append("blood_pressure")
        if len(self.diagnoses) == 0:
            missing.append("diagnoses")
        if len(self.medications) == 0:
            missing.append("medications")
        return missing


class PatientFactExtractor:
    """Extracts clinical facts from patient notes using regex-first, LLM fallback."""

    # Regex patterns for clinical data extraction
    PATTERNS = {
        # A1C patterns: "A1C: 8.2%", "A1C 8.2", "HbA1c: 8.2%", "A1C of 8.2%"
        "a1c": re.compile(
            r"(?:A1C|HbA1c|Hemoglobin A1c)[\s:]*(?:of\s+)?(\d+\.?\d*)\s*%?",
            re.IGNORECASE
        ),

        # Blood pressure patterns: "BP: 140/90", "BP 140/90 mmHg", "Blood Pressure: 140/90"
        "bp": re.compile(
            r"(?:BP|Blood\s*Pressure)[\s:]*(\d{2,3})\s*/\s*(\d{2,3})(?:\s*mmHg)?",
            re.IGNORECASE
        ),

        # Section patterns for extracting diagnoses and medications
        "assessment_section": re.compile(
            r"(?:Assessment|Dx|Diagnosis|Diagnoses|A/P)[\s:]*\n?(.*?)(?=\n\s*(?:Plan|Current Medications|Medications|$))",
            re.IGNORECASE | re.DOTALL
        ),

        "medications_section": re.compile(
            r"(?:Current\s+Medications|Medications|Meds)[\s:]*\n?(.*?)(?=\n\s*(?:Plan|Assessment|$))",
            re.IGNORECASE | re.DOTALL
        ),

        # Individual diagnosis pattern (within assessment section)
        "diagnosis_line": re.compile(
            r"[-•]\s*(.+?)(?:\s*[-–]\s*.+)?$",
            re.MULTILINE
        ),

        # Individual medication pattern (within medications section)
        "medication_line": re.compile(
            r"[-•]\s*(.+?)$",
            re.MULTILINE
        ),
    }

    # Common diagnosis normalization
    DIAGNOSIS_KEYWORDS = {
        "diabetes": "Type 2 Diabetes Mellitus",
        "type 2 diabetes": "Type 2 Diabetes Mellitus",
        "dm": "Type 2 Diabetes Mellitus",
        "t2dm": "Type 2 Diabetes Mellitus",
        "hypertension": "Essential Hypertension",
        "htn": "Essential Hypertension",
        "essential hypertension": "Essential Hypertension",
        "ckd": "Chronic Kidney Disease",
        "chronic kidney disease": "Chronic Kidney Disease",
    }

    def __init__(self):
        """Initialize the extractor with OpenAI client for LLM fallback."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
                    api_key = st.secrets["OPENAI_API_KEY"]
            except:
                pass

        self.client = OpenAI(api_key=api_key) if api_key else None

    def extract(self, note_text: str) -> ExtractedFacts:
        """Extract clinical facts from a patient note.

        Uses regex-first approach, falls back to LLM if regex extraction
        is incomplete.

        Args:
            note_text: The clinical note text to extract from

        Returns:
            ExtractedFacts with extracted data
        """
        # First try regex extraction
        facts = self._extract_with_regex(note_text)

        # If extraction is incomplete, try LLM fallback
        if not facts.is_complete() and self.client:
            missing = facts.missing_fields()
            llm_facts = self._extract_with_llm(note_text, missing)

            # Merge LLM results into facts
            facts = self._merge_facts(facts, llm_facts)
            facts.extraction_method = "regex+llm"
            facts.confidence = 0.85  # Lower confidence for hybrid extraction

        return facts

    def _extract_with_regex(self, note_text: str) -> ExtractedFacts:
        """Extract facts using regex patterns.

        Args:
            note_text: The clinical note text

        Returns:
            ExtractedFacts with regex-extracted data
        """
        raw = {}

        # Extract A1C
        a1c = None
        a1c_match = self.PATTERNS["a1c"].search(note_text)
        if a1c_match:
            try:
                a1c = float(a1c_match.group(1))
                raw["a1c_match"] = a1c_match.group(0)
            except ValueError:
                pass

        # Extract Blood Pressure
        blood_pressure = None
        bp_match = self.PATTERNS["bp"].search(note_text)
        if bp_match:
            try:
                blood_pressure = {
                    "systolic": int(bp_match.group(1)),
                    "diastolic": int(bp_match.group(2))
                }
                raw["bp_match"] = bp_match.group(0)
            except ValueError:
                pass

        # Extract diagnoses from Assessment section
        diagnoses = []
        assessment_match = self.PATTERNS["assessment_section"].search(note_text)
        if assessment_match:
            assessment_text = assessment_match.group(1)
            raw["assessment_section"] = assessment_text.strip()

            # Find diagnosis lines
            for line_match in self.PATTERNS["diagnosis_line"].finditer(assessment_text):
                dx = line_match.group(1).strip()
                if dx:
                    # Clean up and normalize
                    dx_clean = self._normalize_diagnosis(dx)
                    if dx_clean and dx_clean not in diagnoses:
                        diagnoses.append(dx_clean)

        # Extract medications from Medications section
        medications = []
        meds_match = self.PATTERNS["medications_section"].search(note_text)
        if meds_match:
            meds_text = meds_match.group(1)
            raw["medications_section"] = meds_text.strip()

            # Find medication lines
            for line_match in self.PATTERNS["medication_line"].finditer(meds_text):
                med = line_match.group(1).strip()
                if med:
                    # Clean up medication name
                    med_clean = self._clean_medication(med)
                    if med_clean and med_clean not in medications:
                        medications.append(med_clean)

        return ExtractedFacts(
            a1c=a1c,
            blood_pressure=blood_pressure,
            diagnoses=diagnoses,
            medications=medications,
            extraction_method="regex",
            confidence=1.0 if a1c and blood_pressure and diagnoses and medications else 0.7,
            raw_extractions=raw
        )

    def _extract_with_llm(self, note_text: str, missing_fields: list[str]) -> ExtractedFacts:
        """Extract facts using LLM when regex fails.

        Args:
            note_text: The clinical note text
            missing_fields: List of fields that regex couldn't extract

        Returns:
            ExtractedFacts with LLM-extracted data
        """
        # Build prompt for specific missing fields
        field_instructions = []
        if "a1c" in missing_fields:
            field_instructions.append("- a1c: The A1C/HbA1c value as a number (e.g., 8.2)")
        if "blood_pressure" in missing_fields:
            field_instructions.append('- blood_pressure: Object with "systolic" and "diastolic" integer values')
        if "diagnoses" in missing_fields:
            field_instructions.append("- diagnoses: Array of diagnosis strings from the assessment")
        if "medications" in missing_fields:
            field_instructions.append("- medications: Array of medication strings with dosages")

        prompt = f"""Extract the following clinical data from this patient note. Return ONLY a JSON object with the requested fields. If a field cannot be found, use null for single values or empty array for lists.

Fields to extract:
{chr(10).join(field_instructions)}

Patient Note:
{note_text}

Return only valid JSON, no markdown formatting."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a clinical data extraction assistant. Extract structured data from clinical notes. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=500
            )

            result = json.loads(response.choices[0].message.content)

            return ExtractedFacts(
                a1c=result.get("a1c"),
                blood_pressure=result.get("blood_pressure"),
                diagnoses=result.get("diagnoses", []),
                medications=result.get("medications", []),
                extraction_method="llm",
                confidence=0.8,
                raw_extractions={"llm_response": result}
            )

        except Exception as e:
            # Return empty facts if LLM fails
            return ExtractedFacts(
                extraction_method="llm_failed",
                confidence=0.0,
                raw_extractions={"error": str(e)}
            )

    def _merge_facts(self, regex_facts: ExtractedFacts, llm_facts: ExtractedFacts) -> ExtractedFacts:
        """Merge regex and LLM extraction results.

        Prefers regex results when available, fills in with LLM results.

        Args:
            regex_facts: Facts extracted via regex
            llm_facts: Facts extracted via LLM

        Returns:
            Merged ExtractedFacts
        """
        merged_raw = {**regex_facts.raw_extractions, **llm_facts.raw_extractions}

        return ExtractedFacts(
            a1c=regex_facts.a1c if regex_facts.a1c is not None else llm_facts.a1c,
            blood_pressure=regex_facts.blood_pressure if regex_facts.blood_pressure else llm_facts.blood_pressure,
            diagnoses=regex_facts.diagnoses if regex_facts.diagnoses else llm_facts.diagnoses,
            medications=regex_facts.medications if regex_facts.medications else llm_facts.medications,
            extraction_method="regex+llm",
            confidence=0.85,
            raw_extractions=merged_raw
        )

    def _normalize_diagnosis(self, dx: str) -> str:
        """Normalize a diagnosis string.

        Args:
            dx: Raw diagnosis string

        Returns:
            Normalized diagnosis string, or empty string if negated
        """
        dx_lower = dx.lower()

        # Skip negated diagnoses
        negation_patterns = [
            r'^no\s+',
            r'^no\s+evidence\s+of\s+',
            r'^denies\s+',
            r'^negative\s+for\s+',
            r'^without\s+',
            r'^ruled\s+out\s+',
        ]
        for pattern in negation_patterns:
            if re.match(pattern, dx_lower):
                return ""

        # Remove common suffixes like "- controlled", "- at goal"
        dx_clean = re.sub(r'\s*[-–]\s*(controlled|at goal|not at goal|suboptimally controlled|poorly controlled|well controlled|stable|new diagnosis.*?)$', '', dx, flags=re.IGNORECASE)
        dx_clean = dx_clean.strip()

        # Check for keyword normalization
        dx_lower = dx_clean.lower()
        for keyword, normalized in self.DIAGNOSIS_KEYWORDS.items():
            if keyword in dx_lower:
                return normalized

        return dx_clean

    def _clean_medication(self, med: str) -> str:
        """Clean up a medication string.

        Args:
            med: Raw medication string

        Returns:
            Cleaned medication string
        """
        # Remove parenthetical notes
        med_clean = re.sub(r'\s*\([^)]+\)\s*', ' ', med)
        # Normalize whitespace
        med_clean = ' '.join(med_clean.split())
        return med_clean.strip()


# Convenience function
def extract_patient_facts(note_text: str) -> ExtractedFacts:
    """Extract clinical facts from a patient note.

    Args:
        note_text: The clinical note text

    Returns:
        ExtractedFacts with extracted data
    """
    extractor = PatientFactExtractor()
    return extractor.extract(note_text)


# Test function
def test_extraction():
    """Test extraction on sample note."""
    sample_note = """Subjective: Patient reports fatigue, increased thirst.

Objective:
- Vitals: BP 142/94 mmHg, HR 78 bpm
- Labs: A1C 8.2%, Fasting glucose 165 mg/dL

Assessment:
- Type 2 Diabetes Mellitus - suboptimally controlled
- Essential Hypertension - not at goal

Current Medications:
- Metformin 1000mg BID
- Amlodipine 5mg daily

Plan:
- Follow up in 3 months"""

    extractor = PatientFactExtractor()
    facts = extractor.extract(sample_note)

    print("Extracted Facts:")
    print(f"  A1C: {facts.a1c}")
    print(f"  BP: {facts.blood_pressure}")
    print(f"  Diagnoses: {facts.diagnoses}")
    print(f"  Medications: {facts.medications}")
    print(f"  Method: {facts.extraction_method}")
    print(f"  Confidence: {facts.confidence}")
    print(f"  Complete: {facts.is_complete()}")

    return facts


if __name__ == "__main__":
    test_extraction()
