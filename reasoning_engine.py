"""Reasoning engine - identifies care gaps using deterministic rules.

This is the "Therefore" step - pure code-based reasoning with no LLM.
LLM extracts facts, CODE reasons, LLM explains.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
from extraction import ExtractedFacts


@dataclass
class GapResult:
    """Result of a care gap analysis."""

    gap_type: str  # "A1C_THRESHOLD", "HTN_ACE_ARB", "BP_CONTROL", etc.
    gap_detected: bool
    patient_fact: dict  # {"value": 8.2, "source": "PATIENT:PT001"}
    guideline_fact: dict  # {"threshold": 7.0, "source": "GUIDE:001"}
    comparison: str  # "8.2 > 7.0" or "HTN present AND Lisinopril missing"
    therefore: str  # "Therefore, A1C is above target."
    recommendation: str  # "Consider treatment escalation."
    severity: str = "moderate"  # "low", "moderate", "high"
    guideline_id: str = ""  # Reference to guideline document

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ReasoningResult:
    """Complete reasoning output for a patient."""

    patient_id: str
    gaps: list[GapResult] = field(default_factory=list)
    gaps_found: int = 0
    gaps_closed: int = 0
    overall_status: str = "unknown"  # "all_gaps_closed", "gaps_identified", "needs_review"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "patient_id": self.patient_id,
            "gaps": [g.to_dict() for g in self.gaps],
            "gaps_found": self.gaps_found,
            "gaps_closed": self.gaps_closed,
            "overall_status": self.overall_status,
        }


class CareGapRules:
    """Deterministic rules for identifying care gaps.

    Each rule is pure code - no LLM involvement.
    """

    # Guideline thresholds (from medical KB)
    A1C_TARGET = 7.0  # ADA guideline
    BP_SYSTOLIC_TARGET = 140  # mmHg
    BP_DIASTOLIC_TARGET = 90  # mmHg

    # ACE inhibitor / ARB medication names
    ACE_INHIBITORS = [
        "lisinopril", "enalapril", "ramipril", "benazepril",
        "captopril", "fosinopril", "moexipril", "perindopril",
        "quinapril", "trandolapril"
    ]
    ARBS = [
        "losartan", "valsartan", "irbesartan", "candesartan",
        "olmesartan", "telmisartan", "azilsartan", "eprosartan"
    ]

    @classmethod
    def check_a1c_threshold(
        cls,
        facts: ExtractedFacts,
        patient_id: str
    ) -> GapResult:
        """Check if A1C is above goal.

        Rule: If patient has diabetes AND A1C >= 7.0%, gap is detected.

        Args:
            facts: Extracted patient facts
            patient_id: Patient identifier

        Returns:
            GapResult for A1C threshold check
        """
        # Check if patient has diabetes
        has_diabetes = any(
            "diabetes" in dx.lower()
            for dx in facts.diagnoses
        )

        if not has_diabetes:
            return GapResult(
                gap_type="A1C_THRESHOLD",
                gap_detected=False,
                patient_fact={"has_diabetes": False, "source": f"PATIENT:{patient_id}"},
                guideline_fact={"applies_to": "diabetic patients", "source": "GUIDE:001"},
                comparison="Patient does not have diabetes diagnosis",
                therefore="A1C threshold rule does not apply to non-diabetic patients.",
                recommendation="No action needed for A1C.",
                severity="low",
                guideline_id="guideline_001_a1c_threshold"
            )

        if facts.a1c is None:
            return GapResult(
                gap_type="A1C_THRESHOLD",
                gap_detected=True,
                patient_fact={"a1c": None, "source": f"PATIENT:{patient_id}"},
                guideline_fact={"threshold": cls.A1C_TARGET, "source": "GUIDE:001"},
                comparison="A1C value not found in patient record",
                therefore="Therefore, A1C status cannot be determined. Testing may be overdue.",
                recommendation="Order A1C test to assess glycemic control.",
                severity="moderate",
                guideline_id="guideline_001_a1c_threshold"
            )

        gap_detected = facts.a1c >= cls.A1C_TARGET

        if gap_detected:
            # Determine severity based on how far above goal
            if facts.a1c >= 9.0:
                severity = "high"
                rec = "Urgent treatment intensification needed. Consider adding second agent or insulin."
            elif facts.a1c >= 8.0:
                severity = "moderate"
                rec = "Consider adding second diabetes agent or adjusting current regimen."
            else:
                severity = "low"
                rec = "Monitor closely. Reinforce lifestyle modifications and medication adherence."

            return GapResult(
                gap_type="A1C_THRESHOLD",
                gap_detected=True,
                patient_fact={"a1c": facts.a1c, "source": f"PATIENT:{patient_id}"},
                guideline_fact={"threshold": cls.A1C_TARGET, "source": "GUIDE:001"},
                comparison=f"{facts.a1c}% >= {cls.A1C_TARGET}%",
                therefore=f"Therefore, A1C of {facts.a1c}% is above the target of {cls.A1C_TARGET}%.",
                recommendation=rec,
                severity=severity,
                guideline_id="guideline_001_a1c_threshold"
            )
        else:
            return GapResult(
                gap_type="A1C_THRESHOLD",
                gap_detected=False,
                patient_fact={"a1c": facts.a1c, "source": f"PATIENT:{patient_id}"},
                guideline_fact={"threshold": cls.A1C_TARGET, "source": "GUIDE:001"},
                comparison=f"{facts.a1c}% < {cls.A1C_TARGET}%",
                therefore=f"Therefore, A1C of {facts.a1c}% is at goal (target < {cls.A1C_TARGET}%).",
                recommendation="Continue current diabetes management. Maintain lifestyle modifications.",
                severity="low",
                guideline_id="guideline_001_a1c_threshold"
            )

    @classmethod
    def check_htn_ace_arb(
        cls,
        facts: ExtractedFacts,
        patient_id: str
    ) -> GapResult:
        """Check if patient with diabetes + HTN is on ACE/ARB.

        Rule: If patient has diabetes AND hypertension AND NOT on ACE/ARB, gap is detected.

        Args:
            facts: Extracted patient facts
            patient_id: Patient identifier

        Returns:
            GapResult for HTN ACE/ARB check
        """
        # Check diagnoses
        has_diabetes = any("diabetes" in dx.lower() for dx in facts.diagnoses)
        has_htn = any("hypertension" in dx.lower() for dx in facts.diagnoses)

        if not has_diabetes or not has_htn:
            missing = []
            if not has_diabetes:
                missing.append("diabetes")
            if not has_htn:
                missing.append("hypertension")

            return GapResult(
                gap_type="HTN_ACE_ARB",
                gap_detected=False,
                patient_fact={
                    "has_diabetes": has_diabetes,
                    "has_htn": has_htn,
                    "source": f"PATIENT:{patient_id}"
                },
                guideline_fact={
                    "requires": "diabetes AND hypertension",
                    "source": "GUIDE:002"
                },
                comparison=f"Patient missing: {', '.join(missing)}",
                therefore="ACE/ARB requirement rule does not apply - patient does not have both diabetes and hypertension.",
                recommendation="No action needed for ACE/ARB based on current diagnoses.",
                severity="low",
                guideline_id="guideline_002_htn_ace_inhibitor"
            )

        # Check medications for ACE inhibitor or ARB
        meds_lower = [m.lower() for m in facts.medications]

        has_ace = any(
            ace in med
            for med in meds_lower
            for ace in cls.ACE_INHIBITORS
        )
        has_arb = any(
            arb in med
            for med in meds_lower
            for arb in cls.ARBS
        )

        on_ace_or_arb = has_ace or has_arb

        if on_ace_or_arb:
            med_type = "ACE inhibitor" if has_ace else "ARB"
            # Find the specific medication
            found_med = None
            for med in facts.medications:
                med_lower = med.lower()
                if any(ace in med_lower for ace in cls.ACE_INHIBITORS):
                    found_med = med
                    break
                if any(arb in med_lower for arb in cls.ARBS):
                    found_med = med
                    break

            return GapResult(
                gap_type="HTN_ACE_ARB",
                gap_detected=False,
                patient_fact={
                    "has_diabetes": True,
                    "has_htn": True,
                    "on_ace_arb": True,
                    "medication": found_med,
                    "source": f"PATIENT:{patient_id}"
                },
                guideline_fact={
                    "requires": "ACE inhibitor or ARB for DM + HTN",
                    "source": "GUIDE:002"
                },
                comparison=f"HTN present AND {med_type} ({found_med}) present",
                therefore=f"Therefore, patient is appropriately on {med_type} therapy for diabetes with hypertension.",
                recommendation="Continue current ACE/ARB therapy. Monitor potassium and creatinine.",
                severity="low",
                guideline_id="guideline_002_htn_ace_inhibitor"
            )
        else:
            return GapResult(
                gap_type="HTN_ACE_ARB",
                gap_detected=True,
                patient_fact={
                    "has_diabetes": True,
                    "has_htn": True,
                    "on_ace_arb": False,
                    "current_meds": facts.medications,
                    "source": f"PATIENT:{patient_id}"
                },
                guideline_fact={
                    "requires": "ACE inhibitor or ARB for DM + HTN",
                    "source": "GUIDE:002"
                },
                comparison="HTN present AND ACE/ARB absent",
                therefore="Therefore, patient with diabetes and hypertension is NOT on recommended ACE inhibitor or ARB therapy.",
                recommendation="Initiate ACE inhibitor (e.g., Lisinopril 5-10mg daily) unless contraindicated. Provides BP control and renal protection.",
                severity="high",
                guideline_id="guideline_002_htn_ace_inhibitor"
            )

    @classmethod
    def check_bp_control(
        cls,
        facts: ExtractedFacts,
        patient_id: str
    ) -> GapResult:
        """Check if blood pressure is controlled.

        Rule: If patient has hypertension AND BP >= 140/90, gap is detected.

        Args:
            facts: Extracted patient facts
            patient_id: Patient identifier

        Returns:
            GapResult for BP control check
        """
        has_htn = any("hypertension" in dx.lower() for dx in facts.diagnoses)

        if not has_htn:
            return GapResult(
                gap_type="BP_CONTROL",
                gap_detected=False,
                patient_fact={"has_htn": False, "source": f"PATIENT:{patient_id}"},
                guideline_fact={"applies_to": "hypertensive patients", "source": "GUIDE:004"},
                comparison="Patient does not have hypertension diagnosis",
                therefore="BP control rule does not apply to non-hypertensive patients.",
                recommendation="No action needed for BP control.",
                severity="low",
                guideline_id="guideline_004_bp_target"
            )

        if facts.blood_pressure is None:
            return GapResult(
                gap_type="BP_CONTROL",
                gap_detected=True,
                patient_fact={"bp": None, "source": f"PATIENT:{patient_id}"},
                guideline_fact={
                    "systolic_target": cls.BP_SYSTOLIC_TARGET,
                    "diastolic_target": cls.BP_DIASTOLIC_TARGET,
                    "source": "GUIDE:004"
                },
                comparison="BP value not found in patient record",
                therefore="Therefore, BP status cannot be determined.",
                recommendation="Check blood pressure at next visit.",
                severity="moderate",
                guideline_id="guideline_004_bp_target"
            )

        systolic = facts.blood_pressure.get("systolic", 0)
        diastolic = facts.blood_pressure.get("diastolic", 0)

        bp_elevated = systolic >= cls.BP_SYSTOLIC_TARGET or diastolic >= cls.BP_DIASTOLIC_TARGET

        if bp_elevated:
            # Determine severity
            if systolic >= 160 or diastolic >= 100:
                severity = "high"
                rec = "Significant hypertension. Consider adding/adjusting antihypertensive medications urgently."
            elif systolic >= 140 or diastolic >= 90:
                severity = "moderate"
                rec = "Consider intensifying antihypertensive therapy. Reinforce lifestyle modifications."
            else:
                severity = "low"
                rec = "Monitor BP closely. Reinforce dietary sodium restriction and exercise."

            return GapResult(
                gap_type="BP_CONTROL",
                gap_detected=True,
                patient_fact={
                    "bp": f"{systolic}/{diastolic}",
                    "systolic": systolic,
                    "diastolic": diastolic,
                    "source": f"PATIENT:{patient_id}"
                },
                guideline_fact={
                    "systolic_target": cls.BP_SYSTOLIC_TARGET,
                    "diastolic_target": cls.BP_DIASTOLIC_TARGET,
                    "source": "GUIDE:004"
                },
                comparison=f"{systolic}/{diastolic} >= {cls.BP_SYSTOLIC_TARGET}/{cls.BP_DIASTOLIC_TARGET}",
                therefore=f"Therefore, BP of {systolic}/{diastolic} mmHg is above target of <{cls.BP_SYSTOLIC_TARGET}/{cls.BP_DIASTOLIC_TARGET} mmHg.",
                recommendation=rec,
                severity=severity,
                guideline_id="guideline_004_bp_target"
            )
        else:
            return GapResult(
                gap_type="BP_CONTROL",
                gap_detected=False,
                patient_fact={
                    "bp": f"{systolic}/{diastolic}",
                    "systolic": systolic,
                    "diastolic": diastolic,
                    "source": f"PATIENT:{patient_id}"
                },
                guideline_fact={
                    "systolic_target": cls.BP_SYSTOLIC_TARGET,
                    "diastolic_target": cls.BP_DIASTOLIC_TARGET,
                    "source": "GUIDE:004"
                },
                comparison=f"{systolic}/{diastolic} < {cls.BP_SYSTOLIC_TARGET}/{cls.BP_DIASTOLIC_TARGET}",
                therefore=f"Therefore, BP of {systolic}/{diastolic} mmHg is at goal (target <{cls.BP_SYSTOLIC_TARGET}/{cls.BP_DIASTOLIC_TARGET} mmHg).",
                recommendation="Continue current antihypertensive regimen. Maintain lifestyle modifications.",
                severity="low",
                guideline_id="guideline_004_bp_target"
            )


class ReasoningEngine:
    """Identifies clinical care gaps using deterministic rules.

    This is the "Therefore" step - pure code-based reasoning.
    No LLM involved in the actual gap detection logic.
    """

    def __init__(self):
        """Initialize the reasoning engine."""
        self.rules = CareGapRules()

    def evaluate_patient(
        self,
        facts: ExtractedFacts,
        patient_id: str
    ) -> ReasoningResult:
        """Evaluate all care gap rules for a patient.

        Args:
            facts: Extracted patient facts
            patient_id: Patient identifier

        Returns:
            ReasoningResult with all gap evaluations
        """
        gaps = []

        # Run all rules
        gaps.append(self.rules.check_a1c_threshold(facts, patient_id))
        gaps.append(self.rules.check_htn_ace_arb(facts, patient_id))
        gaps.append(self.rules.check_bp_control(facts, patient_id))

        # Count gaps
        gaps_found = sum(1 for g in gaps if g.gap_detected)
        gaps_closed = sum(1 for g in gaps if not g.gap_detected)

        # Determine overall status
        if gaps_found == 0:
            overall_status = "all_gaps_closed"
        elif any(g.severity == "high" for g in gaps if g.gap_detected):
            overall_status = "urgent_gaps_identified"
        else:
            overall_status = "gaps_identified"

        return ReasoningResult(
            patient_id=patient_id,
            gaps=gaps,
            gaps_found=gaps_found,
            gaps_closed=gaps_closed,
            overall_status=overall_status
        )

    def get_detected_gaps(self, result: ReasoningResult) -> list[GapResult]:
        """Get only the gaps that were detected.

        Args:
            result: Full reasoning result

        Returns:
            List of detected gaps only
        """
        return [g for g in result.gaps if g.gap_detected]

    def get_closed_gaps(self, result: ReasoningResult) -> list[GapResult]:
        """Get only the gaps that were closed (not detected).

        Args:
            result: Full reasoning result

        Returns:
            List of closed gaps only
        """
        return [g for g in result.gaps if not g.gap_detected]

    def format_summary(self, result: ReasoningResult) -> str:
        """Format a human-readable summary of the reasoning result.

        Args:
            result: Reasoning result to summarize

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append(f"Patient {result.patient_id} - Care Gap Analysis")
        lines.append("=" * 50)

        detected = self.get_detected_gaps(result)
        closed = self.get_closed_gaps(result)

        if detected:
            lines.append(f"\nGAPS IDENTIFIED ({len(detected)}):")
            for gap in detected:
                lines.append(f"\n  [{gap.severity.upper()}] {gap.gap_type}")
                lines.append(f"    Comparison: {gap.comparison}")
                lines.append(f"    {gap.therefore}")
                lines.append(f"    Recommendation: {gap.recommendation}")
        else:
            lines.append("\nNo care gaps identified.")

        if closed:
            lines.append(f"\nGAPS CLOSED ({len(closed)}):")
            for gap in closed:
                lines.append(f"  - {gap.gap_type}: {gap.therefore}")

        lines.append(f"\nOverall Status: {result.overall_status}")

        return "\n".join(lines)


# Convenience functions
def evaluate_patient_gaps(facts: ExtractedFacts, patient_id: str) -> ReasoningResult:
    """Evaluate care gaps for a patient.

    Args:
        facts: Extracted patient facts
        patient_id: Patient identifier

    Returns:
        ReasoningResult with all gap evaluations
    """
    engine = ReasoningEngine()
    return engine.evaluate_patient(facts, patient_id)


# Test function
def test_reasoning():
    """Test reasoning engine on sample data."""
    from extraction import PatientFactExtractor
    from care_database import get_database

    db = get_database()
    extractor = PatientFactExtractor()
    engine = ReasoningEngine()

    print("=" * 60)
    print("REASONING ENGINE TEST")
    print("=" * 60)

    patients = db.get_all_patients()

    for patient in patients:
        patient_id = patient["patient_id"]
        note = db.get_latest_note(patient_id)

        if not note:
            continue

        facts = extractor.extract(note["note_text"])
        result = engine.evaluate_patient(facts, patient_id)

        print(f"\n{engine.format_summary(result)}")
        print("-" * 60)


if __name__ == "__main__":
    test_reasoning()
