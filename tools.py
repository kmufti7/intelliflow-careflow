"""Tools module - utility functions and tools for CareFlow agents.

Includes booking tool, vector search, and clinical utilities.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class BookingResult:
    """Result of an appointment booking."""
    success: bool
    appointment_id: Optional[str] = None
    patient_id: Optional[str] = None
    doctor_id: Optional[str] = None
    doctor_name: Optional[str] = None
    specialty: Optional[str] = None
    slot_datetime: Optional[str] = None
    reason: Optional[str] = None
    message: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class BookingTool:
    """Tool for booking appointments with specialists."""

    # Specialty mappings for gap-based referrals
    GAP_TO_SPECIALTY = {
        "A1C_THRESHOLD": "Endocrinology",
        "HTN_ACE_ARB": "Cardiology",
        "BP_CONTROL": "Cardiology",
        "KIDNEY_FUNCTION": "Nephrology",
        "STATIN": "Cardiology",
        "FOOT_EXAM": "Podiatry",
        "EYE_EXAM": "Ophthalmology",
    }

    def __init__(self, db=None):
        """Initialize booking tool with database connection.

        Args:
            db: CareDatabase instance (will get from singleton if not provided)
        """
        self.db = db

    def _get_db(self):
        """Get database connection."""
        if self.db is None:
            from care_database import get_database
            self.db = get_database()
        return self.db

    def book_appointment(
        self,
        patient_id: str,
        specialty: str,
        reason: str,
        preferred_date: Optional[str] = None
    ) -> BookingResult:
        """Book an appointment for a patient with a specialist.

        Args:
            patient_id: Patient identifier
            specialty: Medical specialty (e.g., "Endocrinology", "Cardiology")
            reason: Reason for the appointment
            preferred_date: Optional preferred date (YYYY-MM-DD)

        Returns:
            BookingResult with booking details
        """
        db = self._get_db()

        # Verify patient exists
        patient = db.get_patient(patient_id)
        if not patient:
            return BookingResult(
                success=False,
                patient_id=patient_id,
                message="Patient not found",
                error=f"No patient found with ID {patient_id}"
            )

        # Find doctors with the specified specialty
        doctors = db.get_doctors_by_specialty(specialty)
        if not doctors:
            # Try case-insensitive match
            all_doctors = db.get_all_doctors()
            doctors = [d for d in all_doctors if d["specialty"].lower() == specialty.lower()]

        if not doctors:
            return BookingResult(
                success=False,
                patient_id=patient_id,
                specialty=specialty,
                message=f"No doctors found for specialty: {specialty}",
                error=f"Specialty '{specialty}' not available"
            )

        # Find available slot
        available_slots = []
        for doctor in doctors:
            slots = db.get_available_slots(doctor["doctor_id"])
            if slots:
                # Filter by preferred date if specified
                if preferred_date:
                    slots = [s for s in slots if s["slot_datetime"].startswith(preferred_date)]
                available_slots.extend(slots)

        if not available_slots:
            # Try to find any available slot in the future
            for doctor in doctors:
                slots = db.get_available_slots(doctor["doctor_id"])
                available_slots.extend(slots)

        if not available_slots:
            return BookingResult(
                success=False,
                patient_id=patient_id,
                specialty=specialty,
                message=f"No available slots for {specialty}",
                error="No available appointment slots"
            )

        # Sort by datetime and pick the earliest
        available_slots.sort(key=lambda s: s["slot_datetime"])
        selected_slot = available_slots[0]

        # Create the appointment
        try:
            appointment_id = db.create_appointment(
                patient_id=patient_id,
                doctor_id=selected_slot["doctor_id"],
                slot_datetime=selected_slot["slot_datetime"],
                reason=reason
            )

            # Format datetime for display
            slot_dt = datetime.fromisoformat(selected_slot["slot_datetime"])
            formatted_datetime = slot_dt.strftime("%B %d, %Y at %I:%M %p")

            return BookingResult(
                success=True,
                appointment_id=appointment_id,
                patient_id=patient_id,
                doctor_id=selected_slot["doctor_id"],
                doctor_name=selected_slot["doctor_name"],
                specialty=selected_slot["specialty"],
                slot_datetime=selected_slot["slot_datetime"],
                reason=reason,
                message=f"Appointment booked with {selected_slot['doctor_name']} ({selected_slot['specialty']}) on {formatted_datetime}"
            )

        except Exception as e:
            return BookingResult(
                success=False,
                patient_id=patient_id,
                specialty=specialty,
                message="Failed to book appointment",
                error=str(e)
            )

    def book_for_gap(
        self,
        patient_id: str,
        gap_type: str,
        gap_description: str = ""
    ) -> BookingResult:
        """Book an appointment based on a detected care gap.

        Args:
            patient_id: Patient identifier
            gap_type: Type of care gap (e.g., "A1C_THRESHOLD")
            gap_description: Description of the gap for the reason

        Returns:
            BookingResult with booking details
        """
        # Determine specialty from gap type
        specialty = self.GAP_TO_SPECIALTY.get(gap_type)

        if not specialty:
            return BookingResult(
                success=False,
                patient_id=patient_id,
                message=f"Unknown gap type: {gap_type}",
                error=f"Cannot determine specialty for gap type '{gap_type}'"
            )

        # Build reason from gap
        reason = f"Care gap follow-up: {gap_type}"
        if gap_description:
            reason = f"{reason} - {gap_description}"

        return self.book_appointment(patient_id, specialty, reason)

    def get_patient_appointments(self, patient_id: str) -> list[dict]:
        """Get all appointments for a patient.

        Args:
            patient_id: Patient identifier

        Returns:
            List of appointment records
        """
        db = self._get_db()
        return db.get_patient_appointments(patient_id)

    def get_available_specialties(self) -> list[str]:
        """Get list of available specialties.

        Returns:
            List of specialty names
        """
        db = self._get_db()
        doctors = db.get_all_doctors()
        specialties = list(set(d["specialty"] for d in doctors))
        return sorted(specialties)


class VectorSearchTool:
    """Tool for searching FAISS vector indexes."""

    def __init__(self):
        """Initialize vector search tool."""
        self._patient_index = None
        self._guidelines_index = None

    def _get_patient_index(self):
        """Get patient notes index."""
        if self._patient_index is None:
            from vector_store_faiss import get_patient_index
            self._patient_index = get_patient_index()
            try:
                self._patient_index.load()
            except:
                pass
        return self._patient_index

    def _get_guidelines_index(self):
        """Get guidelines index."""
        if self._guidelines_index is None:
            from vector_store_faiss import get_guidelines_index
            self._guidelines_index = get_guidelines_index()
            try:
                self._guidelines_index.load()
            except:
                pass
        return self._guidelines_index

    def search_patients(self, query: str, top_k: int = 3) -> list[dict]:
        """Search patient notes index.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of search results
        """
        try:
            index = self._get_patient_index()
            if index.index is None:
                return []
            return index.query(query, top_k)
        except Exception as e:
            return []

    def search_guidelines(self, query: str, top_k: int = 3) -> list[dict]:
        """Search guidelines index.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of search results
        """
        try:
            index = self._get_guidelines_index()
            if index.index is None:
                return []
            return index.query(query, top_k)
        except Exception as e:
            return []


class ClinicalUtilities:
    """Clinical utility functions."""

    @staticmethod
    def calculate_age(birth_date: str) -> int:
        """Calculate age from birth date.

        Args:
            birth_date: Birth date in YYYY-MM-DD format

        Returns:
            Age in years
        """
        dob = datetime.strptime(birth_date, "%Y-%m-%d")
        today = datetime.now()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age

    @staticmethod
    def format_bp(systolic: int, diastolic: int) -> str:
        """Format blood pressure for display.

        Args:
            systolic: Systolic pressure
            diastolic: Diastolic pressure

        Returns:
            Formatted string
        """
        return f"{systolic}/{diastolic} mmHg"

    @staticmethod
    def format_a1c(a1c: float) -> str:
        """Format A1C for display.

        Args:
            a1c: A1C value

        Returns:
            Formatted string
        """
        return f"{a1c}%"

    @staticmethod
    def is_bp_elevated(systolic: int, diastolic: int) -> bool:
        """Check if blood pressure is elevated.

        Args:
            systolic: Systolic pressure
            diastolic: Diastolic pressure

        Returns:
            True if elevated
        """
        return systolic >= 140 or diastolic >= 90

    @staticmethod
    def is_a1c_above_goal(a1c: float, goal: float = 7.0) -> bool:
        """Check if A1C is above goal.

        Args:
            a1c: A1C value
            goal: Target A1C (default 7.0%)

        Returns:
            True if above goal
        """
        return a1c >= goal


# ICD-10 code lookup (simplified)
ICD10_CODES = {
    "E11": {"description": "Type 2 diabetes mellitus", "category": "Endocrine"},
    "E11.9": {"description": "Type 2 diabetes mellitus without complications", "category": "Endocrine"},
    "E11.65": {"description": "Type 2 diabetes mellitus with hyperglycemia", "category": "Endocrine"},
    "I10": {"description": "Essential (primary) hypertension", "category": "Cardiovascular"},
    "I11": {"description": "Hypertensive heart disease", "category": "Cardiovascular"},
    "N18": {"description": "Chronic kidney disease", "category": "Renal"},
    "N18.3": {"description": "Chronic kidney disease, stage 3", "category": "Renal"},
}


def lookup_icd_code(code: str) -> dict:
    """Look up an ICD-10 code.

    Args:
        code: ICD-10 code

    Returns:
        dict with code description
    """
    return ICD10_CODES.get(code, {"description": "Unknown code", "category": "Unknown"})


# Convenience functions
def search_vector_store(query: str, index_name: str, top_k: int = 5) -> list:
    """Search a FAISS vector store.

    Args:
        query: Search query
        index_name: Name of the index ("patients" or "guidelines")
        top_k: Number of results

    Returns:
        list of search results
    """
    tool = VectorSearchTool()
    if index_name == "patients":
        return tool.search_patients(query, top_k)
    elif index_name == "guidelines":
        return tool.search_guidelines(query, top_k)
    else:
        return []


def calculate_age(birth_date: str) -> int:
    """Calculate age from birth date.

    Args:
        birth_date: Birth date in YYYY-MM-DD format

    Returns:
        Age in years
    """
    return ClinicalUtilities.calculate_age(birth_date)


def book_appointment(
    patient_id: str,
    specialty: str,
    reason: str,
    db=None
) -> BookingResult:
    """Book an appointment for a patient.

    Args:
        patient_id: Patient ID
        specialty: Medical specialty
        reason: Reason for visit
        db: Optional database connection

    Returns:
        BookingResult
    """
    tool = BookingTool(db)
    return tool.book_appointment(patient_id, specialty, reason)


# Test function
def test_tools():
    """Test the tools module."""
    from care_database import get_database

    db = get_database()

    print("=" * 60)
    print("TOOLS MODULE TEST")
    print("=" * 60)

    # Test booking tool
    print("\n--- Booking Tool Test ---")
    booking_tool = BookingTool(db)

    # Get available specialties
    specialties = booking_tool.get_available_specialties()
    print(f"Available specialties: {specialties}")

    # Try to book an appointment
    result = booking_tool.book_appointment(
        patient_id="PT001",
        specialty="Endocrinology",
        reason="A1C follow-up"
    )
    print(f"\nBooking result: {result.message}")
    if result.success:
        print(f"  Appointment ID: {result.appointment_id}")
        print(f"  Doctor: {result.doctor_name}")
        print(f"  Time: {result.slot_datetime}")

    # Test gap-based booking
    result2 = booking_tool.book_for_gap(
        patient_id="PT004",
        gap_type="HTN_ACE_ARB",
        gap_description="Patient needs ACE inhibitor initiation"
    )
    print(f"\nGap-based booking: {result2.message}")

    # Test clinical utilities
    print("\n--- Clinical Utilities Test ---")
    age = ClinicalUtilities.calculate_age("1965-03-15")
    print(f"Age for DOB 1965-03-15: {age} years")
    print(f"BP 142/94 elevated: {ClinicalUtilities.is_bp_elevated(142, 94)}")
    print(f"A1C 8.2 above goal: {ClinicalUtilities.is_a1c_above_goal(8.2)}")

    # Test ICD lookup
    print("\n--- ICD-10 Lookup Test ---")
    code = lookup_icd_code("E11")
    print(f"E11: {code['description']}")


if __name__ == "__main__":
    test_tools()
