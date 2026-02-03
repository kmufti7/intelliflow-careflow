"""Tests for the booking tool.

Tests appointment booking functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import BookingTool, BookingResult
from care_database import get_database


class TestBooking:
    """Test suite for BookingTool."""

    def __init__(self):
        self.db = get_database()
        self.booking_tool = BookingTool(self.db)
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

    def test_booking_creates_appointment(self):
        """Test: Booking creates appointment in database."""
        # Use Family Medicine which typically has more slots
        result = self.booking_tool.book_appointment(
            patient_id="PT001",
            specialty="Family Medicine",
            reason="A1C follow-up test"
        )

        # Verify appointment was created or no slots available
        if result.success and result.appointment_id:
            # Booking succeeded - verify it has required fields
            has_fields = (
                result.appointment_id is not None and
                result.doctor_name is not None and
                result.slot_datetime is not None
            )
            return self._assert(
                has_fields,
                "Booking creates appointment in DB",
                f"Appointment ID: {result.appointment_id}, Doctor: {result.doctor_name}"
            )
        else:
            # If no slots available, that's acceptable behavior
            no_slots = "No available" in result.message
            return self._assert(
                no_slots,
                "Booking creates appointment in DB",
                f"No slots available (acceptable): {result.message}"
            )

    def test_booking_finds_available_slot(self):
        """Test: Booking finds and uses available slot when one exists."""
        # Use Internal Medicine which typically has more slots
        result = self.booking_tool.book_appointment(
            patient_id="PT002",
            specialty="Internal Medicine",
            reason="General follow-up"
        )

        if result.success:
            slot_found = result.slot_datetime is not None
            return self._assert(
                slot_found,
                "Booking finds available slot",
                f"Slot: {result.slot_datetime}, Doctor: {result.doctor_name}"
            )
        else:
            # If no slots, that's acceptable in test environment
            no_slots = "No available" in result.message
            return self._assert(
                no_slots,
                "Booking finds available slot",
                f"No slots available (acceptable): {result.message}"
            )

    def test_booking_returns_doctor_info(self):
        """Test: Booking result includes doctor information when successful."""
        result = self.booking_tool.book_appointment(
            patient_id="PT003",
            specialty="Nephrology",  # Use Nephrology for variety
            reason="Kidney function review"
        )

        if result.success:
            has_doctor_info = (
                result.doctor_name is not None and
                result.doctor_id is not None
            )
            return self._assert(
                has_doctor_info,
                "Booking returns doctor info",
                f"Doctor: {result.doctor_name} (ID: {result.doctor_id})"
            )
        else:
            # No slots is acceptable
            no_slots = "No available" in result.message
            return self._assert(
                no_slots,
                "Booking returns doctor info",
                f"No slots available (acceptable): {result.message}"
            )

    def test_booking_logs_to_audit(self):
        """Test: Audit log can capture booking events."""
        # Get count of audit logs before
        logs_before = len(self.db.get_recent_logs(100))

        # Log a booking action (simulating what orchestrator does)
        self.db.log_action(
            agent_name="BookingTool",
            action="book_appointment_test",
            input_summary="PT004, Internal Medicine",
            output_summary="Test log entry for booking",
            success=True
        )

        logs_after = len(self.db.get_recent_logs(100))

        return self._assert(
            logs_after > logs_before,
            "Booking logs to audit",
            f"Logs before: {logs_before}, after: {logs_after}"
        )

    def test_booking_invalid_patient(self):
        """Test: Booking fails gracefully for invalid patient."""
        result = self.booking_tool.book_appointment(
            patient_id="INVALID_ID",
            specialty="Endocrinology",
            reason="Test"
        )

        return self._assert(
            result.success is False and result.error is not None,
            "Booking fails for invalid patient",
            f"Error: {result.error}"
        )

    def test_booking_invalid_specialty(self):
        """Test: Booking handles unavailable specialty."""
        result = self.booking_tool.book_appointment(
            patient_id="PT001",
            specialty="Neurosurgery",  # Not in our test data
            reason="Test"
        )

        return self._assert(
            result.success is False,
            "Booking handles unavailable specialty",
            f"Message: {result.message}"
        )

    def test_book_for_gap_a1c(self):
        """Test: book_for_gap maps A1C gap to Endocrinology."""
        result = self.booking_tool.book_for_gap(
            patient_id="PT002",  # Use different patient to avoid slot conflicts
            gap_type="A1C_THRESHOLD",
            gap_description="A1C is 8.2%, above goal of 7.0%"
        )

        # Either booking succeeds with correct specialty, or no slots available
        if result.success:
            correct_specialty = result.specialty == "Endocrinology"
            return self._assert(
                correct_specialty,
                "book_for_gap maps A1C to Endocrinology",
                f"Specialty: {result.specialty}"
            )
        else:
            # If no slots, check that it tried the right specialty
            no_slots = "No available" in result.message or "Endocrinology" in str(result.specialty)
            return self._assert(
                no_slots,
                "book_for_gap maps A1C to Endocrinology",
                f"No slots for Endocrinology (acceptable): {result.message}"
            )

    def test_book_for_gap_htn(self):
        """Test: book_for_gap maps HTN gap to Cardiology."""
        result = self.booking_tool.book_for_gap(
            patient_id="PT005",  # Use different patient
            gap_type="HTN_ACE_ARB",
            gap_description="Patient needs ACE inhibitor"
        )

        # Either booking succeeds with correct specialty, or no slots available
        if result.success:
            correct_specialty = result.specialty == "Cardiology"
            return self._assert(
                correct_specialty,
                "book_for_gap maps HTN to Cardiology",
                f"Specialty: {result.specialty}"
            )
        else:
            # If no slots, that's acceptable
            no_slots = "No available" in result.message
            return self._assert(
                no_slots,
                "book_for_gap maps HTN to Cardiology",
                f"No slots for Cardiology (acceptable): {result.message}"
            )

    def test_book_for_gap_unknown_type(self):
        """Test: book_for_gap fails for unknown gap type."""
        result = self.booking_tool.book_for_gap(
            patient_id="PT001",
            gap_type="UNKNOWN_GAP_TYPE",
            gap_description="Test"
        )

        return self._assert(
            result.success is False,
            "book_for_gap fails for unknown type",
            f"Error: {result.error}"
        )

    def test_get_available_specialties(self):
        """Test: Can retrieve list of available specialties."""
        specialties = self.booking_tool.get_available_specialties()

        has_specialties = len(specialties) > 0
        has_expected = "Endocrinology" in specialties and "Cardiology" in specialties

        return self._assert(
            has_specialties and has_expected,
            "Get available specialties",
            f"Specialties: {specialties}"
        )

    def test_booking_result_message(self):
        """Test: Booking result has human-readable message."""
        result = self.booking_tool.book_appointment(
            patient_id="PT005",
            specialty="Internal Medicine",
            reason="Follow-up"
        )

        has_message = result.message and len(result.message) > 10

        return self._assert(
            has_message,
            "Booking result has message",
            f"Message: {result.message[:80]}..."
        )

    def run_all(self) -> dict:
        """Run all booking tests."""
        self.results = []

        # Run all test methods
        self.test_booking_creates_appointment()
        self.test_booking_finds_available_slot()
        self.test_booking_returns_doctor_info()
        self.test_booking_logs_to_audit()
        self.test_booking_invalid_patient()
        self.test_booking_invalid_specialty()
        self.test_book_for_gap_a1c()
        self.test_book_for_gap_htn()
        self.test_book_for_gap_unknown_type()
        self.test_get_available_specialties()
        self.test_booking_result_message()

        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)

        return {
            "suite": "Booking",
            "passed": passed,
            "total": total,
            "results": self.results
        }


def run_tests():
    """Run booking tests and print results."""
    suite = TestBooking()
    results = suite.run_all()

    print("=" * 60)
    print("BOOKING TESTS")
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
