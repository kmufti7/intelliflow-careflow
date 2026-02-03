"""Database module for CareFlow - SQLite with clinical domain tables."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import json
import uuid


class CareDatabase:
    """SQLite database for CareFlow clinical data and audit logging."""

    def __init__(self, db_path: str = "data/careflow.db"):
        """Initialize the database connection.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Connect to the database and create tables."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create all domain tables."""
        cursor = self.conn.cursor()

        # Patients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                patient_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                dob TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Patient notes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_notes (
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                note_date TEXT NOT NULL,
                note_text TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)

        # Doctors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS doctors (
                id TEXT PRIMARY KEY,
                doctor_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                specialty TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Doctor slots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS doctor_slots (
                id TEXT PRIMARY KEY,
                doctor_id TEXT NOT NULL,
                slot_datetime TEXT NOT NULL,
                is_available INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id)
            )
        """)

        # Appointments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS appointments (
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                doctor_id TEXT NOT NULL,
                slot_datetime TEXT NOT NULL,
                reason TEXT,
                status TEXT DEFAULT 'scheduled',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
                FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id)
            )
        """)

        # Audit logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                agent_name TEXT NOT NULL,
                action TEXT NOT NULL,
                input_summary TEXT,
                output_summary TEXT,
                decision_reasoning TEXT,
                confidence_score REAL,
                duration_ms INTEGER DEFAULT 0,
                success INTEGER DEFAULT 1,
                error_message TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()

    # ==================== Patient Methods ====================

    def add_patient(self, patient_id: str, name: str, dob: str) -> str:
        """Add a new patient.

        Args:
            patient_id: Unique patient identifier
            name: Patient name
            dob: Date of birth (YYYY-MM-DD)

        Returns:
            The generated record ID
        """
        record_id = str(uuid.uuid4())
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO patients (id, patient_id, name, dob)
            VALUES (?, ?, ?, ?)
        """, (record_id, patient_id, name, dob))
        self.conn.commit()
        return record_id

    def get_all_patients(self) -> List[dict]:
        """Get all patients.

        Returns:
            List of patient records
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM patients ORDER BY name")
        return [dict(row) for row in cursor.fetchall()]

    def get_patient(self, patient_id: str) -> Optional[dict]:
        """Get a patient by ID.

        Args:
            patient_id: Patient identifier

        Returns:
            Patient record or None
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    # ==================== Patient Notes Methods ====================

    def add_patient_note(self, patient_id: str, note_date: str, note_text: str) -> str:
        """Add a clinic note for a patient.

        Args:
            patient_id: Patient identifier
            note_date: Date of the note (YYYY-MM-DD)
            note_text: The clinic note text

        Returns:
            The generated record ID
        """
        record_id = str(uuid.uuid4())
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO patient_notes (id, patient_id, note_date, note_text)
            VALUES (?, ?, ?, ?)
        """, (record_id, patient_id, note_date, note_text))
        self.conn.commit()
        return record_id

    def get_patient_notes(self, patient_id: str) -> List[dict]:
        """Get all notes for a patient.

        Args:
            patient_id: Patient identifier

        Returns:
            List of note records
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM patient_notes
            WHERE patient_id = ?
            ORDER BY note_date DESC
        """, (patient_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_latest_note(self, patient_id: str) -> Optional[dict]:
        """Get the most recent note for a patient.

        Args:
            patient_id: Patient identifier

        Returns:
            Most recent note or None
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM patient_notes
            WHERE patient_id = ?
            ORDER BY note_date DESC
            LIMIT 1
        """, (patient_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    # ==================== Doctor Methods ====================

    def add_doctor(self, doctor_id: str, name: str, specialty: str) -> str:
        """Add a new doctor.

        Args:
            doctor_id: Unique doctor identifier
            name: Doctor name
            specialty: Medical specialty

        Returns:
            The generated record ID
        """
        record_id = str(uuid.uuid4())
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO doctors (id, doctor_id, name, specialty)
            VALUES (?, ?, ?, ?)
        """, (record_id, doctor_id, name, specialty))
        self.conn.commit()
        return record_id

    def get_all_doctors(self) -> List[dict]:
        """Get all doctors.

        Returns:
            List of doctor records
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM doctors ORDER BY specialty, name")
        return [dict(row) for row in cursor.fetchall()]

    def get_doctors_by_specialty(self, specialty: str) -> List[dict]:
        """Get doctors by specialty.

        Args:
            specialty: Medical specialty

        Returns:
            List of matching doctor records
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM doctors
            WHERE specialty = ?
            ORDER BY name
        """, (specialty,))
        return [dict(row) for row in cursor.fetchall()]

    # ==================== Slot Methods ====================

    def add_slot(self, doctor_id: str, slot_datetime: str) -> str:
        """Add an appointment slot for a doctor.

        Args:
            doctor_id: Doctor identifier
            slot_datetime: Datetime of the slot (ISO format)

        Returns:
            The generated record ID
        """
        record_id = str(uuid.uuid4())
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO doctor_slots (id, doctor_id, slot_datetime, is_available)
            VALUES (?, ?, ?, 1)
        """, (record_id, doctor_id, slot_datetime))
        self.conn.commit()
        return record_id

    def get_available_slots(self, doctor_id: str = None) -> List[dict]:
        """Get available appointment slots.

        Args:
            doctor_id: Optional doctor filter

        Returns:
            List of available slot records
        """
        cursor = self.conn.cursor()
        if doctor_id:
            cursor.execute("""
                SELECT ds.*, d.name as doctor_name, d.specialty
                FROM doctor_slots ds
                JOIN doctors d ON ds.doctor_id = d.doctor_id
                WHERE ds.doctor_id = ? AND ds.is_available = 1
                ORDER BY ds.slot_datetime
            """, (doctor_id,))
        else:
            cursor.execute("""
                SELECT ds.*, d.name as doctor_name, d.specialty
                FROM doctor_slots ds
                JOIN doctors d ON ds.doctor_id = d.doctor_id
                WHERE ds.is_available = 1
                ORDER BY ds.slot_datetime
            """)
        return [dict(row) for row in cursor.fetchall()]

    # ==================== Appointment Methods ====================

    def create_appointment(self, patient_id: str, doctor_id: str,
                          slot_datetime: str, reason: str) -> str:
        """Create an appointment and mark slot as unavailable.

        Args:
            patient_id: Patient identifier
            doctor_id: Doctor identifier
            slot_datetime: Datetime of the appointment
            reason: Reason for visit

        Returns:
            The generated appointment ID
        """
        record_id = str(uuid.uuid4())
        cursor = self.conn.cursor()

        # Create appointment
        cursor.execute("""
            INSERT INTO appointments (id, patient_id, doctor_id, slot_datetime, reason, status)
            VALUES (?, ?, ?, ?, ?, 'scheduled')
        """, (record_id, patient_id, doctor_id, slot_datetime, reason))

        # Mark slot as unavailable
        cursor.execute("""
            UPDATE doctor_slots
            SET is_available = 0
            WHERE doctor_id = ? AND slot_datetime = ?
        """, (doctor_id, slot_datetime))

        self.conn.commit()
        return record_id

    def get_patient_appointments(self, patient_id: str) -> List[dict]:
        """Get appointments for a patient.

        Args:
            patient_id: Patient identifier

        Returns:
            List of appointment records
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT a.*, d.name as doctor_name, d.specialty
            FROM appointments a
            JOIN doctors d ON a.doctor_id = d.doctor_id
            WHERE a.patient_id = ?
            ORDER BY a.slot_datetime
        """, (patient_id,))
        return [dict(row) for row in cursor.fetchall()]

    # ==================== Audit Log Methods ====================

    def log_action(
        self,
        agent_name: str,
        action: str,
        input_summary: str = "",
        output_summary: str = "",
        decision_reasoning: str = None,
        confidence_score: float = None,
        duration_ms: int = 0,
        success: bool = True,
        error_message: str = None,
        session_id: str = None,
        metadata: dict = None,
    ) -> str:
        """Log an action to the audit trail.

        Args:
            agent_name: Name of the agent performing the action
            action: Description of the action
            input_summary: Summary of input data
            output_summary: Summary of output data
            decision_reasoning: Reasoning for the decision
            confidence_score: Confidence score (0-1)
            duration_ms: Duration in milliseconds
            success: Whether the action succeeded
            error_message: Error message if failed
            session_id: Session identifier
            metadata: Additional metadata

        Returns:
            The generated log ID
        """
        log_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO audit_logs (
                id, timestamp, session_id, agent_name, action,
                input_summary, output_summary, decision_reasoning,
                confidence_score, duration_ms, success, error_message, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log_id,
            timestamp,
            session_id,
            agent_name,
            action,
            input_summary[:200] if input_summary else None,
            output_summary,
            decision_reasoning,
            confidence_score,
            duration_ms,
            1 if success else 0,
            error_message,
            json.dumps(metadata) if metadata else None,
        ))
        self.conn.commit()
        return log_id

    def get_recent_logs(self, limit: int = 100) -> List[dict]:
        """Get recent audit logs.

        Args:
            limit: Maximum number of logs to return

        Returns:
            List of log entries
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM audit_logs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

    # ==================== Utility Methods ====================

    def is_empty(self) -> bool:
        """Check if database has no patients (needs seeding).

        Returns:
            True if no patients exist
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM patients")
        count = cursor.fetchone()[0]
        return count == 0

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


# Singleton instance
_db_instance: Optional[CareDatabase] = None


def get_database(db_path: str = "data/careflow.db") -> CareDatabase:
    """Get or create the database instance.

    Args:
        db_path: Path to the database file

    Returns:
        CareDatabase instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = CareDatabase(db_path)
        _db_instance.connect()
    return _db_instance
