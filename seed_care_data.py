"""Seed data script for CareFlow - populates sample clinical data."""

from datetime import datetime, timedelta
from care_database import get_database


def seed_patients(db):
    """Seed 5 patients with specific clinical profiles for gap analysis testing.

    Patient profiles designed to test care gap detection:
    - Patient 1: A1C=8.2, has HTN, NO Lisinopril (both gaps)
    - Patient 2: A1C=7.4, has HTN, has Lisinopril (no gaps)
    - Patient 3: A1C=9.1, no HTN (only A1C gap)
    - Patient 4: A1C=6.8, has HTN, NO Lisinopril (only HTN gap)
    - Patient 5: A1C=7.0, no HTN, has Lisinopril (no gaps - Lisinopril for other reason)
    """
    patients = [
        {
            "patient_id": "PT001",
            "name": "Maria Garcia",
            "dob": "1965-03-15",
            "note": """Subjective: Patient reports fatigue, increased thirst, and occasional headaches over the past month.

Objective:
- Vitals: BP 142/94 mmHg, HR 78 bpm, Temp 98.4°F, Weight 187 lbs
- Labs: A1C 8.2%, Fasting glucose 165 mg/dL, Creatinine 1.1 mg/dL

Assessment:
- Type 2 Diabetes Mellitus - suboptimally controlled
- Essential Hypertension - not at goal

Current Medications:
- Metformin 1000mg BID
- Amlodipine 5mg daily

Plan:
- Increase Metformin to 1000mg BID (already at max)
- Consider adding second diabetes agent
- Recheck A1C in 3 months
- Follow up in 3 months for BP and diabetes management"""
        },
        {
            "patient_id": "PT002",
            "name": "James Wilson",
            "dob": "1958-07-22",
            "note": """Subjective: Patient here for routine diabetes and hypertension follow-up. Reports feeling well, checking blood sugars regularly, averaging 130-150 mg/dL fasting.

Objective:
- Vitals: BP 128/82 mmHg, HR 72 bpm, Temp 98.2°F, Weight 195 lbs
- Labs: A1C 7.4%, Fasting glucose 138 mg/dL, Creatinine 0.9 mg/dL, K+ 4.2

Assessment:
- Type 2 Diabetes Mellitus - reasonably controlled
- Essential Hypertension - controlled on current regimen

Current Medications:
- Metformin 1000mg BID
- Lisinopril 20mg daily
- Atorvastatin 40mg daily

Plan:
- Continue current medications
- Annual diabetic eye exam - patient to schedule
- Recheck A1C in 3 months
- Follow up in 6 months"""
        },
        {
            "patient_id": "PT003",
            "name": "Sarah Chen",
            "dob": "1972-11-08",
            "note": """Subjective: Patient reports increased fatigue, frequent urination, and blurry vision for 2 weeks. No chest pain, no edema, no shortness of breath.

Objective:
- Vitals: BP 118/76 mmHg, HR 82 bpm, Temp 98.6°F, Weight 165 lbs
- Labs: A1C 9.1%, Fasting glucose 212 mg/dL, Creatinine 0.8 mg/dL

Assessment:
- Type 2 Diabetes Mellitus - poorly controlled, new diagnosis 6 months ago
- No evidence of hypertension

Current Medications:
- Metformin 500mg BID (started 6 months ago)

Plan:
- Increase Metformin to 1000mg BID
- Start Glipizide 5mg daily
- Diabetes education referral
- Recheck A1C in 3 months
- Follow up in 1 month to assess medication tolerance"""
        },
        {
            "patient_id": "PT004",
            "name": "Robert Thompson",
            "dob": "1960-04-30",
            "note": """Subjective: Patient presents for hypertension follow-up. Reports occasional morning headaches. Denies chest pain, shortness of breath, or visual changes. Monitors BP at home, averaging 145/92.

Objective:
- Vitals: BP 148/94 mmHg, HR 76 bpm, Temp 98.4°F, Weight 210 lbs
- Labs: A1C 6.8%, Fasting glucose 118 mg/dL, Creatinine 1.0 mg/dL, K+ 4.5

Assessment:
- Essential Hypertension - not at goal despite medication
- Type 2 Diabetes Mellitus - well controlled

Current Medications:
- Metformin 500mg BID
- Hydrochlorothiazide 25mg daily
- Amlodipine 10mg daily

Plan:
- Blood pressure not at goal on current regimen
- Consider adding ACE inhibitor for additional BP control and renal protection
- Low sodium diet counseling
- Recheck BP in 2 weeks
- Follow up in 1 month"""
        },
        {
            "patient_id": "PT005",
            "name": "Linda Martinez",
            "dob": "1968-09-12",
            "note": """Subjective: Patient here for chronic kidney disease and diabetes follow-up. Reports feeling well overall. Good medication compliance. No swelling, no urinary changes.

Objective:
- Vitals: BP 122/78 mmHg, HR 68 bpm, Temp 98.2°F, Weight 155 lbs
- Labs: A1C 7.0%, Fasting glucose 128 mg/dL, Creatinine 1.4 mg/dL, eGFR 52, K+ 4.8

Assessment:
- Type 2 Diabetes Mellitus - at goal
- Chronic Kidney Disease Stage 3a - stable
- No hypertension

Current Medications:
- Metformin 500mg BID (reduced dose for CKD)
- Lisinopril 10mg daily (for renal protection)
- Atorvastatin 20mg daily

Plan:
- Continue current medications
- Lisinopril providing renal protection - continue
- Monitor potassium with ACE inhibitor use
- Nephrology referral if eGFR drops below 45
- Follow up in 3 months"""
        }
    ]

    for p in patients:
        db.add_patient(p["patient_id"], p["name"], p["dob"])
        # Add note dated within last month
        note_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        db.add_patient_note(p["patient_id"], note_date, p["note"])

    print(f"Seeded {len(patients)} patients with clinic notes")


def seed_doctors(db):
    """Seed 10 doctors with various specialties."""
    doctors = [
        {"doctor_id": "DR001", "name": "Dr. Emily Watson", "specialty": "Endocrinology"},
        {"doctor_id": "DR002", "name": "Dr. Michael Brown", "specialty": "Endocrinology"},
        {"doctor_id": "DR003", "name": "Dr. Jennifer Lee", "specialty": "Cardiology"},
        {"doctor_id": "DR004", "name": "Dr. David Kim", "specialty": "Cardiology"},
        {"doctor_id": "DR005", "name": "Dr. Susan Patel", "specialty": "Nephrology"},
        {"doctor_id": "DR006", "name": "Dr. Richard Chen", "specialty": "Nephrology"},
        {"doctor_id": "DR007", "name": "Dr. Amanda Foster", "specialty": "Internal Medicine"},
        {"doctor_id": "DR008", "name": "Dr. Christopher Jones", "specialty": "Internal Medicine"},
        {"doctor_id": "DR009", "name": "Dr. Patricia Williams", "specialty": "Family Medicine"},
        {"doctor_id": "DR010", "name": "Dr. Thomas Anderson", "specialty": "Family Medicine"},
    ]

    for d in doctors:
        db.add_doctor(d["doctor_id"], d["name"], d["specialty"])

    print(f"Seeded {len(doctors)} doctors")


def seed_appointment_slots(db):
    """Seed 30 appointment slots spread across next 2 weeks."""
    doctors = db.get_all_doctors()

    # Generate slots for next 2 weeks
    base_date = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    slot_count = 0

    for day_offset in range(14):  # 2 weeks
        current_date = base_date + timedelta(days=day_offset)

        # Skip weekends
        if current_date.weekday() >= 5:
            continue

        # Create 3-4 slots per day, rotating through doctors
        for hour_offset in [0, 2, 4]:  # 9am, 11am, 1pm
            slot_time = current_date + timedelta(hours=hour_offset)
            doctor = doctors[slot_count % len(doctors)]
            db.add_slot(doctor["doctor_id"], slot_time.isoformat())
            slot_count += 1

            if slot_count >= 30:
                break

        if slot_count >= 30:
            break

    print(f"Seeded {slot_count} appointment slots")


def seed_all():
    """Seed all data if database is empty."""
    db = get_database()

    if not db.is_empty():
        print("Database already contains data. Skipping seed.")
        return False

    print("Seeding CareFlow database...")
    seed_patients(db)
    seed_doctors(db)
    seed_appointment_slots(db)
    print("Seed complete!")
    return True


if __name__ == "__main__":
    seed_all()
