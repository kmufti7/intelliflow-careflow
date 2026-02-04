"""Streamlit app for IntelliFlow CareFlow."""

import streamlit as st
from datetime import datetime
import time

from intelliflow_core.governance_ui import (
    init_governance_state,
    add_governance_log as _add_governance_log,
)
from intelliflow_core.helpers import format_timestamp_short

from care_database import get_database
from seed_care_data import seed_all
from extraction import PatientFactExtractor, ExtractedFacts
from reasoning_engine import ReasoningEngine, ReasoningResult, GapResult
from care_orchestrator import CareOrchestrator

# Page configuration
st.set_page_config(
    page_title="IntelliFlow OS: CareFlow",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
    }
    .agent-message {
        background-color: #F5F5F5;
        border-left: 4px solid #2196F3;
    }
    .clinic-note {
        background-color: #FFF8E1;
        border-left: 4px solid #FFC107;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        max-height: 250px;
        overflow-y: auto;
    }
    .patient-info {
        background-color: #E3F2FD;
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .extracted-facts {
        background-color: #F3E5F5;
        border-left: 4px solid #9C27B0;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin-top: 1rem;
    }
    .fact-item {
        margin-bottom: 0.5rem;
        padding: 0.5rem;
        background-color: white;
        border-radius: 5px;
    }
    .fact-label {
        font-weight: 600;
        color: #6A1B9A;
        font-size: 0.85rem;
    }
    .fact-value {
        font-family: 'Consolas', 'Monaco', monospace;
        color: #333;
    }
    .extraction-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .badge-regex {
        background-color: #C8E6C9;
        color: #2E7D32;
    }
    .badge-llm {
        background-color: #FFECB3;
        color: #FF8F00;
    }
    .badge-hybrid {
        background-color: #B3E5FC;
        color: #0277BD;
    }
    .medication-list, .diagnosis-list {
        list-style-type: none;
        padding-left: 0;
        margin: 0;
    }
    .medication-list li, .diagnosis-list li {
        padding: 0.25rem 0;
        border-bottom: 1px solid #eee;
    }
    .medication-list li:last-child, .diagnosis-list li:last-child {
        border-bottom: none;
    }
    .care-gaps-section {
        background-color: #FFEBEE;
        border-left: 4px solid #F44336;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin-top: 1rem;
    }
    .care-gaps-clear {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin-top: 1rem;
    }
    .gap-item {
        background-color: white;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #F44336;
    }
    .gap-item.severity-high {
        border-left-color: #D32F2F;
        background-color: #FFCDD2;
    }
    .gap-item.severity-moderate {
        border-left-color: #FF9800;
        background-color: #FFE0B2;
    }
    .gap-item.severity-low {
        border-left-color: #FFC107;
        background-color: #FFF8E1;
    }
    .gap-header {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .gap-type {
        font-weight: 700;
        font-size: 0.9rem;
        color: #333;
    }
    .severity-badge {
        display: inline-block;
        padding: 0.15rem 0.4rem;
        border-radius: 3px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-left: 0.5rem;
        text-transform: uppercase;
    }
    .severity-high {
        background-color: #D32F2F;
        color: white;
    }
    .severity-moderate {
        background-color: #FF9800;
        color: white;
    }
    .severity-low {
        background-color: #FFC107;
        color: #333;
    }
    .gap-comparison {
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.8rem;
        color: #666;
        margin-bottom: 0.25rem;
    }
    .gap-therefore {
        font-size: 0.85rem;
        color: #333;
        margin-bottom: 0.25rem;
    }
    .gap-recommendation {
        font-size: 0.85rem;
        color: #1565C0;
        font-style: italic;
    }
    .gap-guideline {
        font-size: 0.75rem;
        color: #888;
        margin-top: 0.25rem;
    }
    .closed-gap {
        background-color: #C8E6C9;
        border-radius: 5px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "db" not in st.session_state:
        st.session_state.db = None
    if "selected_patient_id" not in st.session_state:
        st.session_state.selected_patient_id = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # Use shared governance state from intelliflow_core
    init_governance_state()
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "session_cost" not in st.session_state:
        st.session_state.session_cost = 0.0
    if "extracted_facts" not in st.session_state:
        st.session_state.extracted_facts = None
    if "extractor" not in st.session_state:
        st.session_state.extractor = PatientFactExtractor()
    if "reasoning_engine" not in st.session_state:
        st.session_state.reasoning_engine = ReasoningEngine()
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = CareOrchestrator()


def initialize_database():
    """Initialize database and seed if empty."""
    db = get_database()

    # Auto-seed if empty
    if db.is_empty():
        seed_all()
        add_governance_log("System", "Database seeded", True, "5 patients, 10 doctors, 30 slots")

    return db


def add_governance_log(component: str, action: str, success: bool, details: str = ""):
    """Add an entry to the governance log using shared intelliflow_core component."""
    # Use shared governance logging from intelliflow_core
    _add_governance_log(component, action, success, details if details else None)

    # Also log to database (CareFlow-specific backend persistence)
    if st.session_state.db:
        st.session_state.db.log_action(
            agent_name=component,
            action=action,
            output_summary=details,
            success=success
        )


def render_governance_log():
    """Render the governance log panel using GovernanceLogEntry from intelliflow_core."""
    if not st.session_state.governance_logs:
        st.info("No activity yet. Select a patient to begin.")
        return

    # Build plain text log entries (reversed for newest-first display)
    log_lines = []
    for entry in reversed(st.session_state.governance_logs):
        status = "OK" if entry.success else "ERROR"
        timestamp = format_timestamp_short(entry.timestamp)
        details = f' - {entry.details}' if entry.details else ""
        line = f'{timestamp} [{status:5}] [{entry.component}] {entry.action}{details}'
        log_lines.append(line)

    st.code("\n".join(log_lines), language=None)


def extract_patient_facts(note_text: str) -> ExtractedFacts:
    """Extract facts from patient note and log the extraction."""
    start_time = time.time()

    extractor = st.session_state.extractor
    facts = extractor.extract(note_text)

    duration_ms = int((time.time() - start_time) * 1000)

    # Log extraction to governance log
    method = facts.extraction_method
    add_governance_log(
        "Extractor",
        f"extract_facts ({method})",
        facts.is_complete(),
        f"A1C={facts.a1c}, BP={facts.blood_pressure}, {len(facts.diagnoses)} dx, {len(facts.medications)} meds ({duration_ms}ms)"
    )

    return facts


def evaluate_care_gaps(facts: ExtractedFacts, patient_id: str) -> ReasoningResult:
    """Evaluate care gaps and log the reasoning."""
    start_time = time.time()

    engine = st.session_state.reasoning_engine
    result = engine.evaluate_patient(facts, patient_id)

    duration_ms = int((time.time() - start_time) * 1000)

    # Log reasoning to governance log
    detected_gaps = [g for g in result.gaps if g.gap_detected]
    gap_types = [g.gap_type for g in detected_gaps]
    severities = [g.severity for g in detected_gaps]

    add_governance_log(
        "ReasoningEngine",
        "evaluate_gaps",
        True,
        f"{result.gaps_found} gaps found, {result.gaps_closed} closed ({duration_ms}ms)"
    )

    # Log each detected gap
    for gap in detected_gaps:
        add_governance_log(
            "ReasoningEngine",
            f"gap_detected: {gap.gap_type}",
            True,
            f"[{gap.severity.upper()}] {gap.comparison}"
        )

    return result


def render_extracted_facts(facts: ExtractedFacts):
    """Render the extracted facts panel."""
    # Determine badge class
    method = facts.extraction_method
    if method == "regex":
        badge_class = "badge-regex"
        badge_text = "REGEX"
    elif method == "llm":
        badge_class = "badge-llm"
        badge_text = "LLM"
    else:
        badge_class = "badge-hybrid"
        badge_text = "REGEX+LLM"

    st.markdown(f"""
    <div class="extracted-facts">
        <div style="margin-bottom: 0.75rem;">
            <strong style="font-size: 1rem; color: #6A1B9A;">Extracted Facts</strong>
            <span class="extraction-badge {badge_class}">{badge_text}</span>
            <span style="font-size: 0.75rem; color: #888; margin-left: 0.5rem;">
                Confidence: {facts.confidence:.0%}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Display facts in columns
    col1, col2 = st.columns(2)

    with col1:
        # A1C
        a1c_display = f"{facts.a1c}%" if facts.a1c else "Not found"
        a1c_status = ""
        if facts.a1c:
            if facts.a1c >= 7.0:
                a1c_status = " (Above Goal)"
            else:
                a1c_status = " (At Goal)"

        st.markdown(f"""
        <div class="fact-item">
            <div class="fact-label">A1C</div>
            <div class="fact-value">{a1c_display}{a1c_status}</div>
        </div>
        """, unsafe_allow_html=True)

        # Blood Pressure
        if facts.blood_pressure:
            bp_display = f"{facts.blood_pressure['systolic']}/{facts.blood_pressure['diastolic']} mmHg"
            systolic = facts.blood_pressure['systolic']
            diastolic = facts.blood_pressure['diastolic']
            if systolic >= 140 or diastolic >= 90:
                bp_status = " (Elevated)"
            else:
                bp_status = " (Normal)"
        else:
            bp_display = "Not found"
            bp_status = ""

        st.markdown(f"""
        <div class="fact-item">
            <div class="fact-label">Blood Pressure</div>
            <div class="fact-value">{bp_display}{bp_status}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Diagnoses
        if facts.diagnoses:
            dx_list = "".join([f"<li>{dx}</li>" for dx in facts.diagnoses])
            st.markdown(f"""
            <div class="fact-item">
                <div class="fact-label">Diagnoses ({len(facts.diagnoses)})</div>
                <ul class="diagnosis-list">{dx_list}</ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="fact-item">
                <div class="fact-label">Diagnoses</div>
                <div class="fact-value">Not found</div>
            </div>
            """, unsafe_allow_html=True)

        # Medications
        if facts.medications:
            med_list = "".join([f"<li>{med}</li>" for med in facts.medications])
            st.markdown(f"""
            <div class="fact-item">
                <div class="fact-label">Medications ({len(facts.medications)})</div>
                <ul class="medication-list">{med_list}</ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="fact-item">
                <div class="fact-label">Medications</div>
                <div class="fact-value">Not found</div>
            </div>
            """, unsafe_allow_html=True)


def render_care_gaps(result: ReasoningResult):
    """Render the care gaps analysis panel."""
    detected = [g for g in result.gaps if g.gap_detected]
    closed = [g for g in result.gaps if not g.gap_detected]

    if detected:
        # Has gaps
        section_class = "care-gaps-section"
        title_color = "#C62828"
        title = f"Care Gaps Identified ({len(detected)})"
    else:
        # All clear
        section_class = "care-gaps-clear"
        title_color = "#2E7D32"
        title = "No Care Gaps"

    st.markdown(f"""
    <div class="{section_class}">
        <div style="margin-bottom: 0.75rem;">
            <strong style="font-size: 1rem; color: {title_color};">{title}</strong>
            <span style="font-size: 0.75rem; color: #888; margin-left: 0.5rem;">
                Status: {result.overall_status.replace('_', ' ').title()}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show detected gaps
    for gap in detected:
        severity_class = f"severity-{gap.severity}"
        st.markdown(f"""
        <div class="gap-item {severity_class}">
            <div class="gap-header">
                <span class="gap-type">{gap.gap_type.replace('_', ' ')}</span>
                <span class="severity-badge {severity_class}">{gap.severity}</span>
            </div>
            <div class="gap-comparison">{gap.comparison}</div>
            <div class="gap-therefore">{gap.therefore}</div>
            <div class="gap-recommendation">{gap.recommendation}</div>
            <div class="gap-guideline">Guideline: {gap.guideline_id}</div>
        </div>
        """, unsafe_allow_html=True)

    # Show closed gaps in collapsed section
    if closed:
        with st.expander(f"Gaps Closed ({len(closed)})", expanded=False):
            for gap in closed:
                st.markdown(f"""
                <div class="closed-gap">
                    <strong>{gap.gap_type.replace('_', ' ')}</strong>: {gap.therefore}
                </div>
                """, unsafe_allow_html=True)


def render_patient_note(patient_id: str):
    """Render the clinic note for the selected patient."""
    db = st.session_state.db
    patient = db.get_patient(patient_id)
    note = db.get_latest_note(patient_id)

    if patient and note:
        # Calculate age
        dob = datetime.strptime(patient["dob"], "%Y-%m-%d")
        age = (datetime.now() - dob).days // 365

        st.markdown(f"""
        <div class="patient-info">
            <strong>{patient['name']}</strong> | DOB: {patient['dob']} (Age {age}) | ID: {patient['patient_id']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Latest Clinic Note** " + f"({note['note_date']})")
        st.markdown(f'<div class="clinic-note">{note["note_text"]}</div>', unsafe_allow_html=True)

        # Extract facts if not already done for this patient
        facts_key = f"facts_{patient_id}"
        gaps_key = f"gaps_{patient_id}"

        if facts_key not in st.session_state or st.session_state.get("last_patient_id") != patient_id:
            st.session_state[facts_key] = extract_patient_facts(note["note_text"])
            st.session_state[gaps_key] = evaluate_care_gaps(
                st.session_state[facts_key],
                patient_id
            )
            st.session_state["last_patient_id"] = patient_id

        # Render extracted facts
        st.markdown("---")
        render_extracted_facts(st.session_state[facts_key])

        # Render care gaps
        st.markdown("---")
        render_care_gaps(st.session_state[gaps_key])

    else:
        st.warning("No clinic notes found for this patient.")


def main():
    """Main Streamlit app."""
    init_session_state()

    # Initialize database
    if not st.session_state.initialized:
        st.session_state.db = initialize_database()
        st.session_state.initialized = True
        add_governance_log("System", "Initialized", True)

    db = st.session_state.db

    # Sidebar
    with st.sidebar:
        st.markdown("### Settings")

        # Patient selector
        st.markdown("#### Select Patient")
        patients = db.get_all_patients()

        patient_options = {p["patient_id"]: f"{p['name']} ({p['patient_id']})" for p in patients}

        if patients:
            # Create selection with "None" option
            selected = st.selectbox(
                "Patient",
                options=[""] + list(patient_options.keys()),
                format_func=lambda x: "-- Select a patient --" if x == "" else patient_options.get(x, x),
                key="patient_selector"
            )

            # Handle patient selection change
            if selected and selected != st.session_state.selected_patient_id:
                st.session_state.selected_patient_id = selected
                patient_name = patient_options[selected]
                add_governance_log("UI", "patient_selected", True, patient_name)
                st.session_state.chat_history = []  # Clear chat for new patient

            elif not selected:
                st.session_state.selected_patient_id = None
        else:
            st.warning("No patients in database")

        st.markdown("---")
        st.markdown("#### Patient Summary")
        if st.session_state.selected_patient_id:
            patient = db.get_patient(st.session_state.selected_patient_id)
            if patient:
                st.markdown(f"**Name:** {patient['name']}")
                st.markdown(f"**DOB:** {patient['dob']}")
                st.markdown(f"**ID:** {patient['patient_id']}")

                # Show quick facts if available
                facts_key = f"facts_{st.session_state.selected_patient_id}"
                gaps_key = f"gaps_{st.session_state.selected_patient_id}"

                if facts_key in st.session_state:
                    facts = st.session_state[facts_key]
                    st.markdown("---")
                    st.markdown("**Quick Facts**")
                    if facts.a1c:
                        st.markdown(f"A1C: **{facts.a1c}%**")
                    if facts.blood_pressure:
                        st.markdown(f"BP: **{facts.blood_pressure['systolic']}/{facts.blood_pressure['diastolic']}**")

                if gaps_key in st.session_state:
                    result = st.session_state[gaps_key]
                    st.markdown("---")
                    st.markdown("**Gap Status**")
                    if result.gaps_found > 0:
                        st.error(f"{result.gaps_found} gap(s) identified")
                    else:
                        st.success("No gaps")
        else:
            st.info("Select a patient to view details")

    # Header
    st.markdown('<h1 class="main-title">IntelliFlow OS: CareFlow Module</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Clinical Gap Analysis Engine</p>', unsafe_allow_html=True)

    # Metrics bar
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.total_tokens:,}</div>
            <div class="metric-label">Total Tokens</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${st.session_state.session_cost:.6f}</div>
            <div class="metric-label">Session Cost</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main content - two columns
    left_col, right_col = st.columns([6, 4])

    # Left column - Patient Note + Extracted Facts + Care Gaps + Chat interface
    with left_col:
        # Show patient clinic note if selected
        if st.session_state.selected_patient_id:
            st.markdown("### Patient Analysis")
            render_patient_note(st.session_state.selected_patient_id)
            st.markdown("---")

        st.markdown("### Chat Interface")

        chat_container = st.container()

        with chat_container:
            if st.session_state.chat_history:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>User</strong>
                            <p style="margin: 0.5rem 0 0 0;">{msg['content']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message agent-message">
                            <strong>CareFlow Agent</strong>
                            <p style="margin: 0.5rem 0 0 0;">{msg['content']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                if st.session_state.selected_patient_id:
                    st.markdown("""
                    <div style="text-align: center; color: #888; padding: 2rem; background: #f9f9f9; border-radius: 10px;">
                        <p style="font-size: 1.1rem;">Patient Analyzed</p>
                        <p>Care gaps have been identified above. Ask questions about:</p>
                        <ul style="text-align: left; display: inline-block;">
                            <li>"Explain the A1C gap in more detail"</li>
                            <li>"Why should this patient be on an ACE inhibitor?"</li>
                            <li>"What are the next steps for this patient?"</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="text-align: center; color: #888; padding: 2rem; background: #f9f9f9; border-radius: 10px;">
                        <p style="font-size: 1.1rem;">Welcome to IntelliFlow CareFlow</p>
                        <p>Select a patient from the sidebar to begin clinical gap analysis.</p>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Only show input if patient is selected
        if st.session_state.selected_patient_id:
            with st.form(key="message_form", clear_on_submit=True):
                user_input = st.text_area(
                    "Message",
                    placeholder="Ask about care gaps for this patient...",
                    height=100,
                    label_visibility="collapsed",
                )

                submit_button = st.form_submit_button("Ask CareFlow")

                if submit_button and user_input.strip():
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_input.strip(),
                    })
                    add_governance_log("System", "Query received", True, f"Length: {len(user_input)}")

                    # Process query through orchestrator
                    patient_id = st.session_state.selected_patient_id
                    facts_key = f"facts_{patient_id}"
                    gaps_key = f"gaps_{patient_id}"

                    # Build context from session state
                    patient_context = None
                    gaps_context = None

                    if facts_key in st.session_state:
                        facts = st.session_state[facts_key]
                        patient_context = {
                            "facts": facts,
                            "note": db.get_latest_note(patient_id)
                        }

                    if gaps_key in st.session_state:
                        result = st.session_state[gaps_key]
                        gaps_context = {
                            "gaps": result.gaps,
                            "overall_status": result.overall_status,
                            "gaps_found": result.gaps_found
                        }

                    # Execute query through orchestrator
                    with st.spinner("Processing query..."):
                        orchestrator = st.session_state.orchestrator
                        orchestrator_result = orchestrator.process_query(
                            query=user_input.strip(),
                            patient_id=patient_id,
                            patient_context=patient_context,
                            gaps_context=gaps_context
                        )

                    # Add response to chat
                    st.session_state.chat_history.append({
                        "role": "agent",
                        "content": orchestrator_result.response,
                    })

                    # Log orchestrator execution
                    add_governance_log(
                        "Orchestrator",
                        f"process_query ({orchestrator_result.intent})",
                        orchestrator_result.success,
                        f"Plan: {orchestrator_result.plan_id}, Steps: {len(orchestrator_result.steps_executed)}"
                    )

                    # Log booking if it happened
                    if orchestrator_result.booking_result and orchestrator_result.booking_result.success:
                        add_governance_log(
                            "BookingTool",
                            "appointment_booked",
                            True,
                            f"{orchestrator_result.booking_result.doctor_name} ({orchestrator_result.booking_result.specialty})"
                        )

                    # Update token count (estimate)
                    st.session_state.total_tokens += len(orchestrator_result.response) // 4
                    st.session_state.session_cost += (len(orchestrator_result.response) // 4) * 0.000001

                    st.rerun()

    # Right column - Governance Log
    with right_col:
        st.markdown("### Governance Log")
        render_governance_log()

        if st.button("Clear Logs", key="clear_logs"):
            st.session_state.governance_logs = []
            add_governance_log("System", "Logs cleared", True)
            st.rerun()


if __name__ == "__main__":
    main()
