"""CareFlow orchestrator - coordinates clinical gap analysis workflow.

Executes plans created by the planner agent, coordinating all components.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from datetime import datetime

from openai import OpenAI
from dotenv import load_dotenv

from planner_agent import PlannerAgent, ExecutionPlan, ActionType
from extraction import PatientFactExtractor, ExtractedFacts
from reasoning_engine import ReasoningEngine, ReasoningResult, GapResult
from tools import BookingTool, VectorSearchTool, BookingResult
from chaos_mode import get_chaos_config, check_faiss_chaos, check_pinecone_chaos, ChaosError, FALLBACK_RESPONSE

load_dotenv()


@dataclass
class StepResult:
    """Result of executing a single plan step."""
    step_num: int
    action: str
    success: bool
    duration_ms: int
    output: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class OrchestratorResult:
    """Complete result of orchestrator execution."""
    plan_id: str
    query: str
    intent: str
    success: bool
    response: str
    steps_executed: list[StepResult] = field(default_factory=list)
    total_duration_ms: int = 0
    extracted_facts: Optional[ExtractedFacts] = None
    reasoning_result: Optional[ReasoningResult] = None
    booking_result: Optional[BookingResult] = None
    guidelines_retrieved: list[dict] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "query": self.query,
            "intent": self.intent,
            "success": self.success,
            "response": self.response,
            "steps_executed": [s.to_dict() for s in self.steps_executed],
            "total_duration_ms": self.total_duration_ms,
            "extracted_facts": self.extracted_facts.to_dict() if self.extracted_facts else None,
            "reasoning_result": self.reasoning_result.to_dict() if self.reasoning_result else None,
            "booking_result": self.booking_result.to_dict() if self.booking_result else None,
            "guidelines_retrieved": self.guidelines_retrieved,
            "error": self.error
        }


class CareOrchestrator:
    """Main orchestrator for CareFlow clinical gap analysis.

    Coordinates planner, extraction, reasoning, tools, and response generation.
    """

    def __init__(self, db=None, log_callback: Optional[Callable] = None):
        """Initialize the orchestrator.

        Args:
            db: Database connection (optional, will get from singleton)
            log_callback: Optional callback for logging (component, action, success, details)
        """
        self.db = db
        self.log_callback = log_callback

        # Initialize components
        self.planner = PlannerAgent()
        self.extractor = PatientFactExtractor()
        self.reasoning_engine = ReasoningEngine()
        self.booking_tool = BookingTool(db)
        self.vector_search = VectorSearchTool()

        # Initialize OpenAI client for response generation
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
                    api_key = st.secrets["OPENAI_API_KEY"]
            except:
                pass

        self.client = OpenAI(api_key=api_key) if api_key else None

    def _get_db(self):
        """Get database connection."""
        if self.db is None:
            from care_database import get_database
            self.db = get_database()
        return self.db

    def _log(self, component: str, action: str, success: bool, details: str = ""):
        """Log an action."""
        if self.log_callback:
            self.log_callback(component, action, success, details)

    def process_query(
        self,
        query: str,
        patient_id: Optional[str] = None,
        patient_context: Optional[dict] = None,
        gaps_context: Optional[dict] = None
    ) -> OrchestratorResult:
        """Process a clinical query through the workflow.

        Args:
            query: User query about patient care gaps
            patient_id: Current patient ID
            patient_context: Pre-extracted patient facts (optional)
            gaps_context: Pre-computed gaps (optional)

        Returns:
            OrchestratorResult with response and metadata
        """
        start_time = time.time()

        self._log("Orchestrator", "process_query_start", True, f"Query: {query[:50]}...")

        # Step 1: Create execution plan
        plan = self.planner.create_plan(
            query=query,
            patient_id=patient_id,
            patient_context=patient_context,
            gaps_context=gaps_context
        )

        self._log("Planner", "plan_created", True,
                  f"Intent: {plan.intent}, Steps: {len(plan.steps)}")

        # Initialize result
        result = OrchestratorResult(
            plan_id=plan.plan_id,
            query=query,
            intent=plan.intent,
            success=False,
            response=""
        )

        # Check if patient is required but not provided
        if plan.requires_patient and not patient_id:
            result.error = "No patient selected. Please select a patient first."
            result.response = "Please select a patient from the sidebar before asking clinical questions."
            self._log("Orchestrator", "no_patient", False, "Patient required but not selected")
            return result

        # Execute each step
        try:
            for plan_step in plan.steps:
                step_result = self._execute_step(
                    plan_step,
                    patient_id=patient_id,
                    result=result
                )
                result.steps_executed.append(step_result)

                if not step_result.success and plan_step.action != ActionType.BOOK_APPOINTMENT:
                    # Non-booking failures are critical
                    result.error = step_result.error
                    break

            # Mark success if we got a response
            if result.response:
                result.success = True

        except ChaosError as ce:
            result.error = str(ce)
            result.response = FALLBACK_RESPONSE
            result.success = False
            self._log("ChaosMode", f"failure_injected: {ce.failure_type.value}", False, str(ce))

        except Exception as e:
            result.error = str(e)
            self._log("Orchestrator", "execution_error", False, str(e))

        result.total_duration_ms = int((time.time() - start_time) * 1000)
        self._log("Orchestrator", "process_query_complete", result.success,
                  f"Duration: {result.total_duration_ms}ms")

        return result

    def _execute_step(
        self,
        plan_step,
        patient_id: Optional[str],
        result: OrchestratorResult
    ) -> StepResult:
        """Execute a single plan step.

        Args:
            plan_step: The step to execute
            patient_id: Current patient ID
            result: The orchestrator result to update

        Returns:
            StepResult
        """
        start_time = time.time()
        action = plan_step.action

        self._log("Orchestrator", f"execute_step: {action}", True, plan_step.description)

        try:
            if action == ActionType.EXTRACT_FACTS:
                output = self._execute_extract_facts(patient_id, result)
            elif action == ActionType.RETRIEVE_GUIDELINES:
                output = self._execute_retrieve_guidelines(plan_step.input, result)
            elif action == ActionType.COMPUTE_GAPS:
                output = self._execute_compute_gaps(patient_id, result)
            elif action == ActionType.BOOK_APPOINTMENT:
                output = self._execute_book_appointment(patient_id, plan_step.input, result)
            elif action == ActionType.COMPOSE_RESPONSE:
                output = self._execute_compose_response(result)
            else:
                output = {"error": f"Unknown action: {action}"}

            duration_ms = int((time.time() - start_time) * 1000)
            success = "error" not in output

            return StepResult(
                step_num=plan_step.step,
                action=action,
                success=success,
                duration_ms=duration_ms,
                output=output,
                error=output.get("error")
            )

        except ChaosError:
            raise  # Let ChaosError propagate to process_query handler

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return StepResult(
                step_num=plan_step.step,
                action=action,
                success=False,
                duration_ms=duration_ms,
                error=str(e)
            )

    def _execute_extract_facts(
        self,
        patient_id: str,
        result: OrchestratorResult
    ) -> dict:
        """Execute fact extraction step."""
        db = self._get_db()
        note = db.get_latest_note(patient_id)

        if not note:
            return {"error": f"No notes found for patient {patient_id}"}

        facts = self.extractor.extract(note["note_text"])
        result.extracted_facts = facts

        self._log("Extractor", "extract_facts", facts.is_complete(),
                  f"A1C={facts.a1c}, BP={facts.blood_pressure}, Method={facts.extraction_method}")

        return {
            "a1c": facts.a1c,
            "blood_pressure": facts.blood_pressure,
            "diagnoses": facts.diagnoses,
            "medications": facts.medications,
            "method": facts.extraction_method,
            "confidence": facts.confidence
        }

    def _execute_retrieve_guidelines(
        self,
        query_input: str,
        result: OrchestratorResult
    ) -> dict:
        """Execute guideline retrieval step."""
        # Chaos mode: check for simulated retrieval failures
        check_faiss_chaos()
        check_pinecone_chaos()

        # Build search query from diagnoses or input
        if result.extracted_facts and result.extracted_facts.diagnoses:
            search_query = " ".join(result.extracted_facts.diagnoses)
        else:
            search_query = query_input

        guidelines = self.vector_search.search_guidelines(search_query, top_k=3)
        result.guidelines_retrieved = guidelines

        self._log("VectorSearch", "retrieve_guidelines", len(guidelines) > 0,
                  f"Found {len(guidelines)} guidelines")

        return {
            "guidelines_found": len(guidelines),
            "guidelines": [
                {"id": g["id"], "score": g["score"]}
                for g in guidelines
            ]
        }

    def _execute_compute_gaps(
        self,
        patient_id: str,
        result: OrchestratorResult
    ) -> dict:
        """Execute gap computation step."""
        if not result.extracted_facts:
            # Extract facts first if not done
            self._execute_extract_facts(patient_id, result)

        if not result.extracted_facts:
            return {"error": "Could not extract patient facts"}

        reasoning_result = self.reasoning_engine.evaluate_patient(
            result.extracted_facts,
            patient_id
        )
        result.reasoning_result = reasoning_result

        detected_gaps = [g for g in reasoning_result.gaps if g.gap_detected]
        self._log("ReasoningEngine", "compute_gaps", True,
                  f"{reasoning_result.gaps_found} gaps found, status: {reasoning_result.overall_status}")

        return {
            "gaps_found": reasoning_result.gaps_found,
            "gaps_closed": reasoning_result.gaps_closed,
            "overall_status": reasoning_result.overall_status,
            "detected_gaps": [
                {"type": g.gap_type, "severity": g.severity}
                for g in detected_gaps
            ]
        }

    def _execute_book_appointment(
        self,
        patient_id: str,
        specialty_input: str,
        result: OrchestratorResult
    ) -> dict:
        """Execute appointment booking step."""
        # Determine specialty
        if specialty_input == "auto_detect":
            # Infer from gaps
            if result.reasoning_result:
                detected_gaps = [g for g in result.reasoning_result.gaps if g.gap_detected]
                if detected_gaps:
                    # Use the highest severity gap
                    detected_gaps.sort(key=lambda g: ["low", "moderate", "high"].index(g.severity), reverse=True)
                    top_gap = detected_gaps[0]
                    specialty = self.booking_tool.GAP_TO_SPECIALTY.get(top_gap.gap_type, "Internal Medicine")
                    reason = f"Care gap follow-up: {top_gap.gap_type}"
                else:
                    return {"skipped": True, "reason": "No gaps detected, booking not needed"}
            else:
                specialty = "Internal Medicine"
                reason = "General follow-up"
        else:
            specialty = specialty_input
            reason = f"Follow-up for {specialty}"

        booking_result = self.booking_tool.book_appointment(
            patient_id=patient_id,
            specialty=specialty,
            reason=reason
        )
        result.booking_result = booking_result

        self._log("BookingTool", "book_appointment", booking_result.success,
                  booking_result.message)

        return booking_result.to_dict()

    def _execute_compose_response(
        self,
        result: OrchestratorResult
    ) -> dict:
        """Execute response composition step using LLM."""
        # Build context for response generation
        context_parts = []

        # Add extracted facts
        if result.extracted_facts:
            facts = result.extracted_facts
            context_parts.append(f"""
PATIENT FACTS:
- A1C: {facts.a1c}%
- Blood Pressure: {facts.blood_pressure}
- Diagnoses: {', '.join(facts.diagnoses)}
- Medications: {', '.join(facts.medications)}
- Extraction Method: {facts.extraction_method} (confidence: {facts.confidence:.0%})
""")

        # Add gap analysis
        if result.reasoning_result:
            rr = result.reasoning_result
            detected = [g for g in rr.gaps if g.gap_detected]
            closed = [g for g in rr.gaps if not g.gap_detected]

            context_parts.append(f"""
CARE GAP ANALYSIS:
- Gaps Found: {rr.gaps_found}
- Gaps Closed: {rr.gaps_closed}
- Status: {rr.overall_status}
""")

            if detected:
                context_parts.append("DETECTED GAPS:")
                for gap in detected:
                    context_parts.append(f"""
  [{gap.severity.upper()}] {gap.gap_type}
  - Comparison: {gap.comparison}
  - Therefore: {gap.therefore}
  - Recommendation: {gap.recommendation}
  - Guideline: {gap.guideline_id}
""")

            if closed:
                context_parts.append("\nCLOSED GAPS:")
                for gap in closed:
                    context_parts.append(f"  - {gap.gap_type}: {gap.therefore}")

        # Add guideline context
        if result.guidelines_retrieved:
            context_parts.append("\nRELEVANT GUIDELINES:")
            for g in result.guidelines_retrieved:
                text_preview = g.get("text", "")[:200]
                context_parts.append(f"  - {g['id']}: {text_preview}...")

        # Add booking result
        if result.booking_result and result.booking_result.success:
            br = result.booking_result
            context_parts.append(f"""
APPOINTMENT BOOKED:
- Doctor: {br.doctor_name} ({br.specialty})
- Date/Time: {br.slot_datetime}
- Reason: {br.reason}
""")

        context = "\n".join(context_parts)

        # Generate response with LLM
        if self.client:
            response_text = self._generate_llm_response(result.query, context, result.intent)
        else:
            response_text = self._generate_fallback_response(result)

        result.response = response_text

        return {"response_length": len(response_text)}

    def _generate_llm_response(
        self,
        query: str,
        context: str,
        intent: str
    ) -> str:
        """Generate response using LLM."""
        system_prompt = """You are a clinical decision support assistant for CareFlow.
Your role is to explain care gaps and clinical findings to healthcare providers.

Guidelines:
1. Be concise but thorough
2. Always cite evidence: "[PATIENT: value]" and "[GUIDELINE: id]"
3. Structure responses clearly with findings, reasoning, and recommendations
4. Use clinical terminology appropriately
5. If an appointment was booked, confirm the details

Example citation format:
"The patient's A1C of 8.2% [PATIENT: A1C=8.2%] exceeds the target of <7.0% [GUIDELINE: guideline_001_a1c_threshold]."
"""

        prompt = f"""Based on the following clinical context, respond to the user's question.

USER QUESTION: {query}

CLINICAL CONTEXT:
{context}

Provide a helpful, evidence-based response with proper citations."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            self._log("LLM", "generate_response", False, str(e))
            return f"I encountered an error generating the response: {str(e)}"

    def _generate_fallback_response(
        self,
        result: OrchestratorResult
    ) -> str:
        """Generate response without LLM."""
        parts = []

        if result.reasoning_result:
            rr = result.reasoning_result
            detected = [g for g in rr.gaps if g.gap_detected]

            if detected:
                parts.append(f"**Care Gaps Identified ({len(detected)}):**\n")
                for gap in detected:
                    parts.append(f"- **{gap.gap_type}** [{gap.severity.upper()}]")
                    parts.append(f"  - {gap.therefore}")
                    parts.append(f"  - Recommendation: {gap.recommendation}\n")
            else:
                parts.append("**No care gaps identified.** All evaluated criteria are met.\n")

        if result.booking_result and result.booking_result.success:
            br = result.booking_result
            parts.append(f"\n**Appointment Scheduled:**")
            parts.append(f"- {br.doctor_name} ({br.specialty})")
            parts.append(f"- {br.slot_datetime}")

        return "\n".join(parts) if parts else "Analysis complete. Please see the care gaps panel above."


# Convenience function
def process_clinical_query(
    query: str,
    patient_id: Optional[str] = None,
    db=None,
    log_callback: Optional[Callable] = None
) -> OrchestratorResult:
    """Process a clinical query through the CareFlow orchestrator.

    Args:
        query: User query
        patient_id: Current patient ID
        db: Database connection
        log_callback: Logging callback

    Returns:
        OrchestratorResult
    """
    orchestrator = CareOrchestrator(db=db, log_callback=log_callback)
    return orchestrator.process_query(query, patient_id)


# Test function
def test_orchestrator():
    """Test the orchestrator."""
    from care_database import get_database

    db = get_database()

    def log_callback(component, action, success, details):
        status = "OK" if success else "FAIL"
        print(f"  [{status}] {component}: {action} - {details}")

    orchestrator = CareOrchestrator(db=db, log_callback=log_callback)

    print("=" * 60)
    print("ORCHESTRATOR TEST")
    print("=" * 60)

    # Test gap analysis query
    print("\n--- Test 1: Gap Analysis Query ---")
    result = orchestrator.process_query(
        query="What care gaps does this patient have?",
        patient_id="PT001"
    )
    print(f"\nResponse:\n{result.response[:500]}...")

    # Test booking query
    print("\n--- Test 2: Booking Query ---")
    result2 = orchestrator.process_query(
        query="Book an appointment with endocrinology for A1C follow-up",
        patient_id="PT003"
    )
    print(f"\nResponse:\n{result2.response[:500]}...")

    # Test explanation query
    print("\n--- Test 3: Explanation Query ---")
    result3 = orchestrator.process_query(
        query="Why should this patient be on an ACE inhibitor?",
        patient_id="PT004"
    )
    print(f"\nResponse:\n{result3.response[:500]}...")


if __name__ == "__main__":
    test_orchestrator()
