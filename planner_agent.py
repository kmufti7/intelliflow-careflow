"""Planner agent - decomposes queries into execution steps.

Uses LLM to analyze user query and patient context to create an execution plan.
"""

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class ActionType(str, Enum):
    """Available actions in the execution plan."""
    EXTRACT_FACTS = "extract_patient_facts"
    RETRIEVE_GUIDELINES = "retrieve_guidelines"
    COMPUTE_GAPS = "compute_gaps"
    BOOK_APPOINTMENT = "book_appointment"
    COMPOSE_RESPONSE = "compose_response"
    SEARCH_PATIENTS = "search_patients"
    GET_PATIENT_INFO = "get_patient_info"


@dataclass
class PlanStep:
    """A single step in the execution plan."""
    step: int
    action: str
    input: str
    condition: Optional[str] = None
    description: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = {
            "step": self.step,
            "action": self.action,
            "input": self.input,
            "description": self.description
        }
        if self.condition:
            d["condition"] = self.condition
        return d


@dataclass
class ExecutionPlan:
    """Complete execution plan for a query."""
    plan_id: str
    query: str
    intent: str  # "gap_analysis", "booking", "explanation", "general"
    steps: list[PlanStep] = field(default_factory=list)
    requires_patient: bool = True
    patient_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "query": self.query,
            "intent": self.intent,
            "steps": [s.to_dict() for s in self.steps],
            "requires_patient": self.requires_patient,
            "patient_id": self.patient_id
        }


class PlannerAgent:
    """Agent that plans the execution steps for a clinical query.

    Uses LLM to analyze the query and create an appropriate execution plan.
    """

    # Intent classification prompts
    INTENT_PATTERNS = {
        "gap_analysis": [
            "care gap", "gaps", "what's missing", "not at goal",
            "should be on", "recommend", "need to", "analyze"
        ],
        "booking": [
            "book", "schedule", "appointment", "see a doctor",
            "refer", "referral", "follow up", "follow-up"
        ],
        "explanation": [
            "why", "explain", "tell me more", "what does",
            "how come", "reason", "because"
        ],
        "a1c_specific": [
            "a1c", "hba1c", "hemoglobin", "glucose", "sugar"
        ],
        "bp_specific": [
            "blood pressure", "bp", "hypertension", "htn"
        ],
        "medication": [
            "medication", "drug", "medicine", "lisinopril",
            "ace inhibitor", "arb", "metformin"
        ]
    }

    def __init__(self):
        """Initialize the planner agent with OpenAI client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
                    api_key = st.secrets["OPENAI_API_KEY"]
            except:
                pass

        self.client = OpenAI(api_key=api_key) if api_key else None

    def create_plan(
        self,
        query: str,
        patient_id: Optional[str] = None,
        patient_context: Optional[dict] = None,
        gaps_context: Optional[dict] = None
    ) -> ExecutionPlan:
        """Create an execution plan for the query.

        Args:
            query: User query
            patient_id: Current patient ID if selected
            patient_context: Extracted facts and patient info
            gaps_context: Already computed gaps if available

        Returns:
            ExecutionPlan with steps to execute
        """
        plan_id = str(uuid.uuid4())[:8]

        # Classify intent
        intent = self._classify_intent(query)

        # Build plan based on intent and context
        if intent == "gap_analysis":
            return self._plan_gap_analysis(plan_id, query, patient_id, patient_context, gaps_context)
        elif intent == "booking":
            return self._plan_booking(plan_id, query, patient_id, patient_context, gaps_context)
        elif intent == "explanation":
            return self._plan_explanation(plan_id, query, patient_id, patient_context, gaps_context)
        else:
            return self._plan_general(plan_id, query, patient_id, patient_context, gaps_context)

    def _classify_intent(self, query: str) -> str:
        """Classify the intent of the query.

        Args:
            query: User query

        Returns:
            Intent string
        """
        query_lower = query.lower()

        # Check for booking intent first (highest priority)
        if any(pattern in query_lower for pattern in self.INTENT_PATTERNS["booking"]):
            return "booking"

        # Check for explanation
        if any(pattern in query_lower for pattern in self.INTENT_PATTERNS["explanation"]):
            return "explanation"

        # Check for gap analysis
        if any(pattern in query_lower for pattern in self.INTENT_PATTERNS["gap_analysis"]):
            return "gap_analysis"

        # Default to gap analysis for clinical queries
        if any(pattern in query_lower for pattern in
               self.INTENT_PATTERNS["a1c_specific"] +
               self.INTENT_PATTERNS["bp_specific"] +
               self.INTENT_PATTERNS["medication"]):
            return "gap_analysis"

        return "general"

    def _plan_gap_analysis(
        self,
        plan_id: str,
        query: str,
        patient_id: Optional[str],
        patient_context: Optional[dict],
        gaps_context: Optional[dict]
    ) -> ExecutionPlan:
        """Create a plan for gap analysis queries."""
        steps = []
        step_num = 1

        # Step 1: Extract facts if not already done
        if not patient_context:
            steps.append(PlanStep(
                step=step_num,
                action=ActionType.EXTRACT_FACTS,
                input="patient_note",
                description="Extract clinical facts from patient note using regex/LLM"
            ))
            step_num += 1

        # Step 2: Retrieve relevant guidelines
        steps.append(PlanStep(
            step=step_num,
            action=ActionType.RETRIEVE_GUIDELINES,
            input="extracted_diagnoses",
            description="Search guideline index for relevant clinical guidelines"
        ))
        step_num += 1

        # Step 3: Compute gaps if not already done
        if not gaps_context:
            steps.append(PlanStep(
                step=step_num,
                action=ActionType.COMPUTE_GAPS,
                input="facts + guidelines",
                description="Apply deterministic rules to identify care gaps"
            ))
            step_num += 1

        # Step 4: Compose response
        steps.append(PlanStep(
            step=step_num,
            action=ActionType.COMPOSE_RESPONSE,
            input="all_results",
            description="Generate natural language response with citations"
        ))

        return ExecutionPlan(
            plan_id=plan_id,
            query=query,
            intent="gap_analysis",
            steps=steps,
            requires_patient=True,
            patient_id=patient_id
        )

    def _plan_booking(
        self,
        plan_id: str,
        query: str,
        patient_id: Optional[str],
        patient_context: Optional[dict],
        gaps_context: Optional[dict]
    ) -> ExecutionPlan:
        """Create a plan for booking/scheduling queries."""
        steps = []
        step_num = 1

        # Determine specialty from query or gaps
        specialty = self._infer_specialty(query, gaps_context)

        # Step 1: Extract facts if needed
        if not patient_context:
            steps.append(PlanStep(
                step=step_num,
                action=ActionType.EXTRACT_FACTS,
                input="patient_note",
                description="Extract patient facts to determine appropriate specialty"
            ))
            step_num += 1

        # Step 2: Compute gaps if needed (to justify referral)
        if not gaps_context:
            steps.append(PlanStep(
                step=step_num,
                action=ActionType.COMPUTE_GAPS,
                input="facts",
                description="Identify gaps to determine referral need"
            ))
            step_num += 1

        # Step 3: Book appointment
        steps.append(PlanStep(
            step=step_num,
            action=ActionType.BOOK_APPOINTMENT,
            input=specialty or "auto_detect",
            condition="if_gaps_detected",
            description=f"Book appointment with {specialty or 'appropriate specialist'}"
        ))
        step_num += 1

        # Step 4: Compose response
        steps.append(PlanStep(
            step=step_num,
            action=ActionType.COMPOSE_RESPONSE,
            input="booking_result",
            description="Confirm booking details to user"
        ))

        return ExecutionPlan(
            plan_id=plan_id,
            query=query,
            intent="booking",
            steps=steps,
            requires_patient=True,
            patient_id=patient_id
        )

    def _plan_explanation(
        self,
        plan_id: str,
        query: str,
        patient_id: Optional[str],
        patient_context: Optional[dict],
        gaps_context: Optional[dict]
    ) -> ExecutionPlan:
        """Create a plan for explanation queries."""
        steps = []
        step_num = 1

        # Step 1: Retrieve relevant guidelines for the explanation
        steps.append(PlanStep(
            step=step_num,
            action=ActionType.RETRIEVE_GUIDELINES,
            input=query,
            description="Find guidelines relevant to the explanation request"
        ))
        step_num += 1

        # Step 2: Use existing gaps or compute if needed
        if not gaps_context:
            steps.append(PlanStep(
                step=step_num,
                action=ActionType.COMPUTE_GAPS,
                input="facts + guidelines",
                description="Compute gaps to provide context for explanation"
            ))
            step_num += 1

        # Step 3: Compose detailed explanation
        steps.append(PlanStep(
            step=step_num,
            action=ActionType.COMPOSE_RESPONSE,
            input="explanation_request",
            description="Generate detailed explanation with guideline citations"
        ))

        return ExecutionPlan(
            plan_id=plan_id,
            query=query,
            intent="explanation",
            steps=steps,
            requires_patient=True,
            patient_id=patient_id
        )

    def _plan_general(
        self,
        plan_id: str,
        query: str,
        patient_id: Optional[str],
        patient_context: Optional[dict],
        gaps_context: Optional[dict]
    ) -> ExecutionPlan:
        """Create a plan for general queries."""
        steps = []

        # For general queries, just compose a response using available context
        steps.append(PlanStep(
            step=1,
            action=ActionType.COMPOSE_RESPONSE,
            input="general_query",
            description="Generate response using available patient context"
        ))

        return ExecutionPlan(
            plan_id=plan_id,
            query=query,
            intent="general",
            steps=steps,
            requires_patient=patient_id is not None,
            patient_id=patient_id
        )

    def _infer_specialty(self, query: str, gaps_context: Optional[dict]) -> Optional[str]:
        """Infer the appropriate specialty from query and gaps.

        Args:
            query: User query
            gaps_context: Computed gaps

        Returns:
            Specialty string or None
        """
        query_lower = query.lower()

        # Direct specialty mentions
        specialty_keywords = {
            "endocrinology": ["endocrin", "diabetes", "a1c", "thyroid"],
            "cardiology": ["cardiol", "heart", "bp", "blood pressure", "hypertension"],
            "nephrology": ["nephrol", "kidney", "renal", "ckd"],
            "internal medicine": ["internal", "primary care", "general"],
        }

        for specialty, keywords in specialty_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return specialty

        # Infer from gaps
        if gaps_context:
            gaps = gaps_context.get("gaps", [])
            for gap in gaps:
                if isinstance(gap, dict):
                    gap_type = gap.get("gap_type", "")
                else:
                    gap_type = getattr(gap, "gap_type", "")

                if "A1C" in gap_type:
                    return "Endocrinology"
                elif "HTN" in gap_type or "BP" in gap_type:
                    return "Cardiology"

        return None

    def create_plan_with_llm(
        self,
        query: str,
        patient_context: Optional[dict] = None,
        available_actions: list[str] = None
    ) -> ExecutionPlan:
        """Create a plan using LLM for complex queries.

        Args:
            query: User query
            patient_context: Patient information
            available_actions: List of available action types

        Returns:
            ExecutionPlan
        """
        if not self.client:
            # Fall back to rule-based planning
            return self.create_plan(query)

        if available_actions is None:
            available_actions = [a.value for a in ActionType]

        prompt = f"""You are a clinical workflow planner. Given a user query about a patient, create an execution plan.

Available actions:
- extract_patient_facts: Extract clinical data from patient note
- retrieve_guidelines: Search medical guidelines database
- compute_gaps: Apply rules to identify care gaps
- book_appointment: Schedule an appointment with a specialist
- compose_response: Generate final response to user

User query: "{query}"

Patient context available: {bool(patient_context)}

Return a JSON object with:
{{
    "intent": "gap_analysis" | "booking" | "explanation" | "general",
    "steps": [
        {{"step": 1, "action": "action_name", "input": "what to use", "description": "what this does"}}
    ]
}}

Keep plans concise (2-5 steps). Only include necessary steps based on available context."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a clinical workflow planner. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=500
            )

            result = json.loads(response.choices[0].message.content)

            steps = [
                PlanStep(
                    step=s["step"],
                    action=s["action"],
                    input=s.get("input", ""),
                    condition=s.get("condition"),
                    description=s.get("description", "")
                )
                for s in result.get("steps", [])
            ]

            return ExecutionPlan(
                plan_id=str(uuid.uuid4())[:8],
                query=query,
                intent=result.get("intent", "general"),
                steps=steps,
                requires_patient=True
            )

        except Exception as e:
            # Fall back to rule-based planning
            return self.create_plan(query)


# Convenience function
def create_execution_plan(
    query: str,
    patient_id: Optional[str] = None,
    patient_context: Optional[dict] = None,
    gaps_context: Optional[dict] = None
) -> ExecutionPlan:
    """Create an execution plan for a clinical query.

    Args:
        query: User query
        patient_id: Current patient ID
        patient_context: Patient facts
        gaps_context: Computed gaps

    Returns:
        ExecutionPlan
    """
    planner = PlannerAgent()
    return planner.create_plan(query, patient_id, patient_context, gaps_context)


# Test function
def test_planner():
    """Test the planner agent."""
    planner = PlannerAgent()

    test_queries = [
        "What care gaps does this patient have?",
        "Why should this patient be on an ACE inhibitor?",
        "Book an appointment with endocrinology",
        "Is the A1C at goal?",
        "Schedule a follow-up for blood pressure",
    ]

    print("=" * 60)
    print("PLANNER AGENT TEST")
    print("=" * 60)

    for query in test_queries:
        plan = planner.create_plan(query, patient_id="PT001")
        print(f"\nQuery: {query}")
        print(f"Intent: {plan.intent}")
        print(f"Steps:")
        for step in plan.steps:
            cond = f" [if: {step.condition}]" if step.condition else ""
            print(f"  {step.step}. {step.action}: {step.description}{cond}")


if __name__ == "__main__":
    test_planner()
