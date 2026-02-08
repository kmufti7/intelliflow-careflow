"""Tests for CareFlow Chaos Mode — deterministic failure injection."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chaos_mode import (
    ChaosConfig,
    ChaosError,
    ChaosFailureType,
    FALLBACK_RESPONSE,
    check_faiss_chaos,
    check_pinecone_chaos,
    get_chaos_config,
    set_chaos_config,
)


class TestChaosConfig(unittest.TestCase):
    """Tests for chaos configuration."""

    def tearDown(self):
        set_chaos_config(enabled=False)

    def test_default_config_disabled(self):
        config = set_chaos_config(enabled=False)
        self.assertFalse(config.enabled)
        self.assertFalse(config.is_faiss_failure())
        self.assertFalse(config.is_pinecone_failure())

    def test_enable_faiss_failure(self):
        config = set_chaos_config(enabled=True, failure_type=ChaosFailureType.FAISS_UNAVAILABLE)
        self.assertTrue(config.enabled)
        self.assertTrue(config.is_faiss_failure())
        self.assertFalse(config.is_pinecone_failure())

    def test_enable_pinecone_failure(self):
        config = set_chaos_config(enabled=True, failure_type=ChaosFailureType.PINECONE_UNAVAILABLE)
        self.assertTrue(config.enabled)
        self.assertFalse(config.is_faiss_failure())
        self.assertTrue(config.is_pinecone_failure())


class TestChaosChecks(unittest.TestCase):
    """Tests for chaos failure injection checks."""

    def tearDown(self):
        set_chaos_config(enabled=False)

    def test_faiss_check_raises_when_enabled(self):
        set_chaos_config(enabled=True, failure_type=ChaosFailureType.FAISS_UNAVAILABLE)
        with self.assertRaises(ChaosError) as ctx:
            check_faiss_chaos()
        self.assertEqual(ctx.exception.failure_type, ChaosFailureType.FAISS_UNAVAILABLE)

    def test_pinecone_check_raises_when_enabled(self):
        set_chaos_config(enabled=True, failure_type=ChaosFailureType.PINECONE_UNAVAILABLE)
        with self.assertRaises(ChaosError) as ctx:
            check_pinecone_chaos()
        self.assertEqual(ctx.exception.failure_type, ChaosFailureType.PINECONE_UNAVAILABLE)

    def test_faiss_check_silent_when_disabled(self):
        set_chaos_config(enabled=False)
        check_faiss_chaos()  # Should not raise

    def test_pinecone_check_silent_when_disabled(self):
        set_chaos_config(enabled=False)
        check_pinecone_chaos()  # Should not raise

    def test_faiss_check_silent_when_pinecone_failure_selected(self):
        set_chaos_config(enabled=True, failure_type=ChaosFailureType.PINECONE_UNAVAILABLE)
        check_faiss_chaos()  # Should not raise — wrong failure type


class TestFallbackResponse(unittest.TestCase):
    """Tests for the fallback response content."""

    def test_fallback_contains_service_disruption(self):
        self.assertIn("service disruption", FALLBACK_RESPONSE)

    def test_fallback_contains_no_clinical_decisions_warning(self):
        self.assertIn("No clinical decisions should be made", FALLBACK_RESPONSE)


try:
    from care_orchestrator import CareOrchestrator
    HAS_ORCHESTRATOR = True
except ImportError:
    HAS_ORCHESTRATOR = False


@unittest.skipUnless(HAS_ORCHESTRATOR, "Requires openai and other orchestrator dependencies")
class TestChaosAuditIntegration(unittest.TestCase):
    """Tests that chaos failures are logged through the orchestrator."""

    def tearDown(self):
        set_chaos_config(enabled=False)

    def test_orchestrator_catches_chaos_and_returns_fallback(self):
        """When chaos is enabled, orchestrator returns fallback response."""
        set_chaos_config(enabled=True, failure_type=ChaosFailureType.FAISS_UNAVAILABLE)

        logged_entries = []

        def log_callback(component, action, success, details=""):
            logged_entries.append({
                "component": component,
                "action": action,
                "success": success,
                "details": details,
            })

        orchestrator = CareOrchestrator(log_callback=log_callback)
        result = orchestrator.process_query(
            query="What care gaps does this patient have?",
            patient_id="PT001",
        )

        # Should get fallback response
        self.assertFalse(result.success)
        self.assertEqual(result.response, FALLBACK_RESPONSE)

        # Should have a chaos log entry
        chaos_logs = [e for e in logged_entries if e["component"] == "ChaosMode"]
        self.assertTrue(len(chaos_logs) > 0, "Expected ChaosMode log entry")
        self.assertFalse(chaos_logs[0]["success"])
        self.assertIn("faiss_unavailable", chaos_logs[0]["action"])

    def test_orchestrator_normal_when_chaos_disabled(self):
        """When chaos is disabled, orchestrator proceeds normally (no ChaosError)."""
        set_chaos_config(enabled=False)

        orchestrator = CareOrchestrator()
        result = orchestrator.process_query(
            query="What care gaps does this patient have?",
            patient_id="PT001",
        )

        # Should NOT get fallback response
        self.assertNotEqual(result.response, FALLBACK_RESPONSE)


if __name__ == "__main__":
    unittest.main()
