"""Responder agent - generates natural language responses."""

# TODO: Implement responder agent
# - Format care gap findings
# - Generate actionable recommendations
# - Include citations to guidelines


class ResponderAgent:
    """Generates natural language responses for care gap findings."""

    def __init__(self):
        """Initialize the responder agent."""
        pass

    async def generate_response(self, gaps: list, context: dict) -> str:
        """Generate a response summarizing care gaps.

        Args:
            gaps: List of identified care gaps
            context: Additional context for response generation

        Returns:
            Natural language response
        """
        raise NotImplementedError("Responder agent not yet implemented")
