"""Backward-compatibility module aliasing OrchestratorAgent as SupervisorAgent."""
from .orchestrator import OrchestratorAgent

SupervisorAgent = OrchestratorAgent
__all__ = ["SupervisorAgent"]
