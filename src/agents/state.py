"""Typed state and audit records shared by all workflow agents.

This module deliberately contains data only. It is the boundary between model
observations, clinical information supplied by a person, and untrusted
literature retrieved from the knowledge base.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

Availability = Literal["available", "unavailable", "pending"]

@dataclass
class Measurement:
    value: Optional[float]
    unit: Literal["pixels", "mm", "px2", "mm2", "unknown"]
    calibrated: bool = False
    method: str = ""
    reason_unavailable: Optional[str] = None

@dataclass
class ABCDEFeature:
    criterion: Literal["A", "B", "C", "D", "E"]
    availability: Availability
    raw: Dict[str, Any] = field(default_factory=dict)
    interpretation: Optional[str] = None
    limitation: Optional[str] = None

@dataclass
class Uncertainty:
    """Signals, not probabilities that may be multiplied together."""
    classification: float = 1.0
    segmentation: float = 1.0
    measurement: float = 1.0
    calibration: float = 1.0
    abcde: float = 1.0
    clinical_context: float = 1.0
    retrieval: float = 1.0
    evidence_sufficiency: float = 1.0
    final_synthesis: float = 1.0

@dataclass
class EvidenceRecord:
    id: str
    title: str
    source: str
    content: str
    relevance_score: float = 0.0
    goal: str = ""
    cycle: int = 0
    trust_boundary: str = "UNTRUSTED_EVIDENCE"

@dataclass
class EvidenceAssessment:
    sufficient: bool = False
    covered: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    follow_up_queries: List[str] = field(default_factory=list)
    cycles_completed: int = 0

@dataclass
class ExecutionTrace:
    step: int
    agent: str
    action: str
    result: str
    reason_category: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    detail: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ActionStatus:
    status: Literal["pending", "running", "completed", "failed", "skipped"] = "pending"
    error: Optional[str] = None

@dataclass
class CaseState:
    """The only mutable case contract used by the orchestrator and agents."""
    analysis_id: str
    image_path: str
    image_metadata: Dict[str, Any] = field(default_factory=dict)
    image_quality: Dict[str, Any] = field(default_factory=dict)
    classification: Dict[str, Any] = field(default_factory=dict)
    segmentation: Dict[str, Any] = field(default_factory=dict)
    gradcam: Dict[str, Any] = field(default_factory=dict)
    grad_cam_saved_path: Optional[str] = None
    pixel_measurements: Dict[str, Measurement] = field(default_factory=dict)
    physical_measurements: Dict[str, Measurement] = field(default_factory=dict)
    calibration: Dict[str, Any] = field(default_factory=dict)
    abcde: Dict[str, ABCDEFeature] = field(default_factory=dict)
    clinical_context: Dict[str, Any] = field(default_factory=dict)
    pending_clinical_questions: List[Dict[str, str]] = field(default_factory=list)
    evidence: List[EvidenceRecord] = field(default_factory=list)
    evidence_assessment: EvidenceAssessment = field(default_factory=EvidenceAssessment)
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    verified_claims: List[Dict[str, Any]] = field(default_factory=list)
    explanation: Optional[Dict[str, Any]] = None
    final_report: Optional[Dict[str, Any]] = None
    agent_decisions: List[ExecutionTrace] = field(default_factory=list)
    tool_executions: List[ExecutionTrace] = field(default_factory=list)
    action_status: Dict[str, ActionStatus] = field(default_factory=dict)
    historical_images: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    status: Literal["pending", "running", "needs_clinical_context", "completed", "failed"] = "pending"
    error_code: Optional[str] = None
    user_message: Optional[str] = None
    retryable: bool = False
    diagnosis_result: Any = None
    retrieved_evidence: List[EvidenceRecord] = field(default_factory=list)
    explanation_result: Any = None
    report_result: Any = None
    fusion_result: Any = None  # FusionResult from ABCDEFusionScorer

    @property
    def overall_status(self) -> str:
        return self.status

    def record(self, agent: str, action: str, result: str, reason_category: str,
               detail: Optional[Dict[str, Any]] = None, tool: bool = False) -> None:
        event = ExecutionTrace(len(self.agent_decisions) + len(self.tool_executions) + 1,
                               agent, action, result, reason_category, detail=detail or {})
        (self.tool_executions if tool else self.agent_decisions).append(event)

    def public_trace(self) -> List[Dict[str, Any]]:
        return [asdict(event) for event in self.agent_decisions + self.tool_executions]
