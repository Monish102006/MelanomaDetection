"""Report Agent: turns validated CaseState into an evidence-grounded output."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
from .state import CaseState

@dataclass
class ReportResult:
    dashboard_payload: Dict[str, Any]
    pdf_path: Optional[str] = None

class ReportAgent:
    name = "report"
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    def generate_explanation(self, state: CaseState) -> CaseState:
        from .explainability import EvidenceGroundedExplanationService
        
        # Instantiate the RAG explainability service
        service = EvidenceGroundedExplanationService(self.config)
        
        # Call the explain method to get an ExplanationResult
        state.explanation_result = service.explain(state.diagnosis_result, state.evidence)
        if state.explanation_result and state.grad_cam_saved_path:
            from pathlib import Path
            cam_name = Path(state.grad_cam_saved_path).name
            setattr(state.explanation_result, "grad_cam_saved_path", state.grad_cam_saved_path)
            setattr(state.explanation_result, "grad_cam_url", f"/static/outputs/gradcam_samples/{cam_name}")
        
        # Populate the legacy explanation dict for report payload
        state.explanation = vars(state.explanation_result) if state.explanation_result else {}
        
        state.uncertainty.final_synthesis = min(state.uncertainty.classification, state.uncertainty.evidence_sufficiency)
        state.record(self.name, "generate_explanation", "completed", "grounded_synthesis", tool=True)
        return state
    def generate_report(self, state: CaseState) -> CaseState:
        if state.explanation is None: raise ValueError("Report Agent requires a validated explanation")
        diagnosis = state.diagnosis_result.model_dump(mode="json") if state.diagnosis_result is not None else None
        payload = {"analysis_id": state.analysis_id, "diagnosis": diagnosis, "prediction": state.classification, "raw_measurements": {key: value.__dict__ for key, value in state.pixel_measurements.items()}, "physical_measurements": {key: value.__dict__ for key, value in state.physical_measurements.items()}, "abcde": {key: value.__dict__ for key, value in state.abcde.items()}, "clinical_context": state.clinical_context, "pending_clinical_questions": state.pending_clinical_questions, "evidence": [item.__dict__ for item in state.evidence], "evidence_assessment": state.evidence_assessment.__dict__, "uncertainty": state.uncertainty.__dict__, "explanation": state.explanation, "pipeline_metadata": {"architecture": "three-agent CaseState workflow", "trace": state.public_trace()}}
        # PDF is an optional rendering tool. Failure to render it must not
        # invalidate the structured report, which remains the source of truth.
        pdf_path = None
        if state.diagnosis_result is not None and self.config.get("save_pdf", False):
            try:
                from .explainability import ExplanationResult
                from .report_generator import ReportRenderer
                
                if hasattr(state, "explanation_result") and isinstance(state.explanation_result, ExplanationResult):
                    explanation = state.explanation_result
                else:
                    explanation = ExplanationResult(
                        summary=state.explanation.get("model_observation", ""),
                        reasoning=[state.explanation.get("literature_context", "")], evidence_citations=[],
                        confidence_assessment="requires-review" if state.explanation.get("uncertainty") else "moderate",
                        grad_cam_reliable=bool(state.gradcam.get("alignment_reliable")) if hasattr(state, "gradcam") and state.gradcam else False,
                        limitations=state.explanation.get("uncertainty", []),
                    )
                pdf_path = ReportRenderer(self.config).generate(
                    state.diagnosis_result, state.evidence, explanation, state.analysis_id
                ).pdf_path
            except Exception as exc:
                import traceback
                traceback.print_exc()
                state.flags.append(f"PDF rendering unavailable: {type(exc).__name__}")
        state.final_report, state.report_result = payload, ReportResult(payload, pdf_path)
        state.record(self.name, "generate_report", "completed", "validated_report", tool=True)
        return state
