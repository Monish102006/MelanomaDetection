"""Orchestrator Agent implementing an observe -> decide -> act workflow loop with LLM cognitive reasoning."""
from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import yaml

from .clinical_context import ClinicalContextAgent
from .evidence import EvidenceAgent
from .report import ReportAgent
from .state import CaseState
from .vision import VisionAgent
from .abcde_fusion import compute_fusion

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    name = "orchestrator"

    def __init__(
        self,
        config_path: str = "config.yaml",
        *,
        vision: Any = None,
        clinical: Any = None,
        evidence: Any = None,
        report: Any = None,
    ):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        self.config_path = config_path
        with open(config_path, encoding="utf-8") as file:
            self.config = yaml.safe_load(file) or {}

        orchestrator = self.config.get("orchestrator", {})
        self.confidence_threshold = float(orchestrator.get("classification_confidence_threshold", 70.0)) / 100.0
        self.gradcam_threshold = float(orchestrator.get("gradcam_reliability_threshold", 0.30))
        self.max_retry_attempts = int(orchestrator.get("max_retry_attempts", 1))
        self.max_steps = int(orchestrator.get("max_supervisor_steps", orchestrator.get("max_orchestrator_steps", 12)))

        # Pre-resolve sub-configs so agents never reach into self.config directly
        self._explainability_cfg = self.config.get("explainability", {})
        self._quality_cfg = self.config.get("quality", {})
        self._abcde_weights = orchestrator.get("abcde_weights")

        # Merge report output config WITH explainability config so ReportAgent
        # always receives mode/llm_provider/model for EvidenceGroundedExplanationService
        report_cfg = {**self.config.get("report", {}), **self._explainability_cfg}
        self.vision = vision or VisionAgent(config_path)
        self.evidence = evidence or EvidenceAgent(int(self.config.get("evidence", {}).get("max_retrieval_cycles", 3)))
        self.report = report or ReportAgent(report_cfg)

        # Clinical context is part of the Evidence Agent
        self.clinical = clinical or getattr(self.evidence, "clinical_context", None) or ClinicalContextAgent()

        # Cognitive Agent Mind (LLM Reasoning)
        self._gemini_client = None
        self._llm_model = orchestrator.get("model", self._explainability_cfg.get("model", "gemini-2.5-flash"))
        self._llm_enabled = orchestrator.get("cognitive_reasoning", True)
        self._latest_thought: Optional[str] = None

    def _get_llm_client(self):
        """Lazily initialize Google GenAI Client for cognitive deliberation."""
        if not self._llm_enabled:
            return None
        if self._gemini_client is not None:
            return self._gemini_client

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        api_key = api_key.strip().strip('"').strip("'")
        if not api_key:
            return None

        try:
            from google import genai
            self._gemini_client = genai.Client(api_key=api_key)
            return self._gemini_client
        except Exception as err:
            logger.debug(f"Could not initialize GenAI Client for orchestrator: {err}")
            return None

    def _deliberate_with_mind(self, state: CaseState) -> Optional[Dict[str, Any]]:
        """
        Cognitive reasoning core of the Orchestrator.
        Uses Gemini to evaluate findings, identify clinical uncertainties,
        and formulate targeted evidence retrieval or stop criteria.
        """
        client = self._get_llm_client()
        if client is None:
            return None

        # Build clinical case observations
        diag = state.diagnosis_result
        pred = getattr(getattr(diag, "diagnosis", None), "prediction", "unknown") if diag else "unknown"
        conf = float(getattr(getattr(diag, "diagnosis", None), "confidence", 0.0) or 0.0) if diag else 0.0
        mel_prob = float(diag.probabilities.get("melanoma", 0.0)) if (diag and hasattr(diag, "probabilities")) else 0.0

        features = getattr(diag, "clinical_features", {}) or {}
        feat_summary = []
        for name in ("asymmetry", "border", "color", "diameter", "evolution"):
            f = features.get(name)
            if f:
                feat_summary.append(
                    f"{name.capitalize()}: {getattr(f, 'score_label', 'unknown')} "
                    f"(score: {getattr(f, 'score_numeric', 0.0):.2f})"
                )

        fusion = state.fusion_result
        fusion_summary = (
            f"Composite score: {fusion.composite_score:.2f}, Prediction: {fusion.prediction}"
            if fusion else "Not calculated"
        )

        evidence_titles = [f"'{e.title}' (rel: {e.relevance_score:.2f})" for e in state.evidence[:5]]
        cycles = state.evidence_assessment.cycles_completed
        max_c = self.evidence.max_cycles

        prompt = f"""You are the cognitive Orchestrator Agent for an AI Melanoma Diagnostic system.
You are in the Clinical Evidence Deliberation Phase.
Evaluate the diagnostic findings and determine if literature retrieval is needed or if evidence is sufficient to explain the case.

Patient & Diagnostic Findings:
- Model Prediction: {pred} (Confidence: {conf:.1f}%, Melanoma Probability: {mel_prob:.2f})
- Multimodal ABCDE Fusion: {fusion_summary}
- Clinical Features: {', '.join(feat_summary) if feat_summary else 'Extracted from image'}
- Attention Alignment: {state.uncertainty.classification:.2f}
- Evidence Cycles Completed: {cycles} of {max_c}
- Evidence Retrieved So Far: {', '.join(evidence_titles) if evidence_titles else 'None'}
- Clinical Context / Patient History: {state.clinical_context.get('patient_history', 'None provided')}

Guidelines for your decision:
1. If evidence cycles completed is 0, or if critical uncertainty/discrepancy exists between classifier and ABCD features and cycles < {max_c}:
   Choose action: "retrieve_evidence"
   Provide a targeted "evidence_query" addressing the specific diagnostic nuance (e.g. differential criteria, atypical presentation, specific ABCDE guidelines).
2. If the retrieved literature provides adequate clinical grounding for the findings, or cycles have exhausted:
   Choose action: "generate_explanation"
   Set "evidence_query": null

Respond in valid JSON only:
{{
  "thought": "<concise clinical reasoning explaining your observation and why you chose this action>",
  "action": "retrieve_evidence" or "generate_explanation",
  "reason": "<short label for the reason category>",
  "evidence_query": "<search query string or null>"
}}
"""
        try:
            resp = client.models.generate_content(
                model=self._llm_model,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            raw = (resp.text or "").strip()
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("action") in {"retrieve_evidence", "generate_explanation"}:
                return data
        except Exception as exc:
            logger.debug(f"LLM deliberation error: {exc}")
        return None

    def run(
        self,
        image_path: str,
        analysis_id: Optional[str] = None,
        save_mask: bool = False,
        scale_method: str = "auto",
        scale_reference_mm: Optional[float] = None,
        scale_reference_key: Optional[str] = None,
        clinical_context: Optional[Dict[str, Any]] = None,
        historical_images: Optional[List[str]] = None,
    ) -> CaseState:
        analysis_id = analysis_id or f"{Path(image_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        state = CaseState(
            analysis_id=analysis_id,
            image_path=str(image_path),
            status="running",
            historical_images=historical_images or [],
        )

        for _ in range(self.max_steps):
            self._latest_thought = None
            action, reason = self.choose_next_action(state)

            detail: Dict[str, Any] = {}
            if self._latest_thought:
                detail["thought"] = self._latest_thought
                detail["cognitive_mode"] = "gemini_guided_deliberation"
                print(f"\n[Orchestrator Mind] Thought: {self._latest_thought}")
                print(f"[Orchestrator Mind] Action: {action} ({reason})\n")

            state.record(self.name, action, "selected", reason, detail=detail)

            if action == "finish_case":
                state.status = "completed" if not state.error_code else "failed"
                state.record(self.name, action, "completed", "termination")
                return state

            try:
                if action == "validate_image":
                    self.vision.validate_image(state, self._quality_cfg)
                elif action in {
                    "classify_image",
                    "segment_lesion",
                    "calculate_measurements",
                    "calibrate_scale",
                    "run_gradcam",
                }:
                    self.vision.analyze(
                        state,
                        save_mask=save_mask,
                        scale_method=scale_method,
                        scale_reference_mm=scale_reference_mm,
                        scale_reference_key=scale_reference_key,
                    )
                elif action == "refine_and_reclassify":
                    self.vision.refine_and_reclassify(state)
                elif action == "assess_evolution":
                    ev_result = self.evidence.assess_evolution(
                        state.historical_images + [state.image_path],
                        config=self._explainability_cfg,
                    )
                    state.clinical_context["evolution"] = ev_result
                    if state.diagnosis_result and "evolution" in state.diagnosis_result.clinical_features:
                        ev_status = ev_result.get("status", "unavailable")
                        ev_confidence = float(ev_result.get("confidence", 0.0))
                        state.diagnosis_result.clinical_features["evolution"].score_label = (
                            ev_status.replace("_", " ").title()
                        )
                        state.diagnosis_result.clinical_features["evolution"].score_numeric = ev_confidence
                        state.diagnosis_result.clinical_features["evolution"].details = ev_result
                elif action == "request_clinical_context":
                    self.evidence.update_clinical_context(state, clinical_context)
                    if (
                        state.diagnosis_result
                        and "evolution" in state.clinical_context
                        and "evolution" in state.diagnosis_result.clinical_features
                    ):
                        state.diagnosis_result.clinical_features["evolution"].score_label = "Patient history provided"
                        state.diagnosis_result.clinical_features["evolution"].details = state.clinical_context.get(
                            "evolution", {}
                        )
                elif action == "apply_abcde_fusion":
                    mel_prob = state.diagnosis_result.probabilities.get("melanoma", 0.0)
                    threshold = getattr(state.diagnosis_result, "classification_threshold", None) or 0.5
                    fusion = compute_fusion(
                        melanoma_prob=mel_prob,
                        diagnosis_result=state.diagnosis_result,
                        abcde=state.abcde,
                        clinical_context=state.clinical_context,
                        classification_threshold=threshold,
                        weights=self._abcde_weights,
                    )
                    state.fusion_result = fusion
                    state.diagnosis_result.diagnosis.prediction = fusion.prediction
                    state.diagnosis_result.diagnosis.confidence = float(
                        fusion.composite_score * 100.0
                        if fusion.prediction.lower() == "melanoma"
                        else (1.0 - fusion.composite_score) * 100.0
                    )
                    state.classification.update({
                        "prediction": fusion.prediction,
                        "fusion_score": fusion.composite_score,
                        "fusion_weights": fusion.weights_used,
                        "fusion_components": fusion.component_scores,
                    })
                    state.flags.extend(fusion.flags)
                    print(
                        f"\n[ABCDE Fusion] Score={fusion.composite_score:.4f} Threshold={threshold:.4f} "
                        f"Prediction={fusion.prediction}"
                    )
                    print(f"[ABCDE Fusion] Weights={fusion.weights_used} Components={fusion.component_scores}\n")
                    state.record(
                        self.name,
                        "apply_abcde_fusion",
                        "completed",
                        "abcde_fusion",
                        {
                            "composite_score": fusion.composite_score,
                            "prediction": fusion.prediction,
                            "weights": fusion.weights_used,
                            "components": fusion.component_scores,
                        },
                        tool=True,
                    )
                elif action == "retrieve_evidence":
                    self.evidence.retrieve_evidence(state, reason)
                elif action == "generate_explanation":
                    self.report.generate_explanation(state)
                elif action == "generate_report":
                    self.report.generate_report(state)
                else:
                    raise RuntimeError(f"Unsupported orchestrator action: {action}")
            except Exception as exc:
                state.error_code, state.user_message = (
                    "workflow_action_failed",
                    f"The requested analysis action could not be completed. Action '{action}' failed: {str(exc)}",
                )
                state.flags.append(f"{action} failed: {type(exc).__name__}")
                state.record(
                    self.name,
                    action,
                    "failed",
                    "tool_failure",
                    {"exception": type(exc).__name__},
                    tool=True,
                )
                state.status = "failed"
                return state

        state.error_code, state.user_message, state.status = (
            "orchestrator_step_limit",
            "Analysis ended after reaching the safe workflow step limit.",
            "failed",
        )
        return state

    def choose_next_action(self, state: CaseState) -> Tuple[str, str]:
        """
        Hybrid Guided Reasoning policy:
        Combines mandatory clinical prerequisite ordering with LLM cognitive deliberation.
        """
        # ── 1. Mandatory Safety & Foundational Observations ────────────────
        if not state.image_quality:
            return "validate_image", "image_quality"
        if not state.image_quality.get("accepted"):
            state.error_code, state.user_message, state.retryable = (
                "image_quality_insufficient",
                state.image_quality.get("reason"),
                True,
            )
            return "finish_case", "image_quality"
        if state.diagnosis_result is None:
            return "classify_image", "vision_required"
        if self._should_adapt_from_gradcam(state):
            return "refine_and_reclassify", "gradcam_visual_misaligned"
        if state.historical_images and "evolution" not in state.clinical_context:
            return "assess_evolution", "historical_images_present"
        if "E" not in state.abcde:
            return "request_clinical_context", "missing_evolution"
        if state.fusion_result is None:
            return "apply_abcde_fusion", "abcde_ready"

        # ── 2. Cognitive Deliberation Phase (Orchestrator Mind) ─────────────
        if state.explanation is None:
            # Active deliberation via LLM if cycles remain
            if state.evidence_assessment.cycles_completed < self.evidence.max_cycles:
                mind_decision = self._deliberate_with_mind(state)
                if mind_decision:
                    thought = mind_decision.get("thought", "")
                    action = mind_decision.get("action")
                    reason = mind_decision.get("reason") or "cognitive_deliberation"
                    query = mind_decision.get("evidence_query")
                    self._latest_thought = thought

                    if action == "retrieve_evidence":
                        return "retrieve_evidence", query or reason
                    elif action == "generate_explanation":
                        state.evidence_assessment.sufficient = True
                        return "generate_explanation", "evidence_sufficient"

            # Deterministic clinical fallback if LLM is offline or unconfigured
            if state.evidence_assessment.cycles_completed == 0:
                return (
                    "retrieve_evidence",
                    "differential" if state.uncertainty.classification < self.confidence_threshold else "ABCDE",
                )
            if (
                not state.evidence_assessment.sufficient
                and state.evidence_assessment.cycles_completed < self.evidence.max_cycles
            ):
                return "retrieve_evidence", "ABCDE"
            return "generate_explanation", "validated_inputs_ready"

        # ── 3. Synthesis & Completion ──────────────────────────────────────
        if state.final_report is None:
            return "generate_report", "validated_synthesis_ready"
        return "finish_case", "report_ready"

    def _should_adapt_from_gradcam(self, state: CaseState) -> bool:
        """Evaluate Grad-CAM visual attention metrics to determine if reclassification is warranted."""
        if state.diagnosis_result is None:
            return False
        retry_count = state.gradcam.get("retry_count", 0)
        if retry_count >= self.max_retry_attempts:
            return False
        if state.gradcam and state.gradcam.get("alignment_reliable") is False:
            return True
        ail = state.gradcam.get("attention_inside_lesion")
        if ail is not None and ail < self.gradcam_threshold:
            return True
        return False

