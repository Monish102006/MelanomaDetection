"""Architecture tests for the three-agent + Orchestrator melanoma workflow.

These tests verify the corrected architecture:
- Three top-level agents: Vision, Evidence, Report
- Orchestrator orchestrates the three agents
- Clinical context is an internal utility, not a top-level agent
- CaseState is the central contract
"""
from pathlib import Path
from types import SimpleNamespace

from src.agents.clinical_context import ClinicalContextAgent
from src.agents.evidence import EvidenceAgent
from src.agents.report import ReportAgent
from src.agents.state import CaseState, EvidenceRecord, Measurement
from src.agents.orchestrator import OrchestratorAgent
from src.agents.vision import VisionAgent


class FakeVision:
    name = "vision"
    def validate_image(self, state, quality_config):
        state.image_quality = {"accepted": True, "reason": ""}
        return state
    def analyze(self, state, **kwargs):
        state.diagnosis_result = SimpleNamespace(
            model_dump=lambda **_: {"validated": True},
            diagnosis=SimpleNamespace(prediction="melanoma", confidence=62.0),
            probabilities={"melanoma": 0.62},
            clinical_features={},
            measurements={"lesion": {"diameter_px": 184, "area_px": 100, "perimeter_px": 44}},
            explainability=SimpleNamespace(attention_inside_lesion=0.8),
        )
        state.classification = {"prediction": "melanoma", "confidence": 62.0, "probabilities": {"melanoma": .62}}
        state.uncertainty.classification = .62
        state.pixel_measurements["diameter"] = Measurement(184, "pixels", False)
        state.physical_measurements["diameter"] = Measurement(None, "mm", False, reason_unavailable="no scale")
        state.abcde.update({key: SimpleNamespace(availability="available") for key in "ABCD"})
        return state


class Backend:
    def __init__(self): self.calls = []
    def hybrid_search(self, query):
        self.calls.append(query)
        return [{"text": "asymmetry border color diameter evolution clinical evidence", "metadata": {"source": "guideline", "title": "ABCDE"}, "rerank_score": .8}]


# ── Clinical Context (internal utility, not a top-level agent) ──

def test_clinical_context_never_fabricates_evolution():
    state = CaseState("id", "image.jpg")
    ClinicalContextAgent().update(state)
    assert state.abcde["E"].availability == "unavailable"
    assert state.pending_clinical_questions[0]["field"] == "evolution"
    assert "single_timepoint_capture" not in str(state.abcde["E"].raw)


# ── Evidence Agent ──

def test_evidence_agent_is_goal_driven_and_bounded():
    backend = Backend()
    state = CaseState("id", "image.jpg")
    state.abcde["E"] = SimpleNamespace(availability="unavailable")
    EvidenceAgent(max_cycles=2, backend=backend).retrieve_evidence(state, "ABCDE")
    assert state.evidence_assessment.sufficient
    assert len(backend.calls) >= 1
    assert all(item.trust_boundary == "UNTRUSTED_EVIDENCE" for item in state.evidence)


# ── Measurement safety ──

def test_pixel_measurement_cannot_be_promoted_to_mm_or_threshold():
    state = CaseState("id", "image.jpg")
    state.pixel_measurements["diameter"] = Measurement(184, "pixels", False)
    state.physical_measurements["diameter"] = Measurement(None, "mm", False, reason_unavailable="no calibration")
    state.abcde["E"] = SimpleNamespace(availability="unavailable")
    queries = EvidenceAgent(backend=Backend())._plan_queries(state, "measurement")
    assert not any("184" in query or "6 mm" in query for query in queries)
    assert state.physical_measurements["diameter"].value is None


# ── Vision Agent ──

def test_vision_maps_model_segmentation_gradcam_and_calibration_safely():
    result = SimpleNamespace(
        diagnosis=SimpleNamespace(prediction="melanoma", confidence=81.0),
        probabilities={"melanoma": .81},
        segmentation=SimpleNamespace(status="completed", area_px=100, perimeter_px=44, mask_path=None),
        measurements={"lesion": {"area_px": 100, "diameter_px": 184, "measurement_method": "feret"}, "asymmetry": {}, "border": {}, "color": {}},
        clinical_features={key: SimpleNamespace(score_label="observed") for key in ("asymmetry", "border", "color")},
        explainability=SimpleNamespace(model_dump=lambda: {"attention_inside_lesion": .9}),
        scale_calibration=SimpleNamespace(model_dump=lambda: {"calibration_valid": False, "calibration_reason": "no reference", "calibration_confidence": 0}),
    )
    engine = SimpleNamespace(diagnose=lambda *args, **kwargs: result)
    state = VisionAgent(engine=engine).analyze(CaseState("id", "image.jpg"), save_mask=False)
    assert state.classification["confidence"] == 81.0
    assert state.segmentation["mask_consistency"] == 1.0
    assert state.gradcam["alignment_reliable"]
    assert state.physical_measurements["diameter"].value is None
    assert "millimetre thresholds" in state.abcde["D"].interpretation


# ── Orchestrator workflow ──

def test_orchestrator_conditional_routing_and_report_output(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("quality: {}\norchestrator: {classification_confidence_threshold: 70, max_orchestrator_steps: 12}\nevidence: {max_retrieval_cycles: 2}\nreport: {}\n")
    orchestrator = OrchestratorAgent(str(config), vision=FakeVision(), evidence=EvidenceAgent(2, Backend()), report=ReportAgent())
    state = orchestrator.run("image.jpg")
    actions = [event.action for event in state.agent_decisions]
    assert state.status == "completed"
    assert "request_clinical_context" in actions
    assert "retrieve_evidence" in actions
    assert state.final_report["physical_measurements"]["diameter"]["value"] is None
    assert state.final_report["pending_clinical_questions"]
    assert state.final_report["pipeline_metadata"]["trace"]


# ── Report Agent ──

def test_report_requires_validated_state_and_propagates_uncertainty():
    state = CaseState("id", "image.jpg")
    state.classification = {"prediction": "unknown", "confidence": 0}
    state.explanation = {"model_observation": "unavailable", "literature_context": "none", "uncertainty": ["missing evidence"]}
    ReportAgent().generate_report(state)
    assert state.final_report["uncertainty"]["final_synthesis"] == 1.0
    assert state.report_result.dashboard_payload["explanation"]["uncertainty"] == ["missing evidence"]


# ── Architecture correctness ──

def test_three_agent_architecture():
    """Verify the corrected architecture has exactly three top-level agents."""
    import src.agents as agents_module
    # __all__ should export exactly 4 names: Orchestrator + 3 agents
    assert set(agents_module.__all__) == {"OrchestratorAgent", "VisionAgent", "EvidenceAgent", "ReportAgent"}


def test_report_metadata_says_three_agent():
    """Report metadata should say three-agent, not five-agent."""
    state = CaseState("id", "image.jpg")
    state.classification = {"prediction": "test", "confidence": 50}
    state.explanation = {"model_observation": "test", "literature_context": "test", "uncertainty": []}
    ReportAgent().generate_report(state)
    assert "three-agent" in state.final_report["pipeline_metadata"]["architecture"]


def test_clinical_context_not_in_top_level_all():
    """ClinicalContextAgent should be importable but not in __all__."""
    import src.agents as agents_module
    # Should be importable (it's used internally by Orchestrator)
    assert hasattr(agents_module, "ClinicalContextAgent")
    # But NOT in __all__ (it's not a top-level agent)
    assert "ClinicalContextAgent" not in agents_module.__all__


def test_orchestrator_uses_clinical_context_internally():
    """Orchestrator should use clinical context functionality internally."""
    import src.agents.orchestrator as sup_module
    # The orchestrator imports and uses ClinicalContextAgent as an internal utility
    assert hasattr(sup_module, "ClinicalContextAgent")


# ── Grad-CAM Visual Guided Reclassification ──

class MisalignedFakeVision(FakeVision):
    """Vision mock simulating misaligned initial Grad-CAM visual attention."""
    def __init__(self):
        self.reclassified = False

    def analyze(self, state, **kwargs):
        state = super().analyze(state, **kwargs)
        state.gradcam = {
            "attention_inside_lesion": 0.15,
            "attention_outside_lesion": 0.85,
            "alignment_reliable": False,
            "grad_cam_path": "outputs/gradcam_samples/test_initial_gradcam.png",
        }
        state.grad_cam_saved_path = "outputs/gradcam_samples/test_initial_gradcam.png"
        return state

    def refine_and_reclassify(self, state):
        self.reclassified = True
        state.classification["prediction"] = "melanoma"
        state.classification["confidence"] = 88.0
        state.gradcam.update({
            "attention_inside_lesion": 0.82,
            "attention_outside_lesion": 0.18,
            "alignment_reliable": True,
            "grad_cam_path": "outputs/gradcam_samples/test_refined_gradcam.png",
            "retry_count": state.gradcam.get("retry_count", 0) + 1,
            "adaptation_applied": "roi_crop_and_artifact_cleanup",
        })
        state.grad_cam_saved_path = "outputs/gradcam_samples/test_refined_gradcam.png"
        state.flags.append("Grad-CAM visual guided reclassification: focus improved.")
        state.record(self.name, "refine_and_reclassify", "improved", "gradcam_visual_adaptation", tool=True)
        return state


def test_orchestrator_reclassifies_on_misaligned_gradcam(tmp_path):
    """Verify Orchestrator detects misaligned Grad-CAM visual and triggers refine_and_reclassify."""
    config = tmp_path / "config.yaml"
    config.write_text("quality: {}\norchestrator: {classification_confidence_threshold: 70, gradcam_reliability_threshold: 0.30, max_retry_attempts: 1, max_orchestrator_steps: 12}\nevidence: {max_retrieval_cycles: 2}\nreport: {}\n")
    vision = MisalignedFakeVision()
    orchestrator = OrchestratorAgent(str(config), vision=vision, evidence=EvidenceAgent(2, Backend()), report=ReportAgent())
    state = orchestrator.run("image.jpg")

    actions = [event.action for event in state.agent_decisions]
    assert "refine_and_reclassify" in actions
    assert vision.reclassified is True
    assert state.gradcam["alignment_reliable"] is True
    assert state.gradcam["attention_inside_lesion"] == 0.82
    assert state.grad_cam_saved_path == "outputs/gradcam_samples/test_refined_gradcam.png"
    assert state.status == "completed"


def test_orchestrator_respects_max_retry_attempts_for_gradcam(tmp_path):
    """Verify Orchestrator does not loop indefinitely if Grad-CAM remains unreliable."""
    class AlwaysUnreliableVision(FakeVision):
        def __init__(self):
            self.retry_count = 0
        def analyze(self, state, **kwargs):
            state = super().analyze(state, **kwargs)
            state.gradcam = {"attention_inside_lesion": 0.10, "alignment_reliable": False}
            return state
        def refine_and_reclassify(self, state):
            self.retry_count += 1
            state.gradcam["retry_count"] = state.gradcam.get("retry_count", 0) + 1
            state.gradcam["alignment_reliable"] = False
            state.record(self.name, "refine_and_reclassify", "not_improved", "gradcam_visual_adaptation", tool=True)
            return state

    config = tmp_path / "config.yaml"
    config.write_text("quality: {}\norchestrator: {classification_confidence_threshold: 70, gradcam_reliability_threshold: 0.30, max_retry_attempts: 1, max_orchestrator_steps: 12}\nevidence: {max_retrieval_cycles: 2}\nreport: {}\n")
    vision = AlwaysUnreliableVision()
    orchestrator = OrchestratorAgent(str(config), vision=vision, evidence=EvidenceAgent(2, Backend()), report=ReportAgent())
    state = orchestrator.run("image.jpg")

    assert vision.retry_count == 1
    assert state.status == "completed"


def test_outputs_not_stored_but_gradcam_visual_stored(tmp_path):
    """Verify run outputs (JSON files, masks) are not stored in local storage, but Grad-CAM visuals are."""
    outputs_dir = tmp_path / "outputs"
    gradcam_samples_dir = outputs_dir / "gradcam_samples"
    gradcam_samples_dir.mkdir(parents=True, exist_ok=True)
    mock_gradcam_file = gradcam_samples_dir / "test_case_gradcam.png"
    mock_gradcam_file.write_bytes(b"\x89PNG\r\n\x1a\nfake_image_bytes")

    class GradcamSavingVision(FakeVision):
        def analyze(self, state, **kwargs):
            state = super().analyze(state, **kwargs)
            state.grad_cam_saved_path = str(mock_gradcam_file)
            state.gradcam["grad_cam_path"] = str(mock_gradcam_file)
            return state

    config = tmp_path / "config.yaml"
    config.write_text(f"quality: {{}}\npaths: {{outputs: {outputs_dir.as_posix()}}}\norchestrator: {{classification_confidence_threshold: 70, max_orchestrator_steps: 12}}\nevidence: {{max_retrieval_cycles: 2}}\nreport: {{save_pdf: false}}\n")
    orchestrator = OrchestratorAgent(str(config), vision=GradcamSavingVision(), evidence=EvidenceAgent(2, Backend()), report=ReportAgent())
    state = orchestrator.run("image.jpg", analysis_id="test_case")

    assert state.status == "completed"
    # Ensure Grad-CAM visual is stored and tracked
    assert state.grad_cam_saved_path == str(mock_gradcam_file)
    assert Path(state.grad_cam_saved_path).exists()

    # Ensure no JSON run dumps were saved in outputs directory
    json_files = list(outputs_dir.glob("*.json"))
    assert len(json_files) == 0, f"Expected no JSON run files in outputs, found: {json_files}"

    # Ensure no masks directory or masks were written
    masks_dir = outputs_dir / "masks"
    mask_files = list(masks_dir.glob("*")) if masks_dir.exists() else []
    assert len(mask_files) == 0, f"Expected no masks saved, found: {mask_files}"


def test_explainability_retries_and_falls_back_to_lite_model(monkeypatch):
    """Verify explainability service retries up to 5 times, switches to lite model after 3 failures."""
    from unittest.mock import MagicMock
    from src.agents.explainability import EvidenceGroundedExplanationService

    service = EvidenceGroundedExplanationService(
        config={
            "mode": "llm",
            "model": "gemini-3.8-flash",
            "fallback_model": "gemini-3.1-flash-lite",
            "max_retries": 5,
            "retry_delay": 0.01,  # fast test
            "fallback_after_retries": 3,
        }
    )

    models_called = []

    class FakeChat:
        def __init__(self, model):
            self.model = model

        def send_message(self, prompt):
            models_called.append(self.model)
            if len(models_called) <= 3:
                raise RuntimeError(f"Service unavailable on {self.model}")
            mock_resp = MagicMock()
            mock_resp.text = "**1. WHAT THIS MEANS**\n\nAI estimate indicating low risk.\n\n**2. WHAT THE IMAGE SHOWED**\n\nLesion features normal."
            return mock_resp

    mock_client = MagicMock()
    mock_client.chats.create = lambda model: FakeChat(model)
    monkeypatch.setattr(service, "_get_gemini_model", lambda: mock_client)

    diag_result = SimpleNamespace(
        diagnosis=SimpleNamespace(prediction="Benign", confidence=90.0),
        probabilities={"melanoma": 0.1, "non_melanoma": 0.9},
        clinical_features={},
        measurements={},
        segmentation=SimpleNamespace(area_px=100, perimeter_px=50),
        explainability=SimpleNamespace(attention_inside_lesion=0.8, mask_cam_iou=0.6),
    )

    result = service.explain(diag_result, evidence=[])

    assert len(models_called) == 4
    # First 3 attempts should be primary model
    assert models_called[:3] == ["gemini-3.8-flash", "gemini-3.8-flash", "gemini-3.8-flash"]
    # 4th attempt should switch to fallback model
    assert models_called[3] == "gemini-3.1-flash-lite"
    assert "llm_model_fallback:gemini-3.1-flash-lite" in result.flags
    assert "llm_retry_success:attempt_4" in result.flags


def test_explainability_falls_back_to_template_after_max_retries(monkeypatch):
    """Verify explainability service falls back to template mode when all 5 retries fail."""
    from unittest.mock import MagicMock
    from src.agents.explainability import EvidenceGroundedExplanationService

    service = EvidenceGroundedExplanationService(
        config={
            "mode": "llm",
            "model": "gemini-3.8-flash",
            "fallback_model": "gemini-3.1-flash-lite",
            "max_retries": 5,
            "retry_delay": 0.01,
            "fallback_after_retries": 3,
        }
    )

    models_called = []

    class FakeChat:
        def __init__(self, model):
            self.model = model

        def send_message(self, prompt):
            models_called.append(self.model)
            raise RuntimeError(f"Server error on {self.model}")

    mock_client = MagicMock()
    mock_client.chats.create = lambda model: FakeChat(model)
    monkeypatch.setattr(service, "_get_gemini_model", lambda: mock_client)

    diag_result = SimpleNamespace(
        diagnosis=SimpleNamespace(prediction="Benign", confidence=85.0),
        probabilities={"melanoma": 0.15, "non_melanoma": 0.85},
        clinical_features={},
        measurements={},
        segmentation=SimpleNamespace(area_px=100, perimeter_px=50),
        explainability=SimpleNamespace(attention_inside_lesion=0.8, mask_cam_iou=0.6),
    )

    result = service.explain(diag_result, evidence=[])

    # All 5 attempts should have been made
    assert len(models_called) == 5
    assert models_called == [
        "gemini-3.8-flash",
        "gemini-3.8-flash",
        "gemini-3.8-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash-lite",
    ]
    # Should fall back to template mode
    assert "llm_fallback_to_template" in result.flags
    assert len(result.reasoning) > 0
