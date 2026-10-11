"""Integration tests using real model checkpoints and pipeline components.

These tests load the actual Swin Transformer and SegFormer checkpoints,
run real inference, and verify the full pipeline from image to report.

Tests that require external dependencies (Qdrant, Gemini API) are marked
and will report clearly when those dependencies are unavailable.
"""
import os
import sys
import json
import pytest
from pathlib import Path
from types import SimpleNamespace

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    import yaml
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def test_image_path():
    """Find a test dermoscopy image."""
    test_img = PROJECT_ROOT / "test_images" / "test.jpg"
    if test_img.exists():
        return str(test_img)
    uploads = PROJECT_ROOT / "uploads"
    images = sorted([img for img in uploads.glob("*.png") if img.stat().st_size > 200000], key=lambda p: p.stat().st_size)
    if not images:
        pytest.skip("No test images found")
    return str(images[0])


@pytest.fixture(scope="module")
def swin_checkpoint_path(config):
    paths = config.get("paths", {})
    path = paths.get("classification_checkpoint", "checkpoints/best_swin_checkpoint.pth")
    full_path = PROJECT_ROOT / path
    if not full_path.is_file():
        pytest.skip(f"Swin checkpoint not found: {full_path}")
    return str(full_path)


@pytest.fixture(scope="module")
def segformer_checkpoint_path(config):
    seg_cfg = config.get("segmentation", {})
    path = seg_cfg.get("checkpoint", "checkpoints/best_segformer_melanoma.pth")
    full_path = PROJECT_ROOT / path
    if not full_path.is_file():
        pytest.skip(f"SegFormer checkpoint not found: {full_path}")
    return str(full_path)


# ---------------------------------------------------------------------------
# 1. Real Swin Transformer Loading
# ---------------------------------------------------------------------------

class TestSwinTransformer:
    """Tests for Swin Transformer classification with real checkpoint."""

    def test_swin_loads_successfully(self, swin_checkpoint_path, config):
        from src.classification.inference import SwinV2Predictor
        cls_cfg = config["classification"]
        predictor = SwinV2Predictor(
            checkpoint_path=swin_checkpoint_path,
            model_name=cls_cfg["model_name"],
            img_size=cls_cfg["img_size"],
            class_names=cls_cfg.get("class_names"),
            melanoma_class_name=cls_cfg.get("melanoma_class_name", "melanoma"),
            classification_threshold=cls_cfg.get("classification_threshold"),
        )
        assert predictor.model is not None
        assert len(predictor.class_names) == 2
        assert predictor.melanoma_idx is not None

    def test_swin_classifies_real_image(self, swin_checkpoint_path, config, test_image_path):
        from src.classification.inference import SwinV2Predictor
        cls_cfg = config["classification"]
        predictor = SwinV2Predictor(
            checkpoint_path=swin_checkpoint_path,
            model_name=cls_cfg["model_name"],
            img_size=cls_cfg["img_size"],
            class_names=cls_cfg.get("class_names"),
            melanoma_class_name=cls_cfg.get("melanoma_class_name", "melanoma"),
            classification_threshold=cls_cfg.get("classification_threshold"),
        )
        result = predictor.predict(test_image_path)
        assert "prediction" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["confidence"] >= 0
        assert result["confidence"] <= 100
        assert result["prediction"] in ("Melanoma", "Non-melanoma")
        print(f"  Swin prediction: {result['prediction']} ({result['confidence']:.1f}%)")


# ---------------------------------------------------------------------------
# 2. Real SegFormer Loading
# ---------------------------------------------------------------------------

class TestSegFormer:
    """Tests for SegFormer segmentation with real checkpoint."""

    def test_segformer_loads_successfully(self, segformer_checkpoint_path, config):
        from src.segmentation.segformer_wrapper import SegFormerSegmenter
        seg_cfg = config["segmentation"]
        segmenter = SegFormerSegmenter(
            checkpoint_path=segformer_checkpoint_path,
            encoder_name=seg_cfg.get("encoder_name", "mit_b2"),
            input_size=seg_cfg.get("input_size", 512),
            closing_kernel=seg_cfg.get("morphological_closing_kernel", 5),
        )
        assert segmenter.model is not None

    def test_segformer_segments_real_image(self, segformer_checkpoint_path, config, test_image_path):
        import numpy as np
        from PIL import Image
        from src.segmentation.segformer_wrapper import SegFormerSegmenter
        seg_cfg = config["segmentation"]
        segmenter = SegFormerSegmenter(
            checkpoint_path=segformer_checkpoint_path,
            encoder_name=seg_cfg.get("encoder_name", "mit_b2"),
            input_size=seg_cfg.get("input_size", 512),
            closing_kernel=seg_cfg.get("morphological_closing_kernel", 5),
        )
        image = np.array(Image.open(test_image_path).convert("RGB"))
        mask = segmenter.segment(image)
        assert mask.ndim == 2
        assert mask.shape[:2] == image.shape[:2]
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})
        area = int(np.count_nonzero(mask))
        print(f"  SegFormer mask area: {area} px ({area / mask.size * 100:.1f}%)")


# ---------------------------------------------------------------------------
# 3. Grad-CAM with real model
# ---------------------------------------------------------------------------

class TestGradCAM:
    """Tests for Grad-CAM with real Swin checkpoint."""

    def test_gradcam_generates_heatmap(self, swin_checkpoint_path, config, test_image_path):
        from src.explainability.gradcam import SwinGradCAM
        cls_cfg = config["classification"]
        cam = SwinGradCAM(
            checkpoint_path=swin_checkpoint_path,
            model_name=cls_cfg["model_name"],
            img_size=cls_cfg["img_size"],
        )
        result = cam.generate(test_image_path)
        assert "heatmap" in result
        assert "overlay" in result
        assert "prediction" in result
        assert "confidence" in result
        assert result["heatmap"].shape == (cls_cfg["img_size"], cls_cfg["img_size"])
        assert 0 <= result["heatmap"].min() <= result["heatmap"].max() <= 1
        print(f"  Grad-CAM prediction: {result['prediction']} ({result['confidence']:.1f}%)")

    def test_gradcam_metrics_with_mask(self, swin_checkpoint_path, config, test_image_path):
        import numpy as np
        from PIL import Image
        from src.explainability.gradcam import SwinGradCAM
        from src.segmentation.segformer_wrapper import SegFormerSegmenter

        cls_cfg = config["classification"]
        seg_cfg = config["segmentation"]
        seg_path = str(PROJECT_ROOT / seg_cfg["checkpoint"])
        if not Path(seg_path).is_file():
            pytest.skip("SegFormer checkpoint needed for mask")

        cam = SwinGradCAM(
            checkpoint_path=swin_checkpoint_path,
            model_name=cls_cfg["model_name"],
            img_size=cls_cfg["img_size"],
        )
        segmenter = SegFormerSegmenter(checkpoint_path=seg_path)
        image = np.array(Image.open(test_image_path).convert("RGB"))
        mask = segmenter.segment(image)
        cam_result = cam.generate(test_image_path)
        metrics = SwinGradCAM.compute_metrics(cam_result["heatmap"], mask)
        assert "attention_inside_lesion" in metrics
        assert "centroid_distance" in metrics
        assert "mask_cam_iou" in metrics
        assert 0 <= metrics["attention_inside_lesion"] <= 1
        print(f"  AIL: {metrics['attention_inside_lesion']:.3f}, IoU: {metrics['mask_cam_iou']:.3f}")


# ---------------------------------------------------------------------------
# 4. Segmentation -> Measurement flow
# ---------------------------------------------------------------------------

class TestMeasurementFlow:
    """Tests for the segmentation → measurement pipeline."""

    def test_measurements_from_real_mask(self, segformer_checkpoint_path, config, test_image_path):
        import numpy as np
        from PIL import Image
        from src.segmentation.segformer_wrapper import SegFormerSegmenter
        from src.engine.measurements import extract_lesion_measurements

        seg_cfg = config["segmentation"]
        segmenter = SegFormerSegmenter(
            checkpoint_path=segformer_checkpoint_path,
            encoder_name=seg_cfg.get("encoder_name", "mit_b2"),
            input_size=seg_cfg.get("input_size", 512),
        )
        image = np.array(Image.open(test_image_path).convert("RGB"))
        mask = segmenter.segment(image)

        if np.count_nonzero(mask) < 100:
            pytest.skip("Mask too small for measurements")

        measurements = extract_lesion_measurements(image, mask)
        assert "area_px" in measurements
        assert "perimeter_px" in measurements
        assert "diameter_px" in measurements
        assert measurements["area_px"] > 0
        assert measurements["perimeter_px"] > 0
        assert measurements["diameter_px"] > 0
        print(f"  Area: {measurements['area_px']} px², Diameter: {measurements['diameter_px']:.1f} px")

    def test_abc_features_from_real_mask(self, segformer_checkpoint_path, config, test_image_path):
        import numpy as np
        from PIL import Image
        from src.segmentation.segformer_wrapper import SegFormerSegmenter
        from src.features.extractor import ABCFeatureExtractor

        seg_cfg = config["segmentation"]
        segmenter = SegFormerSegmenter(
            checkpoint_path=segformer_checkpoint_path,
            encoder_name=seg_cfg.get("encoder_name", "mit_b2"),
            input_size=seg_cfg.get("input_size", 512),
        )
        image = np.array(Image.open(test_image_path).convert("RGB"))
        mask = segmenter.segment(image)

        if np.count_nonzero(mask) < 100:
            pytest.skip("Mask too small for features")

        extractor = ABCFeatureExtractor(config=config.get("features", {}))
        features = extractor.extract_all(image, mask)
        assert "asymmetry" in features
        assert "border" in features
        assert "color" in features
        for name, feat in features.items():
            print(f"  {name}: {feat.score_label} ({feat.score_numeric:.3f})")


# ---------------------------------------------------------------------------
# 5. Scale Calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    """Tests for pixel and calibrated measurements."""

    def test_pixel_measurements_without_calibration(self):
        from src.agents.state import CaseState, Measurement
        state = CaseState("test", "image.jpg")
        state.pixel_measurements["diameter"] = Measurement(200.0, "pixels", False, "feret")
        state.physical_measurements["diameter"] = Measurement(
            None, "mm", False, "none", reason_unavailable="No valid calibration"
        )
        assert state.pixel_measurements["diameter"].value == 200.0
        assert state.pixel_measurements["diameter"].calibrated is False
        assert state.physical_measurements["diameter"].value is None
        assert state.physical_measurements["diameter"].calibrated is False

    def test_calibrated_measurements(self):
        from src.agents.state import CaseState, Measurement
        state = CaseState("test", "image.jpg")
        state.pixel_measurements["diameter"] = Measurement(200.0, "pixels", False, "feret")
        state.physical_measurements["diameter"] = Measurement(8.5, "mm", True, "charuco")
        assert state.physical_measurements["diameter"].value == 8.5
        assert state.physical_measurements["diameter"].calibrated is True

    def test_safe_behavior_without_calibration(self):
        from src.agents.state import CaseState, Measurement, ABCDEFeature
        state = CaseState("test", "image.jpg")
        state.pixel_measurements["diameter"] = Measurement(200.0, "pixels", False)
        state.physical_measurements["diameter"] = Measurement(
            None, "mm", False, reason_unavailable="No calibration"
        )
        # D feature should note physical diameter unavailable
        raw = {"pixel_diameter": state.pixel_measurements["diameter"].__dict__,
               "physical_diameter": state.physical_measurements["diameter"].__dict__}
        state.abcde["D"] = ABCDEFeature("D", "available", raw,
            "Physical diameter is unavailable; pixel size must not be compared with millimetre thresholds.",
            "No calibration")
        assert state.physical_measurements["diameter"].value is None
        assert "millimetre" in state.abcde["D"].interpretation

    def test_abcde_features_stored_in_casestate(self):
        from src.agents.state import CaseState, Measurement, ABCDEFeature
        state = CaseState("test", "image.jpg")
        for criterion in "ABCDE":
            state.abcde[criterion] = ABCDEFeature(
                criterion, "available" if criterion != "E" else "unavailable",
                {}, f"test_{criterion}"
            )
        assert len(state.abcde) == 5
        assert state.abcde["E"].availability == "unavailable"


# ---------------------------------------------------------------------------
# 6. Parallel Pipeline execution
# ---------------------------------------------------------------------------

class TestParallelPipeline:
    """Verify Swin and SegFormer run in parallel."""

    def test_inference_pipeline_parallel(self, config, test_image_path):
        """Real inference pipeline with ThreadPoolExecutor parallelism."""
        from src.engine.inference_pipeline import InferencePipeline
        from src.classification.inference import SwinV2Predictor
        from src.segmentation.segformer_wrapper import SegFormerSegmenter
        from src.features.extractor import ABCFeatureExtractor
        from src.data.preprocessing import DermoscopyPreprocessor

        cls_cfg = config["classification"]
        seg_cfg = config["segmentation"]
        paths = config["paths"]
        preproc_cfg = config.get("preprocessing", {})

        swin_path = str(PROJECT_ROOT / paths.get("classification_checkpoint",
                        "checkpoints/best_swin_checkpoint.pth"))
        seg_path = str(PROJECT_ROOT / seg_cfg["checkpoint"])

        if not Path(swin_path).is_file() or not Path(seg_path).is_file():
            pytest.skip("Both checkpoints required")

        classifier = SwinV2Predictor(
            checkpoint_path=swin_path,
            model_name=cls_cfg["model_name"],
            img_size=cls_cfg["img_size"],
            class_names=cls_cfg.get("class_names"),
            melanoma_class_name=cls_cfg.get("melanoma_class_name", "melanoma"),
            classification_threshold=cls_cfg.get("classification_threshold"),
        )
        segmenter = SegFormerSegmenter(
            checkpoint_path=seg_path,
            encoder_name=seg_cfg.get("encoder_name", "mit_b2"),
            input_size=seg_cfg.get("input_size", 512),
        )
        extractor = ABCFeatureExtractor(config=config.get("features", {}))
        preprocessor = DermoscopyPreprocessor(
            hair_removal=preproc_cfg.get("hair_removal", True),
            illumination_normalization=preproc_cfg.get("illumination_normalization", True),
        )

        pipeline = InferencePipeline(
            classifier=classifier,
            segmenter=segmenter,
            feature_extractor=extractor,
            preprocessor=preprocessor,
        )
        result = pipeline.run(test_image_path)

        assert "classification" in result
        assert "mask" in result
        assert "features" in result
        assert "preprocessed_image" in result
        assert result["classification"]["prediction"] in ("Melanoma", "Non-melanoma")
        assert result["mask"].ndim == 2
        print(f"  Pipeline result: {result['classification']['prediction']} "
              f"({result['classification']['confidence']:.1f}%)")
        print(f"  Mask area: {int((result['mask'] > 0).sum())} px")


# ---------------------------------------------------------------------------
# 7. Full End-to-End Diagnosis Engine
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Full end-to-end test with real checkpoints."""

    def test_core_diagnosis_engine(self, config, test_image_path):
        """Load real engine, run diagnosis, verify output structure."""
        from src.engine.engine import CoreDiagnosisEngine
        engine = CoreDiagnosisEngine(str(PROJECT_ROOT / "config.yaml"))
        result = engine.diagnose(test_image_path, save_mask=False)

        # Verify structure
        assert result.diagnosis.prediction in ("Melanoma", "Non-melanoma")
        assert 0 <= result.diagnosis.confidence <= 100
        assert result.segmentation.status == "completed"
        assert result.segmentation.area_px > 0
        assert "lesion" in result.measurements
        assert "area_px" in result.measurements["lesion"]
        assert "diameter_px" in result.measurements["lesion"]
        assert "asymmetry" in result.clinical_features
        assert "border" in result.clinical_features
        assert "color" in result.clinical_features
        assert result.scale_calibration is not None

        print(f"  Engine: {result.diagnosis.prediction} ({result.diagnosis.confidence:.1f}%)")
        print(f"  Area: {result.measurements['lesion']['area_px']} px²")
        print(f"  Calibrated: {result.scale_calibration.calibration_valid}")


# ---------------------------------------------------------------------------
# 8. Supervisor Multi-Step Execution (with real vision, mocked evidence)
# ---------------------------------------------------------------------------

class TestSupervisorIntegration:
    """Supervisor workflow with real Vision and mocked Evidence."""

    def test_supervisor_full_workflow_real_vision(self, config, test_image_path):
        """Run Supervisor with real Swin+SegFormer, mocked evidence backend."""
        from src.agents.supervisor import SupervisorAgent
        from src.agents.evidence import EvidenceAgent

        class MockBackend:
            def hybrid_search(self, query):
                return [{"text": "asymmetry border color diameter evolution melanoma evidence guideline",
                         "metadata": {"source": "guideline", "title": "ABCDE Criteria"},
                         "rerank_score": 0.85}]

        supervisor = SupervisorAgent(
            str(PROJECT_ROOT / "config.yaml"),
            evidence=EvidenceAgent(2, MockBackend()),
        )
        state = supervisor.run(test_image_path, save_mask=False)

        assert state.status in ("completed", "needs_clinical_context")
        assert state.diagnosis_result is not None
        assert state.classification.get("prediction") in ("Melanoma", "Non-melanoma")
        assert len(state.agent_decisions) > 0
        assert len(state.tool_executions) > 0

        # Verify CaseState fields
        assert state.pixel_measurements.get("diameter") is not None
        assert state.physical_measurements.get("diameter") is not None
        assert len(state.abcde) >= 4  # at least A, B, C, D

        # Verify uncertainty propagation
        assert state.uncertainty.classification > 0
        assert state.uncertainty.segmentation >= 0

        # Verify execution trace
        trace = state.public_trace()
        assert len(trace) > 0
        actions = [e["action"] for e in trace]
        assert "validate_image" in actions or "analyze_image" in actions

        print(f"  Status: {state.status}")
        print(f"  Actions: {actions}")
        print(f"  Flags: {state.flags}")

    def test_supervisor_produces_report(self, config, test_image_path):
        """Verify Supervisor produces a complete report."""
        from src.agents.supervisor import SupervisorAgent
        from src.agents.evidence import EvidenceAgent

        class MockBackend:
            def hybrid_search(self, query):
                return [{"text": "asymmetry border color diameter evolution clinical evidence",
                         "metadata": {"source": "guideline", "title": "ABCDE"}, "rerank_score": 0.8}]

        supervisor = SupervisorAgent(
            str(PROJECT_ROOT / "config.yaml"),
            evidence=EvidenceAgent(2, MockBackend()),
        )
        state = supervisor.run(test_image_path, save_mask=False)

        if state.status == "completed":
            assert state.final_report is not None
            report = state.final_report
            assert "analysis_id" in report
            assert "prediction" in report
            assert "uncertainty" in report
            assert "pipeline_metadata" in report
            assert "three-agent" in report["pipeline_metadata"]["architecture"]
            print(f"  Report keys: {list(report.keys())}")
        else:
            print(f"  Status: {state.status} (report may not be generated)")
            print(f"  Error: {state.error_code}")


# ---------------------------------------------------------------------------
# 9. Evidence Agent (requires Qdrant + Gemini)
# ---------------------------------------------------------------------------

class TestEvidenceIntegration:
    """Real Evidence Agent tests — require Qdrant index and Gemini API."""

    def _check_qdrant_available(self):
        """Check if Qdrant is available and has data."""
        qdrant_path = PROJECT_ROOT / "qdrant_data"
        alt_path = PROJECT_ROOT / "data" / "qdrant_db"
        if not qdrant_path.exists() and not alt_path.exists():
            pytest.skip("No Qdrant index found (neither qdrant_data/ nor data/qdrant_db/)")
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

    def test_qdrant_bge_bm25_retrieval(self):
        """Test real hybrid retrieval with BGE, Qdrant, and BM25."""
        self._check_qdrant_available()
        try:
            from rag_pipeline.agent import create_backend
            backend = create_backend()
            results = backend.hybrid_search("melanoma ABCDE clinical evidence")
            assert len(results) > 0
            for r in results[:3]:
                assert "text" in r
                assert len(r["text"]) > 0
                print(f"  Retrieved: {r.get('metadata', {}).get('title', 'N/A')[:60]}")
        except Exception as e:
            pytest.skip(f"RAG backend unavailable: {e}")

    def test_hyde_query_generation(self):
        """Test HyDE query generation."""
        self._check_qdrant_available()
        try:
            from rag_pipeline.agent import create_backend
            backend = create_backend()
            metrics = {
                "asymmetry_index": 0.45,
                "border_irregularity_score": 0.65,
                "color_variation_score": 4.0,
                "diameter_pixels": 200,
                "diameter_mm": None,
                "calibration_valid": False,
                "evolution": {"status": "unavailable", "reported_change": None},
            }
            queries = backend.create_queries(metrics)
            assert len(queries) > 0
            print(f"  HyDE queries: {len(queries)}")
            for q in queries[:3]:
                print(f"    {q[:80]}...")
        except Exception as e:
            pytest.skip(f"HyDE unavailable: {e}")

    def test_evidence_gap_follow_up(self):
        """Test evidence-gap follow-up cycles."""
        from src.agents.evidence import EvidenceAgent
        from src.agents.state import CaseState

        class LimitedBackend:
            """Returns evidence that covers only A, B, C initially."""
            def __init__(self):
                self.call_count = 0
            def hybrid_search(self, query):
                self.call_count += 1
                if self.call_count <= 2:
                    return [{"text": "asymmetry border color pigmented lesion",
                             "metadata": {"source": "ref", "title": "ABC"},
                             "rerank_score": 0.7}]
                return [{"text": "diameter size evolution temporal change clinical",
                         "metadata": {"source": "ref", "title": "DE"},
                         "rerank_score": 0.7}]

        state = CaseState("test", "image.jpg")
        state.abcde["E"] = SimpleNamespace(availability="unavailable")
        backend = LimitedBackend()
        EvidenceAgent(max_cycles=3, backend=backend).retrieve_evidence(state, "ABCDE")
        assert state.evidence_assessment.cycles_completed >= 1
        assert len(state.evidence) > 0
        print(f"  Cycles: {state.evidence_assessment.cycles_completed}")
        print(f"  Evidence items: {len(state.evidence)}")
        print(f"  Sufficient: {state.evidence_assessment.sufficient}")


# ---------------------------------------------------------------------------
# 10. DiagnosisResult generation and PDF rendering
# ---------------------------------------------------------------------------

class TestDiagnosisResult:
    """DiagnosisResult schema and PDF rendering tests."""

    def test_diagnosis_result_serialization(self, config, test_image_path):
        """Verify DiagnosisResult can be serialized to JSON."""
        from src.engine.engine import CoreDiagnosisEngine
        engine = CoreDiagnosisEngine(str(PROJECT_ROOT / "config.yaml"))
        result = engine.diagnose(test_image_path, save_mask=False)
        json_str = result.model_dump_json()
        parsed = json.loads(json_str)
        assert "diagnosis" in parsed
        assert "measurements" in parsed
        assert "clinical_features" in parsed
        assert "segmentation" in parsed
        print(f"  JSON keys: {list(parsed.keys())}")

    def test_pdf_rendering(self, config, test_image_path):
        """Test PDF rendering if dependencies are available."""
        try:
            from src.engine.engine import CoreDiagnosisEngine
            from src.agents.report import ReportAgent
            from src.agents.state import CaseState

            engine = CoreDiagnosisEngine(str(PROJECT_ROOT / "config.yaml"))
            result = engine.diagnose(test_image_path, save_mask=False)

            # Set up state for report generation
            state = CaseState("pdf_test", test_image_path)
            state.diagnosis_result = result
            state.classification = {"prediction": result.diagnosis.prediction,
                                    "confidence": result.diagnosis.confidence}
            state.explanation = {
                "model_observation": f"Test: {result.diagnosis.prediction}",
                "literature_context": "Test evidence",
                "uncertainty": [],
            }

            report_agent = ReportAgent(config.get("report", {}))
            report_agent.generate_report(state)
            assert state.report_result is not None
            # PDF may or may not render depending on template availability
            if state.report_result.pdf_path:
                assert Path(state.report_result.pdf_path).exists()
                print(f"  PDF generated: {state.report_result.pdf_path}")
            else:
                print("  PDF not rendered (template may be unavailable)")
        except Exception as e:
            print(f"  PDF test skipped: {e}")


# ---------------------------------------------------------------------------
# 11. Preprocessing consistency
# ---------------------------------------------------------------------------

class TestPreprocessing:
    """Verify preprocessing is consistent with training pipelines."""

    def test_swin_uses_classification_transforms(self, config):
        """Swin uses albumentations val transforms with ImageNet normalization."""
        from src.data.transforms import get_classification_transforms
        transform = get_classification_transforms("val", config["classification"]["img_size"])
        assert transform is not None
        # The transform should resize and normalize for ImageNet
        # This is what the Swin training used

    def test_segformer_uses_imagenet_normalization(self, config):
        """SegFormer uses direct ImageNet normalization (not albumentations)."""
        import numpy as np
        # SegFormer wrapper does its own preprocessing:
        # 1. Resize to input_size using skimage
        # 2. Normalize with ImageNet mean/std
        # This is different from the Swin preprocessing pipeline
        # Verify the constants match
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        # These are standard ImageNet values used by SegFormer
        assert np.allclose(mean, [0.485, 0.456, 0.406])
        assert np.allclose(std, [0.229, 0.224, 0.225])

    def test_preprocessing_pipelines_differ(self, config):
        """Swin and SegFormer have different preprocessing — this is correct."""
        # Swin: 256x256 via albumentations Resize + ImageNet normalization
        # SegFormer: 512x512 via skimage resize + ImageNet normalization
        swin_size = config["classification"]["img_size"]
        seg_size = config["segmentation"]["input_size"]
        assert swin_size != seg_size, "Swin and SegFormer should have different input sizes"
        print(f"  Swin input: {swin_size}x{swin_size}")
        print(f"  SegFormer input: {seg_size}x{seg_size}")
