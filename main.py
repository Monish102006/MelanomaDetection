"""
AI Melanoma Diagnosis Pipeline
================================

Entry point for running all pipeline components.

Usage:
    python main.py prepare-data       # Prepare ISIC 2019/2018 datasets
    python main.py validate-data      # Validate dataset quality
    python main.py train              # Train Swin V2 classifier
    python main.py evaluate-cls       # Evaluate classifier on test set
    python main.py evaluate-seg       # Evaluate segmentation
    python main.py diagnose <image>   # Run deterministic vision tools only
    python main.py orchestrate <image># Run the canonical five-agent workflow
    python main.py gradcam <image>    # Generate Grad-CAM visualization
    python main.py serve              # Start FastAPI server
    python main.py build-rag          # Build Qdrant + BM25 hybrid index
    python main.py evaluate-rag       # Run ABCDE RAG evaluation suite
"""

import sys
import logging
from pathlib import Path


def print_usage():
    print(__doc__)
    sys.exit(1)


def main():
    import os
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

    if len(sys.argv) < 2:
        print_usage()

    command = sys.argv[1]

    if command == "prepare-data":
        from src.data.prepare import prepare_all
        # Interactive prompts for paths
        print("=== ISIC Dataset Preparation ===\n")
        i19_img = input("Path to extracted ISIC 2019 images folder: ").strip()
        i19_csv = input("Path to ISIC_2019_Training_GroundTruth.csv: ").strip()
        i18_img = input("Path to extracted ISIC 2018 Task1-2 images folder: ").strip()
        i18_masks = input("Path to extracted ISIC 2018 Task1 masks folder: ").strip()
        prepare_all(i19_img, i19_csv, i18_img, i18_masks)

    elif command == "train":
        from src.classification.train import train_classifier
        config = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
        train_classifier(config)

    elif command == "evaluate-cls":
        from src.evaluation.classification_eval import run_classification_evaluation
        config = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
        run_classification_evaluation(config)

    elif command == "evaluate-seg":
        from src.evaluation.segmentation_eval import run_segmentation_evaluation
        config = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
        run_segmentation_evaluation(config)

    elif command == "diagnose":
        if len(sys.argv) < 3:
            print("Usage: python main.py diagnose <image_path>")
            sys.exit(1)
        from src.engine.engine import CoreDiagnosisEngine
        engine = CoreDiagnosisEngine("config.yaml")
        result = engine.diagnose(sys.argv[2], save_mask=False)
        print(result.to_summary())
        if result.grad_cam_path:
            print(f"\n  Grad-CAM Visual saved: {result.grad_cam_path}")

    elif command == "orchestrate":
        if len(sys.argv) < 3:
            print("Usage: python main.py orchestrate <image_path>")
            sys.exit(1)

        # Configure logging for verbose orchestrator output
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        from src.agents.orchestrator import OrchestratorAgent

        image_path = sys.argv[2]
        print(f"\n{'='*60}")
        print(f"  Three-Agent Melanoma Case Workflow")
        print(f"  Image: {image_path}")
        print(f"{'='*60}\n")

        supervisor = OrchestratorAgent("config.yaml")
        state = supervisor.run(image_path, save_mask=False)

        # Print results summary
        print(f"\n{'='*60}")
        print(f"  Pipeline Status: {state.overall_status.upper()}")
        print(f"{'='*60}")

        for event in state.public_trace():
            print(f"  • {event['agent']} → {event['action']}: {event['result']} ({event['reason_category']})")

        if state.diagnosis_result:
            diag = state.diagnosis_result
            print(f"\n  Diagnosis: {diag.diagnosis.prediction} ({diag.diagnosis.confidence:.1f}%)")
            for name, feat in diag.clinical_features.items():
                print(f"    {name}: {feat.score_label} ({feat.score_numeric:.3f})")

        if state.grad_cam_saved_path:
            print(f"\n  Grad-CAM Visual saved: {state.grad_cam_saved_path}")

        if state.retrieved_evidence:
            print(f"\n  Evidence: {len(state.retrieved_evidence)} items retrieved")
            for ev in state.retrieved_evidence[:3]:
                print(f"    • {ev.title} ({ev.source}) — relevance: {ev.relevance_score:.3f}")

        if state.explanation_result:
            expl = state.explanation_result
            summary = expl.summary if hasattr(expl, "summary") else str(expl)
            print(f"\n  Explanation: {summary}")

        if state.report_result:
            print(f"\n  Report PDF: {state.report_result.pdf_path or 'not generated'}")

        if state.flags:
            print(f"\n  ⚠ Flags:")
            for flag in state.flags:
                print(f"    • {flag}")

        print()

    elif command == "gradcam":
        if len(sys.argv) < 3:
            print("Usage: python main.py gradcam <image_path>")
            sys.exit(1)
        from src.explainability.gradcam import SwinGradCAM
        import yaml
        config_file = Path("config.yaml").resolve()
        with open(config_file) as _f:
            _cfg = yaml.safe_load(_f)
        _paths = _cfg["paths"]
        def _resolve(value, default):
            path = Path(value or default)
            return path if path.is_absolute() else config_file.parent / path
        _ckpt = _resolve(
            _paths.get("classification_checkpoint"),
            str(Path("checkpoints") / "best_swin_checkpoint.pth"),
        )
        cam = SwinGradCAM(str(_ckpt))
        cam.save_visualization(
            sys.argv[2], str(_resolve(_paths.get("outputs"), "outputs") / "gradcam_samples")
        )

    elif command == "validate-data":
        from src.data.validation import (
            validate_classification_dataset,
            validate_segmentation_dataset,
            save_validation_report,
        )
        import yaml
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        paths = config["paths"]

        print("Validating classification dataset...")
        cls_report = validate_classification_dataset(paths["classification_train"])
        save_validation_report(cls_report, "outputs/classification_validation.json")

        print("\nValidating segmentation dataset...")
        seg_report = validate_segmentation_dataset(
            paths["segmentation_images"], paths["segmentation_masks"]
        )
        save_validation_report(seg_report, "outputs/segmentation_validation.json")

    elif command == "serve":
        import uvicorn
        host = sys.argv[2] if len(sys.argv) > 2 else "0.0.0.0"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
        print(f"\nStarting Melanoma Orchestrator API on {host}:{port}")
        # Do not let the reloader watch model/cache files. A reload during a
        # Hugging Face download would restart the process and download BGE
        # again. Opt in explicitly for source-only development.
        reload_enabled = os.getenv("UVICORN_RELOAD", "0").lower() in {"1", "true", "yes"}
        uvicorn.run("api.server:app", host=host, port=port, reload=reload_enabled)

    # ── Agentic RAG v4 Commands ──────────────────────────────

    elif command == "build-rag":
        print("\n=== Building Agentic RAG v4 Hybrid Index ===")
        print("This will read your knowledge base, create Qdrant vectors,")
        print("BM25 index, and parent chunk store.\n")
        from rag_pipeline.create_vector_db import main as build_rag_index
        build_rag_index()

    elif command == "evaluate-rag":
        print("\n=== Running Agentic RAG v4 Evaluation Suite ===")
        from rag_pipeline.evaluate_rag import run_evaluation
        run_evaluation()

    elif command == "rag-test":
        if len(sys.argv) < 3:
            print("Usage: python main.py rag-test <image_path>")
            sys.exit(1)

        import json
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        image_path = sys.argv[2]
        print(f"\n{'='*60}")
        print(f"  Supervisor Agent — Image → Evidence-Grounded Report")
        print(f"  Image: {image_path}")
        print(f"{'='*60}\n")

        # Compatibility command: it now invokes the same supervisor route.
        from src.agents.orchestrator import OrchestratorAgent
        state = OrchestratorAgent("config.yaml").run(image_path, save_mask=False)
        report = state.final_report or {"status": state.status, "flags": state.flags}

        # Display
        print(f"\n{'='*60}")
        print("ABCDE CASE REPORT")
        print(f"{'='*60}")
        print(json.dumps(report, indent=2, default=str))

    else:
        print(f"Unknown command: {command}")
        print_usage()


if __name__ == "__main__":
    main()
