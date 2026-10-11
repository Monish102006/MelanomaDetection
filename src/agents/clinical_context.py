"""Clinical Context Handler: internal component of Evidence Agent for patient history and evolution."""
from __future__ import annotations
from typing import Any, Dict, List
from .state import ABCDEFeature, CaseState

class ClinicalContextHandler:
    name = "evidence"
    QUESTION_MAP = {"evolution": "Has the lesion changed in size, shape, color, elevation, or symptoms? If so, over what time period?"}

    def update(self, state: CaseState, supplied: Dict[str, Any] | None = None) -> CaseState:
        supplied = supplied or {}
        for key in ("evolution", "duration", "reported_size_change", "shape_change", "color_change", "symptoms", "history"):
            if key in supplied and supplied[key] is not None:
                state.clinical_context[key] = supplied[key]
        evolution = state.clinical_context.get("evolution")
        if evolution is None:
            state.abcde["E"] = ABCDEFeature("E", "unavailable", {}, limitation="Evolution cannot be determined from a single image; no history was supplied.")
            state.pending_clinical_questions = [{"field": "evolution", "question": self.QUESTION_MAP["evolution"]}]
            state.uncertainty.clinical_context = 0.0
            state.flags.append("Evolution (E) unavailable: patient history was not supplied.")
        else:
            state.abcde["E"] = ABCDEFeature("E", "available", {"reported_history": evolution}, "Patient-reported evolution recorded; it is not inferred from the image.")
            state.pending_clinical_questions = []
            state.uncertainty.clinical_context = 0.8
        state.record(self.name, "request_clinical_context", "updated", "missing_information", {"pending_fields": [q["field"] for q in state.pending_clinical_questions]})
        return state

    def missing_information(self, state: CaseState) -> List[Dict[str, str]]:
        return list(state.pending_clinical_questions)

    def assess_evolution(
        self,
        image_paths: List[str],
        timestamps: List[str] | None = None,
        config: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Evaluate the Evolution (E) factor by comparing historical images."""
        import os
        import logging
        logger = logging.getLogger(__name__)
        
        timestamps = timestamps or [None] * len(image_paths)
        
        if len(image_paths) < 2:
            return {
                "status": "unable_to_assess",
                "confidence": 0.0,
                "observations": ["Only one image was provided; temporal comparison is not possible."],
                "changes": {
                    "size": "Not assessed", "shape": "Not assessed", "color": "Not assessed",
                    "structure": "Not assessed", "other": "Not assessed"
                },
                "comparison": {
                    "images_compared": len(image_paths),
                    "time_interval": "unknown",
                    "same_lesion_confidence": 0.0
                },
                "limitations": ["A minimum of two images from different time points is required."]
            }
            
        time_interval = self._compute_time_interval(timestamps)
        
        cfg = config or {}
        mode = cfg.get("mode", "template")
        llm_mode = (mode == "llm")
        
        if llm_mode:
            try:
                return self._assess_with_llm(image_paths, time_interval, cfg)
            except Exception as exc:
                logger.warning("LLM evolution assessment failed (%s); falling back to heuristic.", type(exc).__name__)
                
        return self._assess_heuristic(image_paths, time_interval)
        
    def _assess_with_llm(self, image_paths: List[str], time_interval: str, config: Dict[str, Any]) -> Dict[str, Any]:
        import os
        import json
        import re
        from google import genai
        from PIL import Image
        
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
            
        client = genai.Client(api_key=api_key)
        model_name = config.get("model") or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

        
        label_map = {0: "Earliest image", len(image_paths) - 1: "Most recent image"}
        parts = [
            ("You are a dermatology AI assistant specialising in temporal lesion analysis.\n"
             "You will be shown skin lesion images from DIFFERENT time points of the SAME lesion "
             f"(estimated time interval: {time_interval}).\n\n"
             "Compare every pair of images and evaluate the Evolution (E) criterion of ABCDE.\n\n"
             "Assess the following changes:\n"
             "  - Size/diameter change\n"
             "  - Shape or border changes\n"
             "  - Color changes or new color regions\n"
             "  - Structural/pattern changes\n"
             "  - Asymmetry changes\n"
             "  - New surrounding features\n\n"
             "IMPORTANT CONSTRAINTS:\n"
             "  * Do NOT diagnose melanoma solely from evolution.\n"
             "  * Do NOT infer change from lighting, camera angle, zoom, focus, or image quality differences.\n"
             "  * If images look like different lesions or views, report same_lesion_confidence < 0.4.\n"
             "  * If temporal differences cannot be reliably assessed, use status: unable_to_assess.\n\n"
             "Respond ONLY with valid JSON in EXACTLY this structure:\n"
             "{\n"
             '  "evolution": {\n'
             '    "status": "no_significant_evolution | possible_evolution | clear_evolution | unable_to_assess",\n'
             '    "confidence": <0.0-1.0>,\n'
             '    "observations": ["..."],\n'
             '    "changes": {\n'
             '      "size": "...",\n'
             '      "shape": "...",\n'
             '      "color": "...",\n'
             '      "structure": "...",\n'
             '      "other": "..."\n'
             "    },\n"
             '    "comparison": {\n'
             f'      "images_compared": {len(image_paths)},\n'
             '      "time_interval": "...",\n'
             '      "same_lesion_confidence": <0.0-1.0>\n'
             "    },\n"
             '    "limitations": ["..."]\n'
             "  }\n"
             "}")
        ]
        
        for idx, path in enumerate(image_paths):
            label = label_map.get(idx, f"Intermediate image {idx}")
            parts.append(f"\n[{label}]")
            with Image.open(path) as img:
                parts.append(img.copy())
                
        response = client.models.generate_content(model=model_name, contents=parts)
        text = re.sub(r"```(?:json)?", "", response.text).strip()
        
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            data = json.loads(match.group()) if match else {}
            
        ev = data.get("evolution", {})
        status = ev.get("status", "unable_to_assess")
        if status not in ("no_significant_evolution", "possible_evolution", "clear_evolution", "unable_to_assess"):
            status = "unable_to_assess"
            
        changes = ev.get("changes", {})
        comparison = ev.get("comparison", {})
        
        return {
            "status": status,
            "confidence": float(ev.get("confidence", 0.0)),
            "observations": list(ev.get("observations", [])),
            "changes": {
                "size": changes.get("size", "Not assessed"),
                "shape": changes.get("shape", "Not assessed"),
                "color": changes.get("color", "Not assessed"),
                "structure": changes.get("structure", "Not assessed"),
                "other": changes.get("other", "Not assessed"),
            },
            "comparison": {
                "images_compared": int(comparison.get("images_compared", len(image_paths))),
                "time_interval": str(comparison.get("time_interval", time_interval)),
                "same_lesion_confidence": float(comparison.get("same_lesion_confidence", 0.0)),
            },
            "limitations": list(ev.get("limitations", [])),
        }

    def _assess_heuristic(self, image_paths: List[str], time_interval: str) -> Dict[str, Any]:
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            return {
                "status": "unable_to_assess", "confidence": 0.0,
                "observations": ["Required image libraries are not available for heuristic comparison."],
                "changes": {"size": "Not assessed", "shape": "Not assessed", "color": "Not assessed", "structure": "Not assessed", "other": "Not assessed"},
                "comparison": {"images_compared": len(image_paths), "time_interval": time_interval, "same_lesion_confidence": 0.0},
                "limitations": ["PIL/numpy not importable; heuristic assessment unavailable."]
            }
            
        target_size = (256, 256)
        arrays = []
        load_errors = []
        
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    gray = img.convert("L").resize(target_size, Image.LANCZOS)
                    arrays.append(np.asarray(gray, dtype=float))
            except Exception as exc:
                load_errors.append(f"{path}: {type(exc).__name__}")
                
        if len(arrays) < 2:
            return {
                "status": "unable_to_assess", "confidence": 0.0,
                "observations": ["One or more images could not be loaded for comparison."],
                "changes": {"size": "Not assessed", "shape": "Not assessed", "color": "Not assessed", "structure": "Not assessed", "other": "Not assessed"},
                "comparison": {"images_compared": len(image_paths), "time_interval": time_interval, "same_lesion_confidence": 0.0},
                "limitations": ["Image load failures prevented heuristic comparison."] + load_errors
            }
            
        diffs = []
        for i in range(len(arrays) - 1):
            mae = float(np.mean(np.abs(arrays[i] - arrays[i + 1])))
            diffs.append(mae)
            
        max_diff = max(diffs)
        mean_diff = sum(diffs) / len(diffs)
        
        limitations = [
            "Heuristic comparison cannot distinguish genuine lesion change from lighting, zoom, angle, or focus differences.",
            "LLM-based comparison (config: mode=llm) is recommended for reliable evolution assessment."
        ]
        if load_errors:
            limitations.append(f"Image load warnings: {'; '.join(load_errors)}")
            
        if max_diff > 40:
            return {
                "status": "unable_to_assess", "confidence": 0.3,
                "observations": [f"Pixel-level difference between images is very large (mean absolute error: {mean_diff:.1f}/255)."],
                "changes": {"size": "Not assessed", "shape": "Not assessed", "color": "Not assessed", "structure": "Not assessed", "other": "Not assessed"},
                "comparison": {"images_compared": len(image_paths), "time_interval": time_interval, "same_lesion_confidence": max(0.0, 1.0 - max_diff / 255.0)},
                "limitations": limitations
            }
            
        if mean_diff < 8:
            return {
                "status": "no_significant_evolution", "confidence": 0.45,
                "observations": [f"Low pixel-level difference between images (mean absolute error: {mean_diff:.1f}/255)."],
                "changes": {"size": "No substantial size change detected at pixel level.", "shape": "No substantial shape change detected at pixel level.", "color": "No substantial color change detected at pixel level.", "structure": "Heuristic method cannot reliably assess structural changes.", "other": "Not assessed"},
                "comparison": {"images_compared": len(image_paths), "time_interval": time_interval, "same_lesion_confidence": min(1.0, max(0.0, 1.0 - mean_diff / 80.0))},
                "limitations": limitations
            }
            
        return {
            "status": "possible_evolution", "confidence": 0.35,
            "observations": [f"Moderate pixel-level difference between images (mean absolute error: {mean_diff:.1f}/255)."],
            "changes": {"size": "Possible change", "shape": "Possible change", "color": "Possible change", "structure": "Heuristic method cannot reliably assess structural changes.", "other": "Not assessed"},
            "comparison": {"images_compared": len(image_paths), "time_interval": time_interval, "same_lesion_confidence": min(1.0, max(0.0, 1.0 - mean_diff / 120.0))},
            "limitations": limitations
        }

    @staticmethod
    def _compute_time_interval(timestamps: List[str | None]) -> str:
        from datetime import datetime, timezone
        parsed = []
        for ts in timestamps:
            if not ts: continue
            for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(ts, fmt)
                    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
                    parsed.append(dt)
                    break
                except ValueError: continue
        if len(parsed) < 2: return "unknown"
        parsed.sort()
        days = (parsed[-1] - parsed[0]).days
        if days == 0: return "same day"
        if days < 7: return f"{days} day(s)"
        if days < 30: return f"~{days // 7} week(s)"
        if days < 365: return f"~{days // 30} month(s)"
        return f"~{days // 365} year(s)"

# Backward-compatibility alias: ClinicalContextAgent is an internal component of EvidenceAgent
ClinicalContextAgent = ClinicalContextHandler
__all__ = ["ClinicalContextHandler", "ClinicalContextAgent"]
