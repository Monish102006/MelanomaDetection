"""Evidence Agent: coordinates hybrid RAG retrieval and clinical context/evolution evidence."""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional
from .state import CaseState, EvidenceAssessment, EvidenceRecord

class EvidenceAgent:
    name = "evidence"
    def __init__(self, max_cycles: int = 3, backend: Any = None, clinical_context: Any = None):
        self.max_cycles = max_cycles
        self._backend = backend
        from .clinical_context import ClinicalContextHandler
        self.clinical_context = clinical_context or ClinicalContextHandler()

    def update_clinical_context(self, state: CaseState, supplied: Dict[str, Any] | None = None) -> CaseState:
        """Collect and normalize patient clinical context as evidence (e.g. reported evolution, duration, symptoms)."""
        return self.clinical_context.update(state, supplied)

    def assess_evolution(
        self,
        image_paths: List[str],
        timestamps: List[str] | None = None,
        config: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Assess temporal lesion change across visits as clinical evolution evidence."""
        return self.clinical_context.assess_evolution(image_paths, timestamps, config)

    def _backend_instance(self):
        if self._backend is None:
            from rag_pipeline.agent import create_backend
            self._backend = create_backend()
        return self._backend
    def retrieve_evidence(self, state: CaseState, evidence_goal: str) -> CaseState:
        try:
            backend = self._backend_instance()
            queries, accumulated, assessment = self._plan_queries(state, evidence_goal, backend), list(state.evidence), EvidenceAssessment()
            for cycle in range(1, self.max_cycles + 1):
                cycle_queries = queries if cycle == 1 else assessment.follow_up_queries
                if not cycle_queries: break
                documents = []
                for query in cycle_queries: documents.extend(backend.hybrid_search(query))
                accumulated = self._deduplicate(accumulated, documents, evidence_goal, cycle)
                assessment = self._evaluate(state, accumulated, evidence_goal, cycle)
                state.record(self.name, "retrieve_evidence", "sufficient" if assessment.sufficient else "follow_up_required", "evidence_gap", {"cycle": cycle, "goal": evidence_goal, "missing": assessment.missing}, tool=True)
                if assessment.sufficient: break
            state.evidence = state.retrieved_evidence = accumulated
            state.evidence_assessment = assessment
            state.uncertainty.retrieval = 1.0 if accumulated else 0.0
            state.uncertainty.evidence_sufficiency = 1.0 if assessment.sufficient else 0.35
            return state
        except Exception as e:
            state.record(self.name, "retrieve_evidence", "skipped", "backend_unavailable", {"error": str(e)}, tool=True)
            state.evidence = state.retrieved_evidence = []
            state.evidence_assessment = EvidenceAssessment(sufficient=False, missing=["A", "B", "C", "D", "E"], cycles_completed=self.max_cycles)
            state.uncertainty.retrieval = 0.0
            state.uncertainty.evidence_sufficiency = 0.0
            return state
    def _plan_queries(self, state: CaseState, goal: str, backend: Any = None) -> List[str]:
        queries = ["ABCDE clinical guideline evidence for suspicious pigmented lesions"]
        if goal and goal not in {"ABCDE", "differential", "measurement", "clinical_guidelines"}:
            queries.insert(0, goal)
        elif goal == "differential" or state.uncertainty.classification < 0.70:
            queries.append("differential diagnosis of uncertain pigmented skin lesion dermoscopy")
        if goal in {"ABCDE", "measurement", "clinical_guidelines"}:
            queries.extend(["melanoma asymmetry irregular border color variation evidence", "melanoma diameter criterion evidence calibrated measurement"])
        if state.abcde.get("E") and state.abcde["E"].availability == "unavailable":
            queries.append("clinical importance of evolution history in ABCDE melanoma assessment")
        # Preserve the advanced backend's HyDE implementation. Its input uses
        # only typed observations: a pixel diameter remains a pixel diameter,
        # and diameter_mm is absent unless CaseState validates calibration.
        if backend is not None and hasattr(backend, "create_queries"):
            features = getattr(state.diagnosis_result, "clinical_features", {}) or {}
            metrics = {
                "asymmetry_index": getattr(features.get("asymmetry"), "score_numeric", 0.0),
                "border_irregularity_score": getattr(features.get("border"), "score_numeric", 0.0),
                "color_variation_score": getattr(features.get("color"), "score_numeric", 0.0),
                "diameter_pixels": state.pixel_measurements.get("diameter").value if state.pixel_measurements.get("diameter") else 0.0,
                "diameter_mm": state.physical_measurements.get("diameter").value if state.physical_measurements.get("diameter") else None,
                "calibration_valid": bool(state.physical_measurements.get("diameter") and state.physical_measurements["diameter"].calibrated),
                "evolution": state.clinical_context.get("evolution", {"status": "unavailable", "reported_change": None}),
            }
            try:
                queries.extend(backend.create_queries(metrics))
            except Exception:
                # Standard, goal-specific queries remain a safe fallback when
                # the optional LLM-powered HyDE component is unavailable.
                pass
        return list(dict.fromkeys(queries))
    def _evaluate(self, state: CaseState, evidence: List[EvidenceRecord], goal: str, cycle: int) -> EvidenceAssessment:
        required = ["A", "B", "C", "D", "E"] if (goal == "ABCDE" or len(goal) > 20) else [goal]
        corpus = " ".join((item.title + " " + item.content).lower() for item in evidence)
        terms = {"A": ("asymmetr",), "B": ("border", "margin"), "C": ("color", "colour", "pigment"), "D": ("diameter", "size"), "E": ("evolution", "change", "temporal")}
        covered = [key for key in required if key not in terms or any(term in corpus for term in terms[key])]
        missing = [key for key in required if key not in covered]
        return EvidenceAssessment(not missing, covered, missing, [f"clinical evidence for ABCDE {key} criterion melanoma" for key in missing], cycle)
    @staticmethod
    def _deduplicate(existing: List[EvidenceRecord], docs: Iterable[Dict[str, Any]], goal: str, cycle: int) -> List[EvidenceRecord]:
        seen, result = {(e.source, e.title, e.content[:120]) for e in existing}, list(existing)
        for index, doc in enumerate(docs):
            metadata = doc.get("metadata", {}) or {}
            record = EvidenceRecord(str(metadata.get("id", f"{cycle}-{index}")), str(metadata.get("title", metadata.get("source", "Medical reference"))), str(metadata.get("source", "Knowledge base")), str(doc.get("text", "")), float(doc.get("rerank_score", doc.get("rrf_score", doc.get("dense_score", 0.0)))), goal, cycle)
            key = (record.source, record.title, record.content[:120])
            if record.content and key not in seen: result.append(record); seen.add(key)
        return result
