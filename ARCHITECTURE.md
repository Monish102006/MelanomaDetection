# CaseState Agent Architecture

`OrchestratorAgent` runs a bounded decision loop powered by **Hybrid Guided Reasoning**:
- **Safety Prerequisite Ordering**: Enforces foundational medical prerequisites (Image Quality → Vision & ABCD extraction → Temporal Evolution → Multimodal Fusion).
- **Cognitive Mind (LLM Deliberation)**: Uses Google Gemini to dynamically analyze clinical findings, identify diagnostic uncertainties, formulate targeted RAG literature queries, and evaluate evidence sufficiency and stop criteria.
- **Execution Trace**: Records both operational actions and clinical agent thoughts in the execution trace for full transparency.

```text
observe CaseState -> LLM Cognitive Deliberation -> invoke subagent -> update CaseState
                     ^                                                   |
                     +---------------------------------------------------+
```

## Top-Level Agents

The system has exactly **three** top-level agents orchestrated by the Orchestrator:

| Concern | Owner | Contract |
|---|---|---|
| Image quality, Swin, SegFormer, CV, Grad-CAM, calibration | **Vision Agent** | validated observations and raw measurements |
| Medical sources and coverage gaps | **Evidence Agent** | untrusted evidence records and bounded sufficiency assessment |
| Output | **Report Agent** | validated JSON plus optional PDF rendering |
| Routing | **Orchestrator Agent** | selected action and concise execution trace |

```text
Orchestrator Agent
    |
    ├── Vision Agent
    │     ├── Swin Transformer — Classification  ─┐
    │     ├── SegFormer — Segmentation           ─┘  parallel
    │     └── Measurement / ABCD extraction
    │
    ├── Evidence Agent
    │     ├── Hybrid RAG (BGE + Qdrant + BM25 + HyDE)
    │     └── Clinical Context & Evolution Evidence
    │
    └── Report Agent
```

## Evidence Agent & Clinical Context Handling

There is no separate top-level agent named "Clinical Context Agent"; clinical
context handling is an internal component of the **Evidence Agent**.
The Evidence Agent is the sole evidence entry point, combining:
1. **Medical Literature Retrieval**: `rag_pipeline.agent.AdvancedRetrievalBackend`
   retaining hybrid dense/BM25 retrieval, RRF, parent expansion, cross-encoder
   reranking, and HyDE query support.
2. **Clinical Context & Evolution**: Normalizes patient history, assesses
   temporal lesion evolution across image visits for the E criterion, and emits
   missing-information questions without fabricating history from image data.

## Measurement Safety

`CaseState.pixel_measurements` and `CaseState.physical_measurements` use
different `Measurement` objects. A physical value requires `calibrated=True`;
otherwise its value is `None` and the D feature says that physical diameter
is unavailable. Evolution is also `unavailable` until a caller supplies
clinical information.
