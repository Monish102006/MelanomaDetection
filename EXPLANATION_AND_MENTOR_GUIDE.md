# What I Did — Complete RAG Pipeline Explanation for Team & Mentor

> This document explains everything I built, **why I chose each tool**, what the alternatives were, and how it all fits together. Written in simple language so you can explain it confidently.

---

## 1. First — What is RAG and Why Does Our Project Need It?

### The Problem
Our project detects melanoma from skin images using the **ABCDE rule** (Asymmetry, Border, Color, Diameter, Evolution). The image pipeline extracts numbers — like "asymmetry is 30%" or "border irregularity is 0.6" — but **numbers alone don't mean anything to a doctor or patient**.

If someone gets a result saying "asymmetry_index: 38.5", the natural question is: **"Is that dangerous? What does the medical literature say about this level of asymmetry?"**

### The Solution — RAG (Retrieval-Augmented Generation)
RAG is a technique where we:
1. **Store medical literature** (research papers, clinical guidelines) in a searchable database
2. **Retrieve relevant passages** when a case comes in (e.g., "what do medical guidelines say about high asymmetry?")
3. **Generate an evidence-based report** using an LLM (AI language model) that cites the actual medical sources

**Without RAG**: The AI just makes up an explanation (called "hallucination" — very dangerous in medicine).
**With RAG**: The AI is forced to ground its explanation in real published medical evidence.

### Why Not Just Use ChatGPT/Gemini Directly?
Good question. If you just ask ChatGPT "is 38% asymmetry dangerous?", it might give you an answer, but:
- You don't know **where** that information came from
- It might be **outdated or wrong** (hallucination)
- It can't cite **specific page numbers** from specific clinical guidelines
- It doesn't have access to **your specific medical literature**

RAG solves all of this. The AI can ONLY use the papers we give it.

---

## 2. The Medical Literature We Used

We have **4 PDF sources** stored in the `medical_literature/` folder:

| PDF | What It Contains | Why We Chose It |
|---|---|---|
| **ABCD-ABCDE evidence.pdf** | Systematic review of all ABCDE prediction rule studies | This is THE definitive evidence paper on whether ABCDE actually works for detecting melanoma |
| **Dermoscopy + ABCD.pdf** | Review of dermoscopy combined with ABCD scoring | Gives us the clinical scoring thresholds used in dermatoscopy |
| **Guidelines-of-care-for-the-management-of-primary-c.pdf** | AAD (American Academy of Dermatology) clinical guidelines | The gold standard — official treatment & evaluation guidelines from the largest dermatology body |
| **melanoma.pdf** | Research paper on explainable melanoma AI with contrastive learning | Our base research paper — the academic foundation of our project |

> **Why these 4 specifically?** They cover the three levels of medical evidence: (1) clinical guidelines (what doctors follow), (2) systematic reviews (what the research says), and (3) our own base paper. A mentor would expect us to have literature from multiple evidence levels, not just one paper.

---

## 3. How We Store and Search the Literature — The Vector Database

### What is a Vector Database?
Imagine you have 80 pages of medical PDFs. When a case comes in, you need to find the **most relevant paragraphs** — not by keyword matching (that's too simple) but by **meaning**.

A vector database converts every text chunk into a **list of numbers** (called an "embedding" or "vector") that captures its meaning. Then when a query comes in, it converts the query into numbers too, and finds the chunks whose numbers are **closest** — meaning they're semantically similar.

### What We Chose: **Qdrant** + **BAAI/bge-base-en-v1.5**

#### Vector Database: Qdrant

| Option | Pros | Cons | Why We Chose/Rejected |
|---|---|---|---|
| **Qdrant** ✅ (our choice) | Free, runs locally AND in the cloud, Python-native, built-in LangChain support, very fast | Relatively newer | **Best fit** — works locally for development, can switch to cloud for deployment with just an env variable change. No external server needed for local mode. |
| **Pinecone** | Popular, fully managed cloud | Paid service, no local mode, vendor lock-in | Rejected — costs money, can't run offline/locally during development |
| **ChromaDB** | Very simple, good for prototypes | Limited scalability, no built-in cloud option, fewer features | Rejected — too simple for a production-grade project, no cloud deployment story |
| **Weaviate** | Feature-rich, open source | Heavier setup, requires Docker for local, more complex | Rejected — over-engineered for our use case |
| **FAISS** (Facebook) | Extremely fast, industry standard | No metadata filtering, no persistence by default, just a library not a DB | Rejected — missing too many features we need (metadata, persistence, cloud) |

#### Embedding Model: BAAI/bge-base-en-v1.5

| Option | Dimensions | Size | Why We Chose/Rejected |
|---|---|---|---|
| **bge-small-en-v1.5** (v3.0) | 384 | 33M params | We **started** with this — small, fast, but misses clinical nuance |
| **bge-base-en-v1.5** ✅ (v4.0) | 768 | 109M params | **Upgraded to this** — 3x more capacity to understand medical terminology. Best balance of quality vs speed. |
| **bge-large-en-v1.5** | 1024 | 335M params | Rejected — too slow for our hardware, marginal improvement over base |
| **OpenAI text-embedding-ada-002** | 1536 | Cloud-only | Rejected — requires paid API, can't run offline, adds latency |
| **PubMedBERT / BiomedBERT** | Varies | Medical-specific | Considered — but not available as FastEmbed models, would need different infrastructure |

> **Why bge-base specifically?** It's from the MTEB benchmark leaderboard (a competition for embedding models). bge-base ranks in the top 10 while being small enough to run on a laptop CPU. The "base" size (768 dimensions) captures enough semantic detail to distinguish between "asymmetrical border" and "symmetrical border" — something the "small" (384-dim) model struggled with.

---

## 4. How We Parse the PDFs — Layout-Aware Extraction

### The Problem with Basic PDF Parsing
Medical PDFs have **tables**, **multi-column layouts**, and **figure captions**. A basic parser reads them left-to-right, top-to-bottom, which **scrambles the content**:

```
Basic parser reads:    "Table 1: ABCDE Score  Sensitivity  Specificity"                      ................................................................................................................................................................................................                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

                       → becomes garbled: "Table 1: ABCDE 92%  78%  Score Sensitivity"

Layout-aware parser:   Correctly reads the table as structured data
```

### What We Chose: **PyMuPDF** (fitz)

| Option | Approach | Why We Chose/Rejected |
|---|---|---|
| **pypdf** (v3.0) | Basic text extraction, no layout awareness | We **started** with this — simple but loses all table structure |
| **PyMuPDF** ✅ (v4.0) | Layout-aware, preserves tables, columns, figures | **Upgraded to this** — the AAD clinical guidelines have critical tables with sensitivity/specificity data that was being lost |
| **Unstructured.io** | Most advanced, AI-powered layout detection | Rejected — requires extra dependencies, slower, overkill for our 4 PDFs |
| **Docling** (IBM) | Advanced document understanding | Rejected — relatively new, complex setup |
| **pdfplumber** | Good table extraction | Considered — but PyMuPDF is faster and has better LangChain integration |

> **Why does this matter?** The AAD guidelines PDF has a table showing "ABCDE rule: sensitivity 92%, specificity 78%". With the old parser, this table was garbled text. With PyMuPDF, the RAG can now correctly cite this statistic in its reports.

---

## 5. How We Chunk the Documents — Parent-Child Strategy

### The Problem
You can't embed an entire 22-page PDF as one vector — the embedding would be too vague. You need to split it into smaller pieces ("chunks"). But there's a trade-off:

- **Small chunks** (500 chars) → Very precise embeddings, great for search → But too little context for the LLM to reason over
- **Large chunks** (2000 chars) → Rich context for the LLM → But vague embeddings, poor search accuracy

### Our Solution: Parent-Child Chunking (Best of Both Worlds)

We use a **two-tier approach**:
1. **Child chunks (500 chars)**: Used for search — small and focused, so the embedding precisely captures "this paragraph is about asymmetry scoring"
2. **Parent chunks (2000 chars)**: Used for LLM context — after we find the right child, we swap it for its parent (the larger surrounding context)

Think of it like a library: you search the **index cards** (child chunks) to find the right book section, but then you read the **full page** (parent chunk) for complete understanding.

| Strategy | Used For | Pros | Cons |
|---|---|---|---|
| **Flat chunking** (v3.0) | Everything | Simple | Poor either at search OR at context — can't be good at both |
| **Parent-Child** ✅ (v4.0) | Search + Context | Best of both worlds — precise search, rich context | Slightly more complex to implement |
| **Sliding window** | Some systems | Easy overlap | Still one-tier, same trade-off problem |
| **Semantic chunking** | Research systems | Chunks by meaning boundaries | Very slow, requires NLP pipeline |

> **Result**: We went from 526 flat chunks to **989 child chunks + 224 parent chunks**. The retrieval is more precise AND the LLM gets better context.

---

## 6. How We Search — Hybrid Retrieval (Dense + Sparse + Fusion)

### Why Not Just Vector Search?
Vector search (dense retrieval) is great at finding **semantically similar** content. But it can miss **exact medical terms**. For example:

- Query: "dermoscopy ABCD scoring" 
- Vector search might return a passage about "skin examination techniques" (similar meaning but missing the specific term)
- Keyword search (BM25) would find the passage that literally contains "dermoscopy" and "ABCD"

### Our Approach: Combine Both

We use **3 search methods combined**:

#### Step 1: Dense Vector Search (Qdrant)
- Converts query → 768-dimensional vector → finds semantically similar chunks
- Good at: Understanding meaning ("melanocytic neoplasm" matches "melanoma")
- Bad at: Exact term matching

#### Step 2: BM25 Keyword Search (Sparse)
- Classic keyword matching with **medical stemming**
- "Asymmetrical" → stemmed to "asymmetr" → matches "asymmetry", "asymmetric", "asymmetrical"
- Also does **medical synonym expansion**: query for "melanoma" also searches "melanocytic", "malignant", "neoplasm"
- Good at: Exact term matching, rare medical terms
- Bad at: Understanding meaning/paraphrasing

#### Step 3: Reciprocal Rank Fusion (RRF)
- Merges the results from both searches into one ranked list
- A document that appears in **both** search results gets a higher combined score
- This way we get the **best of both worlds**

#### Step 4: Cross-Encoder Re-Ranking
- After fusion, we take the top candidates and re-score them using a **cross-encoder** neural network
- A cross-encoder is much more accurate than the initial search because it reads the **query AND passage together** (not separately)
- Also applies a **score threshold** — passages below the minimum relevance are thrown away, even if they're in the "top 5"

| Component | What It Does | Why It's Needed |
|---|---|---|
| Dense search (Qdrant) | Semantic similarity | Finds passages that mean the same thing |
| BM25 keyword search | Exact term matching | Finds passages with specific medical terms |
| RRF Fusion | Combines both rankings | Passages found by both methods rank highest |
| Cross-encoder re-ranking | Fine-grained relevance scoring | The final quality filter — only truly relevant passages survive |
| Parent expansion | Swaps child → parent chunks | Gives the LLM full context instead of snippets |

> **Why BM25 specifically and not Elasticsearch?** BM25 is the same algorithm Elasticsearch uses internally, but we run it as a lightweight Python library (`rank_bm25`). No need to install and manage a separate Elasticsearch server for 4 PDFs.

---

## 7. How We Generate Queries — HyDE (Hypothetical Document Embeddings)

### The Problem with Simple Queries
In v3.0, when a case had "asymmetry_index: 38.5", we'd search with: `"melanoma skin lesion asymmetry 38.5"`. This is a **short keyword query** — its embedding is far from actual medical passages in vector space.

### The HyDE Solution
Instead of searching with short keywords, we ask the LLM: *"Write a hypothetical medical textbook paragraph about a lesion with 38.5% asymmetry."*

The LLM generates something like:
> *"A pigmented lesion demonstrating significant bilateral asymmetry with an index of 38.5% is highly suspicious. Such marked asymmetry, exceeding the typical threshold of 20%, suggests irregular melanocytic proliferation and warrants urgent clinical evaluation under the ABCDE framework..."*

We then embed **this hypothetical paragraph** and search with it. Because it sounds like actual medical text, its vector embedding is much **closer** to real medical passages in the database — dramatically improving retrieval recall.

| Query Method | How It Works | Quality |
|---|---|---|
| **Keyword queries** (v3.0) | Short search terms like "melanoma asymmetry" | Basic — embedding is far from medical text |
| **HyDE** ✅ (v4.0) | LLM generates fake textbook paragraphs, search with those | Advanced — embedding is very close to real medical passages |
| **Query expansion** | Add synonyms to the query | Medium — helps but still short queries |
| **Multi-query** | Generate multiple query variants | Medium — more queries but still keyword-based |

> **Why HyDE?** It's from a 2023 research paper by Carnegie Mellon University. It's one of the most impactful retrieval techniques in recent years because it bridges the gap between how users ask questions and how documents are written. Perfect for a research-level project.

---

## 8. The Agentic Part — Multi-Hop Reasoning Loop

### What Makes It "Agentic"?
Normal RAG: search → retrieve → generate. Done in one pass.

**Agentic RAG** (what we built): The agent **evaluates its own work** and decides if it needs more evidence. It's like a research assistant that:
1. Searches for evidence about all 5 ABCDE criteria
2. **Checks**: "Do I have enough evidence about Asymmetry? Border? Color? Diameter? Evolution?"
3. If any criterion is missing evidence → generates **follow-up queries** and searches again
4. Repeats up to **3 cycles** until evidence is sufficient

This is called **multi-hop reasoning** because each cycle "hops" to find new, more specific evidence.

### Why 3 Cycles Maximum?
- **1 cycle**: Usually sufficient (our tests show this) — the HyDE queries are so good that the first retrieval covers all 5 criteria
- **2-3 cycles**: Safety net for edge cases where the first pass misses something (e.g., rare medical terms)
- **More than 3**: Diminishing returns — if 3 passes can't find evidence, more passes won't help

---

## 9. The LLM We Use — Gemini 3.1 Flash Lite

| Option | Speed | Cost | Quality | Why We Chose/Rejected |
|---|---|---|---|---|
| **Gemini 3.1 Flash Lite** ✅ | Very fast | Free tier generous | Good for structured tasks | **Best fit** — fast enough for real-time use, free for our scale, good at following structured JSON output instructions |
| **GPT-4** | Slow | Expensive (\$30/1M tokens) | Best quality | Rejected — too expensive for a student project, high latency |
| **GPT-3.5 Turbo** | Fast | Moderate | Moderate | Rejected — worse at structured JSON output, costs money |
| **Claude 3** | Medium | Expensive | Very good | Rejected — expensive, no free tier for API |
| **Llama 3 (local)** | Depends on GPU | Free | Good | Rejected — requires a powerful GPU we don't have |
| **Gemini Pro** | Medium | Free tier | Better quality | Considered — could upgrade to this if Flash Lite quality isn't enough |

> **Why specifically Flash Lite?** For our use case, the LLM's main job is to (1) generate HyDE paragraphs and (2) write structured JSON reports following a template. Both are **structured tasks** where Flash Lite performs well. We don't need the creativity or reasoning depth of GPT-4 — we need speed and structured output compliance. And it's free.

---

## 10. The API — FastAPI with Production Features

### Why FastAPI?

| Option | Language | Speed | Docs | Why We Chose/Rejected |
|---|---|---|---|---|
| **FastAPI** ✅ | Python | Very fast (async) | Auto-generated Swagger | **Best fit** — async support (critical for LLM calls), auto-generates interactive API docs at `/docs`, Pydantic validation built-in |
| **Flask** | Python | Slower (sync) | Manual | Rejected — no async support, would block during 40-second LLM calls |
| **Django REST** | Python | Medium | Good | Rejected — too heavy for a REST API, designed for full web apps |
| **Express.js** | JavaScript | Fast | Manual | Rejected — our entire pipeline is Python, switching language adds complexity |

### Production Features We Added

| Feature | What It Does | Why It Matters |
|---|---|---|
| **API Key Authentication** | Requires `X-API-Key` header | Prevents unauthorized access to the system |
| **Rate Limiting** (slowapi) | Max 10 requests/minute per IP | Prevents abuse, protects our Gemini API quota |
| **Async Execution** | LLM calls run in thread pool | Doesn't block the API server while waiting for 40-second LLM responses |
| **Per-Request Agent** | Fresh agent instance per request | Prevents race conditions when multiple people use the API simultaneously |
| **SSE Streaming** | Real-time reasoning trace | User sees progress instead of waiting 40 seconds for a blank screen |
| **CORS Security** | Explicit allowed origins | Only our frontend can access the API, not random websites |
| **File Upload Security** | Size limits, magic-byte validation, temp cleanup | Prevents malicious file uploads, XSS, path traversal attacks |

---

## 11. Evaluation — How We Know It Works

### Heuristic Metrics (Our Custom Checks)

| Metric | What It Checks | Our Score |
|---|---|---|
| **Context Relevance** | Are retrieved passages about ABCDE/melanoma? | 1.000 |
| **Faithfulness** | Does it avoid diagnosing? Cite sources? Mention limitations? | 1.000 |
| **Answer Completeness** | Does it cover all 5 ABCDE criteria? | 1.000 |
| **Reasoning Quality** | Multiple queries? Hybrid search? Reasoning cycles? | 1.000 |

### RAGAS (Industry-Standard RAG Evaluation)

We also integrated **RAGAS** — the standard library for evaluating RAG systems. It uses an LLM to grade:
- **Faithfulness**: Is the answer actually supported by the retrieved evidence?
- **Answer Relevancy**: Does the answer address the question?
- **Context Precision**: Are the retrieved contexts actually relevant?

> **Why both?** Our heuristic checks are fast structural checks (does it have sources? does it cover all criteria?). RAGAS provides deeper LLM-graded quality assessment. Having both shows rigor.

---

## 12. Deployment & Cloud Readiness

### Qdrant Cloud (Available Everywhere)
Our system can switch between:
- **Local mode** (development): Vectors stored on disk, works offline
- **Cloud mode** (deployment): Vectors stored on Qdrant Cloud (free tier: 1GB), accessible from anywhere

Switching is just changing one line in `.env`:
```
QDRANT_MODE=cloud
```

### Docker (One-Command Deployment)
We created a `Dockerfile` and `docker-compose.yml` so anyone can run the entire system with:
```bash
docker-compose up
```
No need to install Python, create virtual environments, or download models manually.

---

## 13. Security — Why It Matters in Medical AI

| Threat | Our Protection |
|---|---|
| **Prompt Injection** (user tricks AI into ignoring medical safety rules) | Input sanitization removes phrases like "ignore previous instructions" |
| **Malicious File Upload** (uploading a virus disguised as an image) | Magic-byte validation, file size limits, UUID filename randomization, guaranteed cleanup |
| **API Abuse** (overloading the system) | Rate limiting (10 req/min), API key authentication |
| **Data Leakage** (internal errors expose system details) | Error masking — users see "an error occurred", not stack traces |
| **CORS Exploitation** (unauthorized websites accessing our API) | Explicit origin whitelist instead of wildcard `*` |

> **Why is this important?** Medical AI systems handle sensitive health data. A mentor evaluating our project would expect basic security measures. Without them, the system is technically a liability.

---

## 14. Complete Pipeline Flow (How Everything Connects)

```
Patient Image
     │
     ▼
┌─────────────────────────┐
│  Image Pipeline          │
│  (Segmentation + ABCDE   │
│   Feature Extraction)    │
│                          │
│  Output: asymmetry=38.5  │
│  border=0.76, color=45.2 │
│  diameter=320px           │
│  evolution=enlarging      │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  RAG Pipeline (MY WORK)  │
│                          │
│  1. HyDE Query Generation│
│     LLM writes textbook  │
│     paragraphs as queries│
│                          │
│  2. Hybrid Search         │
│     Dense (Qdrant) +      │
│     BM25 (keywords) +    │
│     RRF Fusion            │
│                          │
│  3. Cross-Encoder Rerank  │
│     Neural re-scoring +   │
│     threshold filtering   │
│                          │
│  4. Parent Expansion      │
│     Small chunk → large   │
│     context window        │
│                          │
│  5. Multi-Hop Reasoning   │
│     Evaluate sufficiency  │
│     → follow-up if needed │
│                          │
│  6. Report Generation     │
│     Evidence-grounded     │
│     ABCDE interpretation  │
│     with citations        │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Structured JSON Report  │
│                          │
│  - 5 ABCDE interpretations│
│  - Source citations       │
│  - Limitations           │
│  - Reasoning audit trail  │
│  - Performance metrics    │
└─────────────────────────┘
```

---

## 15. Quick Summary — What to Tell Your Mentor

> "My part of the project is the **Agentic RAG pipeline** — the brain that takes the image analysis numbers and explains them using real medical literature."
>
> "I built a system that stores 4 medical PDFs (clinical guidelines, evidence reviews, our base paper) in a **Qdrant vector database** using **768-dimensional embeddings** from `bge-base-en-v1.5`."
>
> "When a case comes in, it uses **HyDE** to generate intelligent search queries, then does **hybrid search** (semantic vectors + keyword matching + fusion), re-ranks with a **cross-encoder neural network**, and expands results using a **parent-child chunking strategy**."
>
> "The agent then runs a **multi-hop reasoning loop** — it checks if it has enough evidence for all 5 ABCDE criteria, and searches again if needed."
>
> "Finally, it generates a **structured evidence-based report** citing specific papers and page numbers, with proper medical disclaimers."
>
> "The whole system has **production-grade features** — API authentication, rate limiting, async execution, SSE streaming, Docker containerization, and Qdrant Cloud support for deployment."
>
> "All 5 test cases pass with a **perfect 1.000 score** across context relevance, faithfulness, completeness, and reasoning quality."
