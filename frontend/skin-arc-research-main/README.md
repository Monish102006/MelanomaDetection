# Lumina AI

Build a production-quality frontend for a research project called:

"MelanomaAI — Explainable Agentic RAG for Melanoma Analysis"

IMPORTANT:

This is an academic/research decision-support system, NOT a medical diagnosis application.

The backend already exists separately as a Python FastAPI application.

The frontend must be designed so it can connect to the existing backend later.

TECH STACK:

- React

- TypeScript

- Vite

- Tailwind CSS

- Framer Motion

- Three.js / React Three Fiber for 3D

- Lucide icons

DESIGN DIRECTION:

Create a premium futuristic medical-AI research interface.

Visual style:

- cinematic

- dark background

- deep navy/black

- glassmorphism used carefully

- white typography

- cyan/teal/electric-blue accents

- subtle red accents only where appropriate

- professional scientific appearance

- minimal and clean

- high-end AI research laboratory aesthetic

DO NOT make it look like:

- a generic hospital website

- a basic admin dashboard

- a normal college project

- an overly colorful medical template

The website should feel like an advanced AI research product.

==================================================

PAGE STRUCTURE

==================================================

1. HERO SECTION

Create a full-screen cinematic hero.

Title:

"Explainable AI for Melanoma Analysis"

Subtitle:

"An evidence-grounded Agentic RAG framework connecting

dermoscopic analysis, clinical features and medical literature."

Add two buttons:

"Analyze Lesion"

"Explore Architecture"

Create a subtle animated 3D scientific object in the center/right using

React Three Fiber.

The object should represent a stylized skin lesion / scientific cellular

structure.

Use slow rotation, particles and subtle lighting.

Do not create a disturbing or overly realistic medical image.

Add subtle scroll animation.

==================================================

2. HOW THE SYSTEM WORKS

==================================================

Create a horizontal/vertical animated pipeline:

IMAGE

↓

LESION SEGMENTATION

↓

ABCD FEATURE EXTRACTION

↓

MODEL PREDICTION

↓

AGENTIC QUERY GENERATION

↓

QDRANT RETRIEVAL

↓

EVIDENCE EVALUATION

↓

GEMINI SYNTHESIS

↓

EVIDENCE-GROUNDED REPORT

Animate the data flow between nodes.

Each node should be interactive.

When hovering a node, display a short explanation.

==================================================

3. ANALYSIS WORKSPACE

==================================================

Create a premium analysis interface.

Left side:

Image upload area.

Allow:

- drag and drop

- file selection

- image preview

Button:

"START ANALYSIS"

During analysis display an animated processing sequence:

Uploading image

Segmenting lesion

Extracting ABCD features

Running classifier

Generating retrieval queries

Searching medical literature

Evaluating evidence

Generating report

Use animated progress states.

==================================================

4. LESION ANALYSIS

==================================================

After analysis show:

Uploaded image

Segmented lesion

ABCD measurements

Model prediction

Model confidence

Create an interactive visualization for:

Asymmetry

Border

Color

Diameter

Use animated charts/gauges rather than plain text.

IMPORTANT:

Do not invent medical thresholds.

Clearly label these as:

"Image-derived measurements"

and:

"Research model output"

==================================================

5. AGENTIC RAG TRACE

==================================================

Create a visually impressive live AI reasoning/retrieval panel.

Show:

AGENT QUERY GENERATION

Example:

"ABCD / ABCDE clinical criteria for melanoma assessment"

Then:

RETRIEVAL

Qdrant

Show retrieved sources:

ABCD-ABCDE evidence.pdf

Dermoscopy + ABCD.pdf

melanoma.pdf

AAD clinical guideline

Then:

EVIDENCE EVALUATION

Show:

Evidence sufficient

or

Evidence insufficient

Then:

GEMINI SYNTHESIS

Show:

"Generating evidence-grounded report..."

Animate each stage.

This section is extremely important because Agentic RAG

is the main contribution of the project.

==================================================

6. EVIDENCE-GROUNDED REPORT

==================================================

Create a professional research report interface.

Sections:

MODEL PREDICTION

OBSERVED MEASUREMENTS

EVIDENCE INTERPRETATION

SOURCES

LIMITATIONS

CLINICAL CONTEXT

Do NOT show "High Risk" or definitive medical diagnosis unless

explicitly returned by the backend.

Clearly display:

"Research decision-support only.

This system does not provide a medical diagnosis."

For sources display:

Document

Page

Retrieved evidence

Make source cards clickable.

==================================================

7. MEDICAL KNOWLEDGE BASE

==================================================

Create an interactive literature section.

Display:

ABCD-ABCDE evidence

Dermoscopy + ABCD

Explainable Melanoma Diagnosis paper

AAD clinical guideline

Show:

- document type

- pages

- evidence level

- topic

Add search/filter animation.

==================================================

8. SYSTEM ARCHITECTURE

==================================================

Create an animated architecture diagram:

USER IMAGE

↓

SEGMENTATION

↓

ABCD FEATURES

↓

MODEL PREDICTION

↓

AGENT

↓

QUERY GENERATION

↓

QDRANT

↓

MEDICAL LITERATURE

↓

EVIDENCE EVALUATION

↓

GEMINI

↓

FINAL REPORT

Use animated glowing connection lines.

==================================================

9. RESEARCH METRICS

==================================================

Create a research metrics section.

Display cards for:

Documents indexed

Chunks indexed

Embedding model

Vector database

LLM

Retrieval pipeline

Evidence sources

Use values supplied by the backend where possible.

Do not fabricate performance metrics.

==================================================

10. FOOTER

==================================================

Include:

MelanomaAI

Explainable Agentic RAG Research System

Research Prototype

Not a Medical Diagnosis

==================================================

UX REQUIREMENTS

==================================================

- Fully responsive

- Desktop-first

- Mobile compatible

- Smooth scrolling

- Framer Motion animations

- 3D hero

- Micro-interactions

- Loading states

- Error states

- Empty states

- Accessible buttons

- Keyboard accessible

- Professional typography

- No excessive animation

Use reusable React components.

Create clean component architecture.

Do not put everything into one file.

==================================================

BACKEND INTEGRATION

==================================================

The existing backend will run at:

http://127.0.0.1:8000

The frontend should be prepared to call:

POST /analyze-case

The expected request structure is:

{

  "case_id": "IMAGE-TEST-001",

  "prediction": "Melanoma",

  "confidence": 0.91,

  "abcd_metrics": {

    "asymmetry_index": 17.26,

    "border_irregularity_score": 0.47,

    "color_variation_score": 35.17,

    "diameter_mm": null

  }

}

The backend returns an evidence-grounded report.

Create a clean API service layer so the backend URL can be changed

through an environment variable.

Do not expose Gemini API keys, Qdrant credentials or any secret

in the frontend.

==================================================

IMPORTANT

==================================================

Generate the complete frontend.

Prioritize visual quality, professional animations and clear

communication of the Agentic RAG pipeline.

The final result should look like a serious AI research product

demonstration suitable for a university project presentation,

research presentation and portfolio.

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/0072b8d5-a8df-4178-82a5-8f1d070f7cc9).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
