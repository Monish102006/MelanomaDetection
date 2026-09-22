# Phase 1 — Core AI Melanoma Diagnosis Engine

An AI-powered melanoma diagnosis system that classifies dermoscopic skin lesion images, segments lesion boundaries, and extracts clinically relevant morphological features.

## Architecture

<img width="1536" height="1024" alt="Architecture" src="https://github.com/user-attachments/assets/76787992-de9d-4d51-a7e7-6cab9008c991" />

## Setup

### 1. Install PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install SegFormer


### 4. Download Checkpoints

- **SegFormer**: Download `SegFormer_vit_b.pth` from the [SegFormer repository](https://github.com/bowang-lab/SegFormer) → place in `checkpoints/`
- **Swin V2**: Train on Kaggle using `notebooks/train_swin_classifier.ipynb` → download `swin_best.pth` → place in `checkpoints/`

### 5. Prepare Dataset

Organize your ISIC dataset in the following structure:

```
dataset/
├── classification/
│   ├── train/
│   │   ├── melanoma/
│   │   └── benign/
│   ├── val/
│   │   ├── melanoma/
│   │   └── benign/
│   └── test/
│       ├── melanoma/
│       └── benign/
└── segmentation/
    ├── images/
    └── masks/
```

## Usage

```bash
# Validate dataset quality
python main.py validate-data

# Train classifier (or use Kaggle notebook)
python main.py train

# Evaluate classifier
python main.py evaluate-cls

# Evaluate segmentation
python main.py evaluate-seg

# Run full diagnosis on an image
python main.py diagnose path/to/image.jpg

# Generate Grad-CAM visualization
python main.py gradcam path/to/image.jpg
```

## Output

The diagnosis engine produces a `DiagnosisResult` JSON:

```json
{
    "prediction": "Melanoma",
    "confidence": 95.8,
    "probabilities": {"benign": 0.042, "melanoma": 0.958},
    "features": {
        "asymmetry": {"score_label": "High", "score_numeric": 0.42},
        "border": {"score_label": "Irregular", "score_numeric": 0.68},
        "color": {"score_label": "Multiple Colors", "score_numeric": 0.75}
    }
}
```

## Project Structure

| Module | Purpose |
|---|---|
| `src/data/` | Dataset classes, augmentation, validation |
| `src/classification/` | Swin V2 model, training, inference |
| `src/segmentation/` | SegFormer wrapper, lesion locator |
| `src/features/` | ABC feature extraction with registry pattern |
| `src/engine/` | Inference pipeline + Core Diagnosis Engine |
| `src/evaluation/` | Classification and segmentation metrics |
| `src/explainability/` | Grad-CAM development utility |
