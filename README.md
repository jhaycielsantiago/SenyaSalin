# SenyaSalin

> **"SenyaSalin — let the voices of light in."**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green.svg)](https://mediapipe.dev/)
[![Open Schema](https://img.shields.io/badge/Gesture%20Schema-CC%20BY%204.0-lightblue.svg)]()
[![Status](https://img.shields.io/badge/Status-v0.1%20Released-brightgreen.svg)]()

---

## Group Information

| Field | Details |
|---|---|
| **Group Name** | EXPLICIT |
| **Challenge Track** | Tinig sa Liwanag |
| **Project Case** | Accessibility-Focused Speech Technology for Philippine Languages |
| **Project Name** | SenyaSalin |

### Team Members

| Name |
|---|
| **Kent Garcia** |
| **Caleb Ezra Abiera** |
| **Denver Reyes** |
| **Jhayciel Santiago** |

---

## Project Overview

**Deaf and hard-of-hearing Filipinos** continue to encounter communication barriers when accessing healthcare, education, public services, and emergency assistance (PSA, 2020). While the Filipino Sign Language Act recognizes FSL as the national sign language of the Filipino Deaf community, access to qualified interpreters and accessible communication support remains a continuing challenge. Open and standardized FSL datasets, benchmarks, and developer resources remain limited, making it difficult for researchers and developers to build inclusive accessibility technologies.

When a Deaf Filipino walks into a clinic and needs to say *"Masakit po ito"* — they are often left to write on a piece of paper and hope for the best.

**SenyaSalin exists to change that.**

SenyaSalin is an **open-source, accessibility-first pipeline** that recognizes essential Filipino Sign Language (FSL) gestures through a standard webcam and converts them into natural **Taglish speech** in real time — no special hardware, no sign language knowledge required from the hearing person, and no cost to deploy.

Built using computer vision and speech synthesis technologies, SenyaSalin helps bridge communication between Deaf and hearing Filipinos in high-impact situations such as healthcare consultations, emergency response, and public service interactions.

Beyond the application itself, the project introduces an **extensible gesture schema and reproducible recognition pipeline** — a foundation that researchers, NGOs, and developers can fork, extend, and build upon without starting from zero.

### What Makes SenyaSalin Different

Most existing FSL recognition projects output a simple English label — `"HELP"`, `"WATER"`, `"STOP"`. That label is not a conversation. It does not reflect how Filipinos actually communicate.

SenyaSalin outputs **contextual Taglish phrases** — the natural code-switched Filipino-English that real Filipinos use every day:

| Sign | Generic Label Output | SenyaSalin Output |
|---|---|---|
| HELP | `"HELP"` | *"Kailangan ko po ng tulong."* |
| MEDICINE | `"MEDICINE"` | *"Kailangan ko po ng gamot."* |
| PAIN | `"PAIN"` | *"Masakit po ito."* |

This is not a cosmetic difference. Outputting in the natural Taglish register makes the speech output **immediately usable** in real Philippine contexts — and directly contributes to Taglish speech technology, the explicit focus of this challenge.

### Core Pipeline

```
FSL Gesture (Webcam)
      │
      ▼
MediaPipe Hands — 21 landmarks extracted per hand (x, y, z)
      │
      ▼
Keypoint Normalization — wrist-relative, unit-scale vector (126 dims)
      │
      ▼
KNN Classifier — matches gesture to learned sign profile
      │
      ▼
schemas/gestures.json + mappings.json — label → intent → Taglish phrase
      │
      ▼
gTTS — synthesizes Filipino/Taglish speech
      │
      ▼
Audio Output — spoken aloud for the hearing person nearby
```

> **Scope Statement:** SenyaSalin supports **15 essential signs** in v0.1. It does not claim full sentence-level FSL translation. It delivers reliable, contextually appropriate Taglish speech for the highest-need accessibility scenarios and provides open infrastructure for others to extend it further.

---

## Features

### Implemented — v0.1

- Real-time FSL gesture recognition via standard webcam
- 15 essential signs across emergency, health, and daily communication
- **Contextual Taglish phrase output** (not raw English labels)
- Filipino text-to-speech audio via gTTS
- On-screen confidence overlay — gesture label, Taglish phrase, confidence %
- **Threshold-gated output** — below 70% confidence, the system prompts retry
- JSON-driven phrase mapping — add new signs without retraining
- FastAPI backend + React/TypeScript demo frontend
- **Open Gesture Schema** (CC BY 4.0) — documented, structured, contribution-ready
- MIT-licensed codebase — clone, fork, and deploy freely

### Roadmap

- Offline TTS via eSpeak-NG Filipino (full offline capability)
- Expanded vocabulary: 50 signs across health, education, and transport domains
- Community contribution workflow — add new signs via pull request
- Deaf community validation of gesture schema (target: NFAD partnership)
- Regional FSL variant documentation (Manila, Cebu, Davao dialects differ)
- Browser-based version via TensorFlow.js (zero installation)
- Benchmark dataset release for FSL-to-Taglish-speech evaluation
- Future multilingual support: Cebuano, Ilocano, Hiligaynon

---

## Demo Scenario

A Deaf individual enters a clinic and needs assistance.

1. The user performs the FSL sign for HELP.
2. SenyaSalin detects the gesture through a standard webcam.
3. The system recognizes it and generates the phrase: *"Kailangan ko po ng tulong."*
4. The phrase is spoken aloud through the device speaker.

The hearing person nearby immediately understands the user's request — no interpreter required, no prior FSL knowledge needed.

---

## MVP Vocabulary — 15 Essential Signs

Signs were selected based on three criteria: **highest accessibility risk** (where Deaf-hearing communication failure has the most serious consequence), **gestural distinctiveness** (visually separable by a landmark classifier), and **scenario coverage** across the settings where Deaf Filipinos most need communication support.

| # | FSL Sign | Taglish Output | Scenario |
|---|---|---|---|
| 1 | HELP | "Kailangan ko po ng tulong." | Emergency |
| 2 | WATER | "Pwede po bang humingi ng tubig?" | Health / Daily |
| 3 | PAIN / HURT | "Masakit po ito." | Health |
| 4 | THANK YOU | "Maraming salamat po." | Social |
| 5 | YES | "Oo po." | Confirmation |
| 6 | NO | "Hindi po." | Confirmation |
| 7 | STOP | "Tigil po." | Safety |
| 8 | TOILET / BATHROOM | "Nasaan po ang CR?" | Daily Need |
| 9 | FOOD / EAT | "Gutom na po ako." | Daily Need |
| 10 | MEDICINE | "Kailangan ko po ng gamot." | Health / Emergency |
| 11 | UNDERSTAND | "Naiintindihan ko po." | Communication |
| 12 | REPEAT / AGAIN | "Ulit po, pakiulit." | Communication |
| 13 | NAME | "Ano po ang pangalan ninyo?" | Social |
| 14 | CALL (phone) | "Tumawag po kayo ng tulong." | Emergency |
| 15 | OKAY / GOOD | "Okay lang po ako." | Communication |

---

## Benchmark Results — v0.1

All three baseline models were trained and evaluated on the same stratified split (seed=42).
Dataset: 6,750 samples across 15 classes (450 per class), split 70/15/15.

| Model | Accuracy | F1 (weighted) | Inference (mean) |
|---|---|---|---|
| KNN (k=5, distance-weighted) | **87.4%** | 0.874 | 2.1 ms |
| MLP (256→128, ReLU) | 89.4% | 0.895 | 1.3 ms |
| Random Forest (n=200) | **91.8%** | 0.918 | 8.4 ms |

KNN is the **default model** for the real-time demo — its 2ms latency keeps the overlay smooth at 30fps and its predictions are directly interpretable (nearest training samples are inspectable).
Random Forest achieves the highest accuracy and is recommended for evaluation and comparison.

Hardest classes across all models: **REPEAT** and **UNDERSTAND** (overlapping two-hand configurations in the normalized landmark space). Easiest: **THANK_YOU**, **YES**, **NO**.

Full per-class metrics and training logs: [`benchmark/results/`](benchmark/results/)

---

## Technologies Used

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.9+ | Core runtime |
| Video Capture | OpenCV | Webcam frame capture and visual overlay |
| Hand Tracking | MediaPipe Hands | 21-point hand landmark extraction (x, y, z per point) |
| Classification | scikit-learn | KNN, Random Forest, MLP gesture classifiers |
| Numerical Processing | NumPy | Keypoint normalization, feature vectors |
| Text-to-Speech | gTTS | Filipino/Taglish speech synthesis |
| Audio Playback | pygame | Cross-platform audio output |
| Backend API | FastAPI + Uvicorn | HTTP API serving the recognition pipeline |
| Frontend | React + TypeScript + Vite | Browser-based webcam demo |
| Schema Format | JSON | Open gesture schema and Taglish phrase mappings |

---

## Setup Instructions

### Prerequisites

- Python 3.9+
- Node.js 18+ (frontend only)
- A working webcam (built-in or USB)
- Internet connection (required for gTTS speech synthesis)

### 1 — Clone the Repository

```bash
git clone https://github.com/jhaycielsantiago/SenyaSalin.git
cd SenyaSalin
```

### 2 — Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### 4 — Run the Webcam Demo

The v0.1 models are pre-trained. Launch the webcam demo directly:

```bash
python main.py
```

Position your hand clearly in frame and hold one of the 15 supported FSL signs for 1–2 seconds.
The recognized gesture, Taglish phrase, and confidence level appear on screen.
If confidence is above 70%, the phrase is spoken aloud.

Press `Q` to quit.

### 5 — Run the Web App (Optional)

Start the FastAPI backend:

```bash
uvicorn app.backend.main:app --reload
```

Start the React frontend (in a separate terminal):

```bash
cd app/frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

### 6 — Retrain from Scratch (Optional)

If you have collected video data in `data/raw/<LABEL>/`:

```bash
# Extract landmarks
python -m extraction.video_processor --raw-dir data/raw --output-dir data/processed

# Train all baseline models
python -m training.train --export-splits --seed 42

# Run benchmark evaluation
python -m benchmark.run_benchmark
```

### Troubleshooting

| Issue | Solution |
|---|---|
| Webcam not detected | Close any other app using the webcam. Try editing `CAMERA_INDEX` in `config.py`. |
| No audio output | Check system volume. Verify `pygame` installed correctly via `pip show pygame`. |
| Low recognition accuracy | Improve lighting. Ensure your full hand is visible in frame. Hold each sign steady for 1–2 seconds. |
| gTTS network error | Check your internet connection — gTTS requires internet to synthesize speech. |
| MediaPipe install fails | Run `pip install mediapipe --upgrade`. On Apple Silicon Macs, use a native ARM Python environment. |
| `No model found` error | Run `python -m training.train` first to generate `models/saved/knn.pkl`. |

---

## Open Gesture Schema

The gesture schema is SenyaSalin's most important open-source contribution. Every supported sign is defined in a documented, structured JSON format that any developer or researcher can read, validate, and extend — without modifying core code.

```json
{
  "sign_id": "FSL_HELP_001",
  "label": "HELP",
  "description": "Dominant hand in 'A' shape placed on top of non-dominant flat hand, lifted upward",
  "handshape": "A",
  "movement": "upward_lift",
  "location": "neutral_space",
  "intent": "request_assistance",
  "scenario_tags": ["emergency", "health"],
  "validation_status": "team_reviewed",
  "version": "0.1"
}
```

To add a new sign: see [`docs/annotation_guide.md`](docs/annotation_guide.md).
Full schema specification: [`docs/gesture_schema.md`](docs/gesture_schema.md).

All schema files are released under **CC BY 4.0** for maximum research and community reuse.

---

## Repository Structure

```
SenyaSalin/
├── README.md
├── LICENSE
├── requirements.txt
├── requirements-dev.txt
├── main.py                          # Webcam demo entry point
├── config.py                        # Runtime configuration (thresholds, camera, model)
│
├── schemas/
│   ├── gestures.json                # Open gesture schema (CC BY 4.0)
│   ├── mappings.json                # Intent → Taglish/Tagalog/English phrase mappings
│   ├── landmark_schema.json         # Structure spec for .npy landmark files
│   └── benchmark_config.json        # Benchmark parameters and dataset spec
│
├── extraction/
│   ├── landmark_extractor.py        # MediaPipe Hands wrapper
│   └── video_processor.py           # Batch video → .npy extraction CLI
│
├── preprocessing/
│   ├── normalizer.py                # Wrist-relative, unit-scale normalization
│   ├── feature_extractor.py         # Frame aggregation (mean/std/concat)
│   └── data_loader.py               # Stratified train/val/test splitting
│
├── training/
│   ├── base_trainer.py              # Abstract BaseTrainer
│   ├── knn_trainer.py               # K-Nearest Neighbors
│   ├── rf_trainer.py                # Random Forest
│   ├── mlp_trainer.py               # MLP Classifier
│   └── train.py                     # Training CLI
│
├── senyasalin/
│   ├── pipeline.py                  # GestureRecognizer — core inference path
│   └── mapper.py                    # Label → intent → phrase mapping
│
├── benchmark/
│   ├── evaluator.py                 # Test-set evaluation with timing
│   ├── metrics.py                   # Result formatting and saving
│   ├── run_benchmark.py             # Benchmark CLI
│   └── results/
│       ├── knn_v0.1.json            # KNN baseline results (87.4% accuracy)
│       ├── rf_v0.1.json             # Random Forest results (91.8% accuracy)
│       └── mlp_v0.1.json            # MLP results (89.4% accuracy)
│
├── app/
│   ├── backend/
│   │   ├── main.py                  # FastAPI app (health, recognize, gestures, benchmark)
│   │   └── schemas.py               # Pydantic request/response models
│   └── frontend/
│       ├── src/
│       │   ├── App.tsx
│       │   ├── components/GestureCamera.tsx
│       │   └── hooks/useGestureRecognizer.ts
│       └── package.json
│
├── models/
│   └── saved/
│       ├── knn.meta.json            # KNN class names and hyperparameters
│       ├── rf.meta.json             # RF class names and hyperparameters
│       └── mlp.meta.json            # MLP class names and hyperparameters
│       # .pkl model files are gitignored — train locally or download from releases
│
├── data/
│   ├── raw/                         # Raw video files (gitignored)
│   ├── processed/                   # Extracted .npy landmark files (gitignored)
│   └── benchmark/                   # Exported train/val/test split CSVs
│
├── docs/
│   ├── annotation_guide.md          # How to contribute new signs
│   ├── gesture_schema.md            # Full schema specification
│   └── architecture.md              # Technical architecture
│
├── notebooks/
│   └── explore_keypoints.ipynb      # PCA and landmark visualization
│
└── tests/
    ├── test_normalizer.py
    ├── test_feature_extractor.py
    └── test_mapper.py
```

---

## Impact

| Dimension | Detail |
|---|---|
| **Who it serves** | ~1–1.5 million Deaf and hard-of-hearing Filipinos |
| **Primary scenarios** | Medical consultations, emergency communication, public service counters |
| **Immediate value** | A Deaf Filipino can communicate a need in under 10 seconds with no interpreter |
| **Open-source value** | Any researcher or NGO can extend the schema and pipeline without rebuilding from scratch |
| **Language contribution** | Advances Taglish speech technology — code-switched Filipino-English output for accessibility |
| **Long-term potential** | Foundation for a community-maintained, growing FSL-to-speech resource for the Philippines |

---

## License

SenyaSalin source code is released under the **MIT License** — free to use, fork, and modify.

Gesture schema and annotation documentation (`schemas/`, `docs/`) are released under **CC BY 4.0** — free for research, commercial, and community use with attribution.

---

## Acknowledgments

- **MediaPipe** (Google) — hand landmark tracking technology
- **Filipino Deaf community** — the primary motivation and intended beneficiary of this work
- **NFAD** (National Federation of the Deaf Philippines) — target future collaboration partner
- **FEU ACM TechSprint** organizers — for making this hackathon possible

---

<div align="center">

**Team EXPLICIT** | FEU ACM TechSprint

**SenyaSalin** — *Let the voices of light in.*

Built with care for the Filipino Deaf community.

</div>
