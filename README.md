<p align="center">
  <img src="SensyaSalin_Logo.png" alt="SenyaSalin Logo" width="100"/>
</p>

<div align="center">

# SenyaSalin

> **Open Multimodal Filipino Sign Language Communication Benchmark**
>
> Bridging Filipino Sign Language (FSL) with spoken Taglish through open datasets, reproducible pipelines, and baseline models — built for researchers, developers, and accessibility advocates.

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-orange.svg)](https://mediapipe.dev)

</div>

---

## 🌟 Overview

SenyaSalin is an **open-source research infrastructure** project that translates Filipino Sign Language (FSL) gestures into Taglish (Tagalog-English) speech output. Rather than shipping a single application, SenyaSalin provides the **complete, documented workflow** that future developers and researchers can fork, extend, and build upon:

- 📦 A reusable **gesture landmark dataset** — normalized 21-point 3D hand coordinates per frame
- 🔧 A reproducible **MediaPipe Hands extraction pipeline**
- 🗂️ An extensible **intent/response schema** mapping gestures to Taglish phrases
- 🤖 **Baseline recognition models** (KNN, Random Forest, LSTM) with open training recipes
- 📊 A defined **benchmark protocol** for fair model comparison

This project was built for the **Tinig Sa Liwanag** hackathon track, which challenges teams to contribute foundational infrastructure that advances speech and communication technology for Philippine languages.

---

## 🇵🇭 Problem Statement

Millions of Deaf and hard-of-hearing (DHH) Filipinos face severe communication barriers in healthcare, education, and government services.

- The Philippine Statistics Authority (PSA) estimates **~1.78 million Filipinos** had hearing difficulties in 2020.
- **Republic Act 11106 (2018)** legally recognizes FSL as the national sign language, yet implementation remains inadequate.
- Most hearing Filipinos lack FSL training, forcing Deaf individuals to rely on written notes or improvised gestures during critical interactions (e.g., medical consultations).
- Open FSL datasets and developer tools for sign recognition are **scarce** in the Philippines.

SenyaSalin addresses this gap by building an open, extensible pipeline that links FSL gestures to natural Filipino language output — infrastructure any developer can build on without starting from scratch.

---

## 📦 Open Artifacts

SenyaSalin ships the following reusable components:

| Artifact | Description |
|---|---|
| `data/landmarks.json` | Gesture landmark dataset — normalized 21-point 3D hand coordinates per frame |
| `extract_landmarks.py` | MediaPipe Hands-based landmark extraction pipeline |
| `gesture_intents.json` | Intent schema: gesture class → semantic intent → Taglish phrase |
| `train.py` | Baseline model training script (KNN, RF, LSTM) |
| `evaluate.py` | Benchmark evaluation script (accuracy, precision, recall, F1) |
| `infer.py` | Real-time inference on video/webcam with TTS output |
| `Dockerfile` / `requirements.txt` | Locked dependency environment |
| `.github/workflows/` | CI template for automated training on push |
| `CONTRIBUTING.md` | Guide for adding new gesture classes |

---

## 🏗️ Architecture

SenyaSalin is a modular pipeline from webcam video to spoken Taglish output:

```
Webcam / Image Input
    │
    ▼
MediaPipe Hands Detection
    │  (21 3D hand landmarks per frame)
    ▼
Landmark Normalization
    │  (translation- & scale-invariant)
    ▼
Gesture Classifier (KNN / Random Forest / LSTM)
    │
    ▼
Intent Mapping (gesture → semantic intent → Taglish phrase)
    │
    ▼
Taglish TTS Output (gTTS)
```

Each stage is independently replaceable — swap in a better classifier, extend the intent schema with new gesture classes, or add a new language output without touching other parts of the pipeline.

---

## 🤖 Model

### MediaPipe Hands + Classifier

SenyaSalin uses **Google MediaPipe Hands** for real-time hand landmark detection, feeding 21 normalized 3D keypoints per frame into a lightweight classifier.

| Property | Value |
|---|---|
| Detection | MediaPipe Hands |
| Keypoints | 21 landmarks (x, y, z) per hand |
| Normalization | Wrist-origin subtraction + max-distance scaling |
| Classifiers | KNN, Random Forest, LSTM |
| TTS | gTTS (Filipino) |

**Baseline performance on 15-class gesture vocabulary:**

| Model | Accuracy (approx.) | Precision | Recall |
|---|---|---|---|
| KNN (k=3) | 75–85% | ~0.80 | ~0.80 |
| Random Forest (100 trees) | 80–90% | ~0.85 | ~0.85 |
| LSTM (PyTorch) | 85–95% | ~0.90 | ~0.90 |
| CNN on static images *(reference)* | ~97–98% | — | — |

Future contributors are encouraged to benchmark new architectures against these baselines using the provided benchmark protocol.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip
- (Optional) Docker

### Installation

**Option A — pip:**

```bash
git clone https://github.com/jhaycielsantiago/SenyaSalin.git
cd senyasalin
pip install -r requirements.txt
```

Key dependencies:

```
mediapipe>=0.10.0
scikit-learn
torch>=1.8.0
opencv-python
gtts
```

**Option B — Docker:**

```bash
docker build -t senyasalin .
docker run --rm senyasalin
```

---

## 💻 Usage

### 1. Extract Landmarks from Video

```bash
python extract_landmarks.py \
  --input video_clips/ \
  --output data/landmarks.json
```

Processes all video files in `video_clips/`, runs MediaPipe Hands on each frame, normalizes the 21-point landmarks, and saves structured records to `data/landmarks.json`.

### 2. Train a Baseline Model

**KNN:**
```bash
python train.py \
  --model knn \
  --data data/landmarks.json \
  --out models/knn_model.pkl
```

**Random Forest:**
```bash
python train.py \
  --model rf \
  --data data/landmarks.json \
  --out models/rf_model.pkl
```

**LSTM:**
```bash
python train.py \
  --model lstm \
  --data data/landmarks.json \
  --out models/lstm_model.pt \
  --epochs 20
```

### 3. Evaluate a Model

```bash
python evaluate.py \
  --model models/rf_model.pkl \
  --data data/landmarks.json
```

Outputs accuracy, per-class precision/recall, F1, and a confusion matrix.

### 4. Run Inference on a Video / Webcam

```bash
# On a video file
python infer.py --model models/rf_model.pkl --source video.mp4

# Live webcam (device 0)
python infer.py --model models/rf_model.pkl --source 0
```

Detected gestures are mapped through `gesture_intents.json` and spoken aloud via gTTS.

---

## 📊 Benchmark Protocol

To ensure fair and reproducible comparison across models, SenyaSalin defines a fixed benchmark:

| Split | Ratio | Purpose |
|---|---|---|
| Train | 60% | Model training |
| Valid | 20% | Hyperparameter tuning / early stopping |
| Test | 20% | Final held-out evaluation (report these numbers) |

**Primary metrics:**
- **Accuracy** — overall classification accuracy *(main headline metric)*
- **Precision & Recall** — per-class and macro-averaged
- **F1-score** — harmonic mean of precision and recall
- **Confusion matrix** — for granular per-class analysis

When publishing results using this dataset, always evaluate on the **test split only** and report all metrics above for comparability.

---

## 🛠️ Use Cases

SenyaSalin is designed as building-block infrastructure. Here are example applications:

**Healthcare Tablet App** — Deploy at clinic triage desks. A Deaf patient signs `MEDICINE`; the nurse hears *"Kailangan ko po ng gamot"* immediately, bridging the communication gap without a human interpreter.

**Smart Gloves / Wearables** — Hardware engineers building haptic FSL gloves can reuse the landmark dataset and intent schema directly, focusing on hardware while plugging into the existing AI pipeline.

**Public Service Kiosks** — Government centers can embed SenyaSalin so Deaf citizens can interact via FSL at information desks, aligning with RA 11106's mandate to make FSL available in public institutions.

**Academic Benchmarking** — Computer vision researchers can compare new sign-language models against SenyaSalin's baselines and report gains using the standard benchmark.

**FSL Education Tools** — Interactive FSL tutoring apps can use SenyaSalin as the recognition engine, giving real-time feedback to learners.

---

## 🗺️ Roadmap

- [ ] **Scale the dataset** — more gesture classes (numbers, emotions, questions) and more samples per class via crowdsourced contributions
- [ ] **Continuous signing** — sentence-level recognition using Transformers or Hidden Markov Models
- [ ] **Facial & body cues** — integrate MediaPipe Face Mesh and Pose for FSL grammar markers (e.g., negation, intensity)
- [ ] **Multimodal ASR** — combine sign recognition with Taglish speech-to-text for mixed Deaf-hearing conversations
- [ ] **Regional languages** — extend Taglish output to Cebuano, Ilocano, and other Philippine languages
- [ ] **Public benchmark leaderboard** — publish dataset on Hugging Face / Roboflow with an open leaderboard
- [ ] **Bi-directional tools** — speech-to-sign translation via animated avatar for hearing users

---

## 🤝 Contributing

We welcome contributions of new gesture classes, improved models, and dataset expansions.

**To add a new gesture class:**

1. Record video clip(s) of the sign (hands/arms only — no faces)
2. Run `extract_landmarks.py` on the new clips
3. Append the output records to `data/landmarks.json`
4. Add the gesture's intent and Taglish phrase to `gesture_intents.json`
5. Re-run `train.py` and `evaluate.py` to verify performance
6. Open a pull request with the new class details and your updated benchmark metrics

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for full guidelines, labeling conventions, and anonymization requirements.

---

## ⚖️ Licensing & Ethics

| Component | License |
|---|---|
| Code | [MIT](LICENSE) |
| Dataset & Schema | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

**Data ethics commitments:**
- Only hands and arms are captured — no faces, voices, or identifying backgrounds
- Signer IDs are pseudonymized (e.g., `S01`, `S02`) — no names or PII in any file
- Informed consent was obtained from all signers
- Signers represent diverse demographics (age, gender, region) to reduce bias

This project aligns with the spirit of **RA 11106** (Filipino Sign Language Act) and **RA 7277** (Magna Carta for Persons with Disabilities). FSL technology should be accessible, not proprietary.

---

## 👥 Team

**SenyaSalin** was developed by:

| Name | Role |
|---|---|
| Kent Garcia | Model Development |
| Denver Reyes | Demo Video Production |
| Caleb Ezra Abiera | Documentation |
| Jhayciel Santiago | Documentation |

> Built for **Tinig Sa Liwanag** — a hackathon project case advancing speech technology for Philippine languages.
> Polytechnic University of the Philippines - San Pedro

---

# 📚 References

- Philippine Statistics Authority. (2020). *Census data on hearing difficulties.*
- Republic Act 11106 — Filipino Sign Language Act (2018)
- Republic Act 7277 — Magna Carta for Persons with Disabilities
- Google MediaPipe Hands documentation — [mediapipe.dev](https://mediapipe.dev)
- Roboflow FSL Dataset (~11k images, 15 classes)
- gTTS — Google Text-to-Speech for Filipino

---

<p align="center">
  Made with ❤️ for the Filipino Deaf community. <br/>
  <em>"Bawat tinig ng Pilipino, patungo sa liwanag." — Every Filipino voice, into the light.</em>
</p>
