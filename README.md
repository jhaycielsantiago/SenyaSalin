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
[![Model: YOLOv11n](https://img.shields.io/badge/Model-YOLOv11%20Nano-purple.svg)](https://docs.ultralytics.com)
[![Dataset: 10,170 images](https://img.shields.io/badge/Dataset-10%2C170%20images-brightgreen.svg)](#-dataset)

</div>

---

## 🌟 Overview

SenyaSalin is an **open-source research infrastructure** project that translates Filipino Sign Language (FSL) gestures into Taglish (Tagalog-English) speech output. Rather than shipping a single application, SenyaSalin provides the **complete, documented workflow** that future developers and researchers can fork, extend, and build upon:

- 📦 A reusable **annotated image dataset** — 10,170 images across train/valid/test splits
- 🔧 A reproducible **YOLOv11 Nano object detection pipeline**
- 🗂️ An extensible **intent/response schema** mapping detections to Taglish phrases
- 🤖 A trained **YOLOv11n baseline model** with open weights and training recipe
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
| `data/` | Annotated image dataset — 6,859 train / 590 valid / 2,721 test (YOLO format) |
| `data/data.yaml` | Dataset config for Ultralytics training |
| `gesture_intents.json` | Intent schema: gesture class → semantic intent → Taglish phrase |
| `train.py` | YOLOv11n training script |
| `evaluate.py` | Benchmark evaluation script (mAP@50, mAP@50-95, precision, recall) |
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
YOLOv11 Nano Object Detection
    │  (bounding box + class label + confidence score)
    ▼
Gesture Class Prediction
    │
    ▼
Intent Mapping (gesture → semantic intent → Taglish phrase)
    │
    ▼
Taglish TTS Output (gTTS)
```

Each stage is independently replaceable — swap in a larger YOLO variant, extend the intent schema with new gesture classes, or add a new language output without touching other parts of the pipeline.

---

## 🗄️ Dataset

### Splits

The dataset contains **10,170 annotated images** of FSL gestures, split as follows:

| Split | Images | Percentage |
|---|---|---|
| **Train** | 6,859 | 67.4% |
| **Valid** | 590 | 5.8% |
| **Test** | 2,721 | 26.7% |
| **Total** | **10,170** | 100% |

Images are annotated in **YOLO format** — one `.txt` file per image containing bounding box coordinates and class labels.

### Annotation Format

Each annotation file contains one line per detected gesture instance:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized to `[0, 1]` relative to image dimensions. Example:

```
3 0.512 0.489 0.304 0.672
0 0.231 0.315 0.198 0.401
```

### Dataset Directory Structure

```
data/
├── train/
│   ├── images/   # 6,859 images
│   └── labels/   # 6,859 YOLO annotation files
├── valid/
│   ├── images/   # 590 images
│   └── labels/   # 590 YOLO annotation files
└── test/
    ├── images/   # 2,721 images
    └── labels/   # 2,721 YOLO annotation files
```

### Intent Schema Format

Each detected gesture class maps to a semantic intent and Taglish output phrase in `gesture_intents.json`:

```json
{
  "PAIN":     { "intent": "medical_discomfort",  "taglish": "Masakit po ito." },
  "HELP":     { "intent": "request_assistance",  "taglish": "Kailangan ko po ng tulong." },
  "MEDICINE": { "intent": "request_medicine",    "taglish": "Kailangan ko po ng gamot." }
}
```

> **Privacy:** No faces or identifying backgrounds appear in the dataset. All data is released under **CC BY 4.0**.

---

## 🤖 Model

### YOLOv11 Nano (Object Detection)

SenyaSalin uses **YOLOv11 Nano (`yolo11n`)** as its primary gesture detection model — the lightest variant in the YOLOv11 family, optimized for real-time inference on CPU and edge devices.

| Property | Value |
|---|---|
| Architecture | YOLOv11 Nano |
| Task | Object Detection |
| Input | Image / video frame |
| Output | Bounding box + class label + confidence score |
| Framework | [Ultralytics](https://docs.ultralytics.com) |
| Weights format | `.pt` (PyTorch) |

**Why YOLOv11 Nano?**
- Lightweight enough for real-time inference without a GPU — deployable on clinic tablets, kiosks, and low-end hardware
- Single-stage detection means lower latency versus two-stage pipelines
- The Nano variant trades some accuracy for speed, making it a practical baseline; larger YOLOv11 variants (Small, Medium, Large, XL) can be swapped in using the same training recipe

### Training Configuration

Default training setup (see `train.py`):

```yaml
model:   yolo11n.pt      # pretrained nano weights
data:    data/data.yaml  # dataset config with split paths
epochs:  100
imgsz:   640
batch:   16
device:  cpu             # or 'cuda' if GPU available
```

### Reference Baselines (prior work on similar FSL tasks)

| Model | Accuracy (approx.) |
|---|---|
| KNN on landmarks | 75–85% |
| Random Forest on landmarks | 80–90% |
| LSTM on landmark sequences | 85–95% |
| CNN on static images | ~97–98% |
| **YOLOv11n (this project)** | *report after training* |

Future contributors are encouraged to benchmark larger YOLO variants or custom architectures against the YOLOv11n baseline using the provided benchmark protocol.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip
- (Optional) CUDA-capable GPU for faster training
- (Optional) Docker

### Installation

**Option A — pip:**

```bash
git clone https://github.com/jhaycielsantiago/SenyaSalin
cd senyasalin
pip install -r requirements.txt
```

Key dependencies:

```
ultralytics>=8.0.0
torch>=1.8.0
torchvision
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

### 1. Prepare the Dataset Config

Ensure `data/data.yaml` points to your split directories:

```yaml
path: data/
train: train/images
val:   valid/images
test:  test/images

nc: 15   # number of gesture classes
names:
  - PAIN
  - HELP
  - MEDICINE
  - YES
  - NO
  # ... (fill in all 15 class names)
```

### 2. Train the YOLOv11 Nano Model

```bash
python train.py \
  --model yolo11n.pt \
  --data data/data.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --out runs/train/senyasalin_v1
```

Or directly via Ultralytics CLI:

```bash
yolo detect train \
  model=yolo11n.pt \
  data=data/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  name=senyasalin_v1
```

### 3. Evaluate on the Test Set

```bash
python evaluate.py \
  --weights runs/train/senyasalin_v1/weights/best.pt \
  --data data/data.yaml \
  --split test
```

Or via Ultralytics CLI:

```bash
yolo detect val \
  model=runs/train/senyasalin_v1/weights/best.pt \
  data=data/data.yaml \
  split=test
```

Outputs mAP@50, mAP@50-95, precision, recall, and per-class metrics.

### 4. Run Inference on a Video / Webcam

```bash
# On a video file
python infer.py --weights best.pt --source video.mp4

# Live webcam (device 0)
python infer.py --weights best.pt --source 0
```

Detected gestures are mapped through `gesture_intents.json` and spoken aloud via gTTS.

---

## 📊 Benchmark Protocol

To ensure fair and reproducible comparison across models, SenyaSalin defines a fixed benchmark:

| Split | Images | Purpose |
|---|---|---|
| Train | 6,859 | Model training |
| Valid | 590 | Hyperparameter tuning / early stopping |
| Test | 2,721 | Final held-out evaluation (report these numbers) |

**Primary metrics:**
- **mAP@50** — mean Average Precision at IoU threshold 0.50 *(main headline metric)*
- **mAP@50-95** — mean Average Precision averaged across IoU thresholds 0.50–0.95
- **Precision** and **Recall** at the optimal confidence threshold
- **Per-class AP** for granular analysis

When publishing results using this dataset, always evaluate on the **test split only** and report all four metrics above for comparability.

---

## 🛠️ Use Cases

SenyaSalin is designed as building-block infrastructure. Here are example applications:

**Healthcare Tablet App** — Deploy at clinic triage desks. A Deaf patient signs `MEDICINE`; the nurse hears *"Kailangan ko po ng gamot"* immediately, bridging the communication gap without a human interpreter.

**Smart Gloves / Wearables** — Hardware engineers building haptic FSL gloves can reuse the landmark dataset and intent schema directly, focusing on hardware while plugging into the existing AI pipeline.

**Public Service Kiosks** — Government centers can embed SenyaSalin so Deaf citizens can interact via FSL at information desks, aligning with RA 11106's mandate to make FSL available in public institutions.

**Academic Benchmarking** — Computer vision researchers can compare new sign-language models against SenyaSalin's YOLOv11n baseline and report gains using the standard benchmark.

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

1. Collect image samples of the sign (hands/arms only — no faces, min. 50–100 images recommended)
2. Annotate bounding boxes using [Roboflow](https://roboflow.com) or [LabelImg](https://github.com/heartexlabs/labelImg) in YOLO format
3. Export annotations and place images/labels under `data/train/`, `data/valid/`, and `data/test/`
4. Add the new class name to `data/data.yaml` and increment `nc`
5. Add the gesture's intent and Taglish phrase to `gesture_intents.json`
6. Re-run `train.py` and `evaluate.py` to verify performance
7. Open a pull request with the new class details and your updated benchmark metrics

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

| Name |
|---|
| Kent Garcia |
| Denver Reyes |
| Caleb Ezra Abiera |
| Jhayciel Santiago |

> Built for **Tinig Sa Liwanag** — a hackathon project case advancing speech technology for Philippine languages.
> Polytechnic University of the Philippines - San Pedro

---

## 📚 References

- Philippine Statistics Authority. (2020). *Census data on hearing difficulties.*
- Republic Act 11106 — Filipino Sign Language Act (2018)
- Republic Act 7277 — Magna Carta for Persons with Disabilities
- Ultralytics YOLOv11 documentation — [docs.ultralytics.com](https://docs.ultralytics.com)
- Google MediaPipe Hands documentation — [mediapipe.dev](https://mediapipe.dev)
- Roboflow FSL Dataset (~11k images, 15 classes)
- gTTS — Google Text-to-Speech for Filipino

---

<p align="center">
  Made with ❤️ for the Filipino Deaf community. <br/>
  <em>"Bawat kamay ay may salita." — Every hand has a voice.</em>
</p>
