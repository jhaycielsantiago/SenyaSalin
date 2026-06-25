# 🤟 SenyaSalin

> **"SenyaSalin — let the voices of light in."**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green.svg)](https://mediapipe.dev/)
[![Track](https://img.shields.io/badge/Track-Tinig%20sa%20Liwanag-purple.svg)]()
[![Open Schema](https://img.shields.io/badge/Gesture%20Schema-CC%20BY%204.0-lightblue.svg)]()
[![Status](https://img.shields.io/badge/Status-v0.1%20Released-brightgreen.svg)]()

---

## 👥 Group Information

| Field | Details |
|---|---|
| **Group Name** | EXPLICIT |
| **Challenge Track** | Tinig sa Liwanag |
| **Project Case** | Accessibility-Focused Speech Technology for Philippine Languages |
| **Project Name** | SenyaSalin |

### Team Members

| Name |
|---|---|
| **Kent Garcia** |
| **Caleb Ezra Abiera** |
| **Denver Reyes** |
| **Jhayciel Santiago** |

---

## 📌 Project Overview

**Deaf and hard-of-hearing Filipinos** continue to encounter communication barriers when accessing healthcare, education, public services, and emergency assistance (PSA, 2020). While the Filipino Sign Language Act recognizes FSL as the national sign language of the Filipino Deaf community, access to qualified interpreters and accessible communication support remains a continuing challenge. At the same time, open and standardized FSL datasets, benchmarks, and developer resources remain limited, making it difficult for researchers and developers to build inclusive accessibility technologies.

When a Deaf Filipino walks into a clinic and needs to say *"Masakit po ito"* — they are often left to write on a piece of paper and hope for the best.

**SenyaSalin exists to change that.**

SenyaSalin is an **open-source, accessibility-first pipeline** that recognizes essential Filipino Sign Language (FSL) gestures through a standard webcam and converts them into natural **Taglish speech** in real time — no special hardware, no sign language knowledge required from the hearing person, and no cost to deploy.
SenyaSalin is an **accessibility-focused web application** that converts essential Filipino Sign Language (FSL) gestures into understandable Taglish speech in real time.

Built using computer vision and speech synthesis technologies, SenyaSalin helps bridge communication between Deaf and hearing Filipinos in high-impact situations such as healthcare consultations, emergency response, and public service interactions.

Beyond the application itself, the project introduces an extensible gesture schema and reusable recognition pipeline that future developers and researchers can build upon.

But SenyaSalin is not just a recognition app. Its primary contribution to the open-source community is its **extensible gesture schema and reproducible pipeline** — a foundation that researchers, NGOs, and developers can fork, extend, and build upon without starting from zero.

### What Makes SenyaSalin Different

Most existing FSL recognition projects output a simple English label — `"HELP"`, `"WATER"`, `"STOP"`. That label is not a conversation. It does not reflect how Filipinos actually communicate.

SenyaSalin outputs **contextual Taglish phrases** — the natural code-switched Filipino-English that real Filipinos use every day:

| Sign | Generic Label Output | SenyaSalin Output |
|---|---|---|
| HELP | `"HELP"` | *"Kailangan ko po ng tulong."* |
| MEDICINE | `"MEDICINE"` | *"Kailangan ko po ng gamot."* |
| PAIN | `"PAIN"` | *"Masakit po ito."* |

This is not a cosmetic difference. Taglish is widely used in everyday communication across many Filipino contexts, making generated speech more natural and immediately understandable to hearing users. Outputting in their natural register makes the speech output **immediately usable** in real Philippine contexts — and directly contributes to Taglish speech technology, the explicit focus of this challenge.

### Core Pipeline

```
FSL Gesture (Webcam)
      │
      ▼
MediaPipe Hands — 21 landmarks extracted per hand (x, y, z)
      │
      ▼
Keypoint Normalization — translation and scale invariant vector
      │
      ▼
KNN Classifier — matches gesture to learned sign profile
      │
      ▼
gestures.json — maps sign label → intent → Taglish phrase
      │
      ▼
gTTS — synthesizes Filipino/Taglish speech
      │
      ▼
Audio Output — spoken aloud for the hearing person nearby
```

> **Scope Statement:** SenyaSalin supports **15 essential signs** in v0.1. It does not claim full sentence-level FSL translation. It claims to deliver reliable, contextually appropriate Taglish speech for the highest-need accessibility scenarios — and to provide the open infrastructure for others to extend it further.

---

## ✨ Features

### Implemented — v0.1 (Hackathon Release)

- ✅ Real-time FSL gesture recognition via standard webcam
- ✅ 15 essential signs across emergency, health, and daily communication
- ✅ **Contextual Taglish phrase output** (not raw English labels)
- ✅ Filipino text-to-speech audio via gTTS
- ✅ On-screen confidence overlay — shows gesture label, Taglish phrase, and confidence %
- ✅ **Threshold-gated output** — below 70% confidence, the system prompts retry instead of speaking a wrong phrase
- ✅ JSON-driven phrase mapping — add new signs without retraining
- ✅ **Open Gesture Schema** (CC BY 4.0) — documented, structured, contribution-ready
- ✅ MIT-licensed codebase — clone, fork, and deploy freely

### Roadmap — Post-Hackathon

- 🔲 Offline TTS via eSpeak-NG Filipino (full offline capability)
- 🔲 Expanded vocabulary: 50 signs across health, education, and transport domains
- 🔲 Community contribution workflow — add new signs via pull request
- 🔲 Deaf community validation of gesture schema (target: NFAD partnership)
- 🔲 Regional FSL variant documentation (Manila, Cebu, Davao dialects differ)
- 🔲 Browser-based version via TensorFlow.js (zero installation)
- 🔲 Benchmark dataset for FSL-to-Taglish-speech evaluation
- 🔲 Future multilingual support: Cebuano, Ilocano, Hiligaynon

---
## 🎥 Demo Scenario

A Deaf individual enters a clinic and needs assistance.

1. The user performs the FSL sign for HELP.
2. SenyaSalin detects the gesture through a webcam.
3. The system generates the phrase:

"Kailangan ko po ng tulong."

4. The phrase is spoken aloud through the device speaker.

This allows the hearing person nearby to immediately understand the user's request without requiring prior knowledge of Filipino Sign Language.
---

## 🤟 MVP Vocabulary — 15 Essential Signs

Signs were selected based on three criteria: **highest accessibility risk** (where Deaf-hearing communication failure has the most serious consequence), **gestural distinctiveness** (visually separable by a landmark classifier), and **scenario coverage** across the settings where Deaf Filipinos most need communication support.

| # | FSL Sign | Taglish Output | Scenario |
|---|---|---|---|
| 1 | HELP | "Kailangan ko po ng tulong." | 🚨 Emergency |
| 2 | WATER | "Pwede po bang humingi ng tubig?" | 🏥 Health / Daily |
| 3 | PAIN / HURT | "Masakit po ito." | 🏥 Health |
| 4 | THANK YOU | "Maraming salamat po." | 💬 Social |
| 5 | YES | "Oo po." | 💬 Confirmation |
| 6 | NO | "Hindi po." | 💬 Confirmation |
| 7 | STOP | "Tigil po." | 🛑 Safety |
| 8 | TOILET / BATHROOM | "Nasaan po ang CR?" | 🏢 Daily Need |
| 9 | FOOD / EAT | "Gutom na po ako." | 🏢 Daily Need |
| 10 | MEDICINE | "Kailangan ko po ng gamot." | 🏥 Health / Emergency |
| 11 | UNDERSTAND | "Naiintindihan ko po." | 💬 Communication |
| 12 | REPEAT / AGAIN | "Ulit po, pakiulit." | 💬 Communication |
| 13 | NAME | "Ano po ang pangalan ninyo?" | 💬 Social |
| 14 | CALL (phone) | "Tumawag po kayo ng tulong." | 🚨 Emergency |
| 15 | OKAY / GOOD | "Okay lang po ako." | 💬 Communication |

The initial vocabulary focuses on essential communication needs where timely understanding has the greatest impact, including emergency requests, healthcare interactions, and daily communication.

---

## 🛠️ Technologies Used

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.9+ | Core runtime |
| Video Capture | OpenCV (`opencv-python`) | Webcam frame capture and visual overlay |
| Hand Tracking | MediaPipe Hands | 21-point hand landmark extraction (x, y, z per point) |
| Classification | scikit-learn (KNN) | Gesture recognition on normalized keypoint vectors |
| Numerical Processing | NumPy | Keypoint normalization, vector operations |
| Text-to-Speech | gTTS (Google Text-to-Speech) | Filipino/Taglish speech synthesis |
| Audio Playback | pygame | Cross-platform audio output |
| Schema Format | JSON | Open gesture schema and Taglish phrase mappings |

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     SenyaSalin Pipeline                      │
├───────────┬──────────────┬──────────────┬────────────────────┤
│  CAPTURE  │   TRACKING   │  CLASSIFY    │      OUTPUT        │
│           │              │              │                    │
│  Webcam   │  MediaPipe   │    KNN on    │  Taglish Text      │
│  OpenCV   │  Hands       │  normalized  │  + Filipino TTS    │
│           │  21 kp/hand  │  keypoints   │  (gTTS / pygame)   │
│           │  x, y, z     │              │                    │
├───────────┴──────────────┴──────┬───────┴────────────────────┤
│                DATA LAYER       │                            │
│  gestures.json  ←→  intents.json│  Open schema (CC BY 4.0)  │
└─────────────────────────────────┴────────────────────────────┘
```

### Core Dependencies

```
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
gTTS>=2.3.2
pygame>=2.5.0
```

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.9 or higher
- A working webcam (built-in or USB)
- pip (Python package manager)
- Internet connection (only required for gTTS speech synthesis)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/jhaycielsantiago/SenyaSalin
cd senyasalin
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run SenyaSalin

```bash
python main.py
```

Your webcam will open. Position your hand clearly in frame and hold one of the supported FSL signs for 1–2 seconds. The recognized gesture, Taglish phrase, and confidence level will appear on screen. If confidence is above 70%, the phrase will be spoken aloud.

### Step 5 — Exit

Press `Q` while the webcam window is active to quit.

### Troubleshooting

| Issue | Solution |
|---|---|
| Webcam not detected | Close any other app using the webcam. Try `python main.py --camera 1` to switch camera index. |
| No audio output | Check system volume. Verify `pygame` installed correctly via `pip show pygame`. |
| Low recognition accuracy | Improve lighting. Ensure your full hand is visible in frame. Hold each sign steady for 1–2 seconds. |
| gTTS network error | Check your internet connection — gTTS requires internet to synthesize speech. |
| MediaPipe install fails | Run `pip install mediapipe --upgrade`. On Apple Silicon Macs, use a native ARM Python environment. |

---

## 📖 Open Gesture Schema

The gesture schema is SenyaSalin's most important open-source contribution. Every supported sign is defined in a documented, structured JSON format that any developer or researcher can read, validate, and extend — without modifying core code.

```json
{
  "sign_id": "FSL_HELP_001",
  "label": "HELP",
  "description": "Dominant hand in 'A' shape placed on top of non-dominant flat hand, lifted upward",
  "handshape": "A",
  "movement": "upward lift",
  "location": "neutral space",
  "taglish_output": "Kailangan ko po ng tulong.",
  "scenario_tags": ["emergency", "health"],
  "validation_status": "team-reviewed",
  "version": "0.1"
}
```

**To add a new sign:** See [`docs/annotation_guide.md`](./docs/annotation_guide.md). Adding a sign requires only a new JSON entry and a reference keypoint recording — no model retraining needed.

All schema documentation is released under **CC BY 4.0** for maximum research and community reuse.

---

## 📁 Repository Structure

```
senyasalin/
├── README.md                        # This file
├── LICENSE                          # MIT License
├── requirements.txt                 # Pinned Python dependencies
├── main.py                          # Entry point
├── config.py                        # Configuration (thresholds, camera index)
│
├── data/
│   ├── gestures.json                # Open gesture schema
│   ├── intents.json                 # Taglish phrase mappings
│   └── keypoints/                   # Reference keypoint recordings (.npy)
│
├── senyasalin/
│   ├── __init__.py
│   ├── capture.py                   # Webcam handler
│   ├── tracker.py                   # MediaPipe Hands integration
│   ├── classifier.py                # KNN gesture classifier
│   ├── mapper.py                    # Sign label → Taglish phrase
│   └── speaker.py                   # TTS synthesis and playback
│
├── notebooks/
│   └── explore_keypoints.ipynb      # Keypoint visualization
│
├── docs/
│   ├── annotation_guide.md          # How to contribute new signs
│   ├── gesture_schema.md            # Full schema documentation
│   └── architecture.md             # Technical architecture
│
└── tests/
    └── test_classifier.py           # Unit tests
```

---

## 🌍 Impact

| Dimension | Detail |
|---|---|
| **Who it serves** | ~1–1.5 million Deaf and hard-of-hearing Filipinos |
| **Primary scenarios** | Medical consultations, emergency communication, public service counters |
| **Immediate value** | A Deaf Filipino can communicate a need in under 10 seconds with no interpreter |
| **Open-source value** | Any researcher or NGO can extend the schema and pipeline without rebuilding from scratch |
| **Language contribution** | Advances Taglish speech technology — code-switched Filipino-English output for accessibility |
| **Long-term potential** | Foundation for a community-maintained, growing FSL-to-speech resource for the Philippines |

---

## 📄 License

SenyaSalin source code is released under the **MIT License** — free to use, fork, and modify.

Gesture schema and annotation documentation are released under **CC BY 4.0** — free for research, commercial, and community use with attribution.

---

## 🤝 Acknowledgments

- **MediaPipe** (Google) — hand landmark tracking technology
- **Filipino Deaf community** — the primary motivation and intended beneficiary of this work
- **NFAD** (National Federation of the Deaf Philippines) — target future collaboration partner
- **FEU ACM TechSprint** organizers — for making this hackathon possible

---

<div align="center">

**Team EXPLICIT** | FEU ACM TechSprint

**SenyaSalin** — *Let the voices of light in.*

🤟 Built with care for the Filipino Deaf community. 🤟

</div>
