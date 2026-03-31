# Computer-Vision

# 🤟 ASL Sign Language → Sentence Builder

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![MobileNetV2](https://img.shields.io/badge/MobileNetV2-Transfer%20Learning-34A853?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-Training-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Real-time American Sign Language (ASL) alphabet recognition via webcam — spell out words and sentences letter by letter using hand gestures.**

[Features](#-features) · [Demo](#-demo) · [Installation](#-installation) · [Training](#-training-the-model) · [Usage](#-usage) · [How It Works](#-how-it-works) · [Project Structure](#-project-structure)

</div>

---

## 📸 Demo

> Show your hand inside the green ROI box. The system predicts the ASL letter, waits 1 second for confirmation, and builds a sentence in real time.

```
┌─────────────────────────────────────┐
│         ASL Sign → Sentence         │
│  FPS: 28.4                          │
│                                     │
│       ┌─────────────────┐           │
│  H    │                 │           │
│(91.2%)│   [hand ROI]    │           │
│       │_________________│ ← hold bar│
│                                     │
│  Word:  HELLO                       │
│  Sent:  HELLO WORLD_                │
│  q:quit  c:clear  b:backspace       │
└─────────────────────────────────────┘
```

**Supported signs:** `A–Z` + `space` + `del` + `nothing` (pause) = **29 classes**

---

## ✨ Features

| Feature | Details |
|---|---|
| ✋ **Live Webcam** | Real-time recognition at ~30 FPS from any webcam |
| 🧠 **MobileNetV2** | Lightweight transfer learning model — runs on CPU |
| 🔤 **Sentence Builder** | Spell words letter by letter; `space` commits a word |
| ⏳ **Hold-to-Confirm** | 1-second hold timer prevents accidental keystrokes |
| 🗳️ **Voting Buffer** | Rolling window of 8 frames with majority vote for stability |
| 💡 **CLAHE Preprocessing** | Adaptive histogram equalization handles varied lighting |
| ⌨️ **Keyboard Controls** | `q` quit · `c` clear · `b` backspace |
| 📊 **2-Phase Training** | Frozen base → fine-tune last 30 layers for best accuracy |

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/asl-sentence-builder.git
cd asl-sentence-builder
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📋 <strong>requirements.txt</strong> (click to expand)</summary>

```txt
tensorflow>=2.12.0
opencv-python>=4.8.0
numpy>=1.24.0
```

</details>

### 4. Place the model weights

Put your trained model file in the project root:

```
asl-sentence-builder/
└── asl.keras    ← trained model goes here
```

> Don't have a model yet? See [Training the Model](#-training-the-model) below.

---

## 🚀 Usage

### Run the real-time recognizer

```bash
python asl.py
```

A webcam window will open. Place your hand inside the **green box** at the center-top of the frame.

### Hand gesture controls

| Gesture / Key | Action |
|---|---|
| Hold a letter sign for **1 second** | Appends that letter to the current word |
| `space` sign | Commits current word to sentence, starts new word |
| `del` sign | Deletes last character |
| `nothing` sign | Pauses input (use to reset between letters) |
| `b` key | Backspace |
| `c` key | Clears everything |
| `q` key | Quit |

### Tune parameters (top of `asl.py`)

```python
CONF_THRESHOLD = 0.85   # raise to reduce false triggers
PREDICT_EVERY  = 3      # predict every N frames (lower = more CPU)
BUFFER_SIZE    = 8      # rolling vote window size
MIN_VOTE_COUNT = 5      # minimum votes needed to confirm a label
HOLD_TIME      = 1.0    # seconds to hold a sign before it's accepted
```

---

## 🧠 Training the Model

Training is done in Google Colab using `Sign_Language_ASL.ipynb`.

### Dataset

- **Source:** [`grassknoted/asl-alphabet`](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) on Kaggle
- **Size:** 87,000 images · 200×200 px · 29 classes
- **Subset used:** 20% per class (~17,400 images) for faster iteration

### Training pipeline

```
Dataset (87k images)
    │
    ▼  20% random sample per class
Subset (~17,400 images)
    │
    ▼  ImageDataGenerator (80/20 train/val split)
    │   augmentation: rotation ±15°, zoom 15%,
    │   shift 10%, brightness 0.8–1.2
    │   horizontal_flip = False  (ASL is hand-specific!)
    │
    ▼
┌──────────────────────────────────────────────────┐
│  PHASE 1 — Frozen base (4 epochs, lr=1e-3)       │
│  MobileNetV2 (ImageNet) → GAP → BN →             │
│  Dense(512, relu) → Dropout(0.5) →               │
│  Dense(256, relu) → Dropout(0.3) →               │
│  Dense(29, softmax)                               │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  PHASE 2 — Fine-tune last 30 layers (2 epochs,   │
│  lr=1e-5)  — earlier layers stay frozen          │
└──────────────────────────────────────────────────┘
    │
    ▼
  asl.keras  (saved to Google Drive)
```

### Run training in Colab

1. Open `Sign_Language_ASL.ipynb` in Google Colab
2. Enable **GPU runtime**: Runtime → Change runtime type → T4 GPU
3. Set up Kaggle credentials (needed for `kagglehub`):
   ```python
   import os
   os.environ["KAGGLE_USERNAME"] = "your_username"
   os.environ["KAGGLE_KEY"]      = "your_api_key"
   ```
4. Run all cells — the best model saves automatically to Google Drive as `best_model_finetuned.keras`
5. Download it, rename to `asl.keras`, and place it in the project root

### Callbacks used during training

| Callback | Config |
|---|---|
| `EarlyStopping` | monitors `val_accuracy`, patience=3 |
| `ReduceLROnPlateau` | factor=0.5, patience=2, min_lr=1e-7 |
| `ModelCheckpoint` | saves best `val_accuracy` checkpoint |

---

## ⚙️ How It Works

```
Webcam Frame (640×480)
    │
    ▼  flip horizontally (mirror mode)
    │
    ▼  crop square ROI (50% of frame, upper center)
    │
    ▼  PREPROCESS
    │   BGR → RGB
    │   RGB → LAB → CLAHE on L channel → LAB → RGB
    │   resize to 224×224
    │   normalize 0–1
    │
    ▼  MobileNetV2 model (every 3rd frame)
    │   → softmax probabilities (29 classes)
    │
    ▼  VOTING BUFFER (deque, maxlen=8)
    │   majority vote over last 8 predictions
    │   accept only if count ≥ 5 AND conf > 0.85
    │
    ▼  HOLD TIMER
    │   same label held ≥ 1.0 s → append to word/sentence
    │
    ▼  SENTENCE STATE
        current_word + sentence → displayed on screen
```

### Why CLAHE?
Training images (Kaggle dataset) are photographed under controlled studio lighting. Webcam feeds vary widely. CLAHE (Contrast Limited Adaptive Histogram Equalization) normalizes local contrast so the model sees a consistent input regardless of your room lighting.

### Why no horizontal flip?
ASL signs are **hand-specific** — the same letter signed with the left vs right hand looks mirrored and would confuse the classifier. Augmentation intentionally skips `horizontal_flip`.

---

## 📁 Project Structure

```
asl-sentence-builder/
├── asl.py                     # real-time webcam inference + sentence builder
├── Sign_Language_ASL.ipynb    # Colab training notebook
├── asl.keras                  # trained model weights (you provide)
├── requirements.txt
└── README.md
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `Error loading model` | Ensure `asl.keras` is in the project root and is a valid Keras file |
| Webcam not opening | Try changing `cv2.VideoCapture(0)` → `cv2.VideoCapture(1)` |
| Very low confidence | Improve room lighting; keep hand clearly inside the green ROI box |
| Letters appending too fast | Increase `HOLD_TIME` (e.g. `1.5`) |
| Letters not appending | Lower `CONF_THRESHOLD` slightly (e.g. `0.75`) or slow down gestures |
| High CPU usage | Increase `PREDICT_EVERY` (e.g. `5`) to predict less frequently |
| Colab training crashes | Reduce `BATCH_SIZE` to `32` or lower `fraction` to `0.1` |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/add-word-suggestions`
3. Commit your changes: `git commit -m 'feat: add word auto-complete'`
4. Push to branch: `git push origin feature/add-word-suggestions`
5. Open a Pull Request

**Ideas for contributions:**
- Add text-to-speech for the completed sentence
- Support for ASL words / phrases (not just alphabet)
- Dynamic hand segmentation instead of a fixed ROI box
- Web interface using WebRTC + TensorFlow.js

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

Made with 🤟 to bridge communication gaps · Star ⭐ the repo if it helped you!

</div>
