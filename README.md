# 🌊 Marine AI - Embedded Intelligent Microscopy System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg?logo=yolo&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.0%2B-orange.svg?logo=gradio&logoColor=white)](https://gradio.app/)
[![Platform](https://img.shields.io/badge/Target-Raspberry%20Pi%205%20%7C%20Linux%20%7C%20Windows-green.svg)](docs/PI5_DEPLOYMENT.md)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-informational.svg)](https://git-lfs.github.com/)

**An autonomous edge-AI microscopy platform for real-time automated identification, classification, and enumeration of marine zooplankton.**

*Developed by **Team CodeFather** for Smart India Hackathon (SIH).*

[Overview](#-overview) • [Key Features](#-key-features) • [Architecture](#-system-architecture) • [Getting Started](#-getting-started) • [Pi 5 Deployment](#-raspberry-pi-5-deployment--optimization) • [Model Optimization](#-model-compilation--drift-verification) • [Project Structure](#-project-structure) • [Team](#-team-codefather)

</div>

---

## 🌐 Overview

Marine biodiversity monitoring and water-quality surveillance traditionally rely on manual microscopic examination of planktonic organisms. This manual methodology is **time-consuming** (15–30 minutes per sample), **subjective**, and **difficult to deploy at scale** in remote oceanographic field surveys.

**Marine AI** transforms this workflow into an **autonomous, offline-first edge AI system** capable of running on a **Raspberry Pi 5** connected to a digital USB microscope:
- ⚡ **Rapid Analysis**: Drops sample classification time from 20 minutes to under **2 seconds**.
- 🔬 **13 Taxonomic Classes**: High-precision multi-object detection and counting.
- 🎯 **Edge-Optimized**: Native inference via **NCNN**, **ONNX**, and **INT8 TFLite** with hardware acceleration.
- 🛡️ **Offline Reliability**: Zero internet connection needed; robust against sensor disconnects and hardware faults.

---

## ✨ Key Features

- **Real-Time Automated Enumeration**: Concurrently detects, boxes, and tallies overlapping micro-organisms across multi-scale dimensions (2 µm to 200 µm).
- **USB Microscope Integration (`camera.py`)**: Asynchronous, background-threaded capture pipeline with double-buffering. Web UI requests never block camera I/O, and driver queues stay bounded.
- **Dynamic Compiled Backend Switching**: Automatic priority fallback (`NCNN` → `TFLite` → `ONNX` → `PyTorch`) with subprocess isolation so native crashes never bring down the web service.
- **Interactive Web Interface (`app.py`)**: Clean Gradio interface with dark scientific theme, taxonomic diversity statistics, color-coded species legend, and live-streamed video analysis.
- **Field Diagnostics (`pi_healthcheck.py`)**: Built-in benchmark and system health audit for SoC throttling, temperature, memory usage, and thread-pool efficiency.
- **Model Drift Verification (`optimize_model.py`)**: Mathematical regression verification comparing compiled model bounding boxes and confidence scores against PyTorch reference runs.

---

## 🧬 Target Taxonomic Classes

The model classifies 13 marine taxa with designated visualization palettes:

| Class | Identification Palette | Typical Biology |
|---|---|---|
| **Chaetognath** | 🔴 `#FF0000` | Arrow worms; predatory marine invertebrates |
| **Larval Fish** | 🟢 `#00FF00` | Early developmental stages of teleost fish |
| **Hydromedusa** | 🔵 `#0000FF` | Hydrozoan jellyfish stages |
| **Lobate Ctenophore** | 🟡 `#FFFF00` | Comb jellies with oral lobes |
| **Pleurobrachia** | 🟣 `#FF00FF` | Sea gooseberries; tentaculate ctenophores |
| **Shrimp** | 🔷 `#00FFFF` | Decapod crustacean larvae and juveniles |
| **Siphonophore** | 🟪 `#800080` | Colonial hydrozoans |
| **Stomatopod Larva** | 🩵 `#008080` | Mantis shrimp larval stages |
| **Thaliac** | 🟠 `#FFA500` | Pelagic tunicates (salps, doliolids) |
| **Polychaete Worm** | 🟦 `#4B0082` | Bristle worms |
| **Cumacean** | 🌸 `#EE82EE` | Hooded shrimp; small benthic crustaceans |
| **Ctenophore** | 🌲 `#006400` | Non-lobate comb jellies |
| **Unknown** | ⚪ `#808080` | Unclassified/unresolved organisms |

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────┐
│            USB Digital Microscope / Camera             │
│            (UVC / V4L2 1080p MJPEG Video Stream)       │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│             Camera Pipeline (camera.py)                │
│    • Double-buffered capture thread (1-frame depth)    │
│    • Aspect-ratio preserving downscaler (INTER_AREA)   │
│    • Automatic device reconnection & fault recovery    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│               Inference & Backend Selection            │
│    • Candidate probe & subprocess fault isolation      │
│    • NCNN NEON (fp16) | ONNX Runtime | INT8 TFLite     │
│    • Thread pool allocation (capped to physical cores) │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                Gradio Application (app.py)             │
│    • Image Upload & Live Microscope Feed               │
│    • Video Streaming Annotation with Frame Striding    │
│    • Taxonomic Classification Table & Diversity Stats  │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites
- **Python**: Version `3.10`, `3.11`, or `3.13`
- **Git & Git LFS**: Ensure Git Large File Storage is installed (`git lfs install`)

### 1. Clone the Repository
```bash
git clone https://github.com/KARTHICK-3056/Marine-AI.git
cd Marine-AI
```

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch Application
```bash
python app.py
```
Open your browser at **`http://127.0.0.1:7860`**.

---

## 🥧 Raspberry Pi 5 Deployment & Optimization

The Raspberry Pi 5 (4× Cortex-A76 cores, 8GB RAM) is optimized via tailored environment configurations and compiled runtimes.

> 📖 **Full Deployment Guide**: Refer to **[docs/PI5_DEPLOYMENT.md](docs/PI5_DEPLOYMENT.md)** for OS tuning, headless systemd service setup, and offline bundle builds.

### Quick Pi 5 Setup
```bash
# 1. Setup OS tuning (CPU governor, ZRAM, USB autosuspend)
sudo ./scripts/setup_pi_os.sh

# 2. Run with thread caps and jemalloc optimization
./run_optimized.sh

# 3. Check health and benchmark inference
python scripts/pi_healthcheck.py --bench
```

### Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `CPU_THREADS` | `min(4, cpu_count)` | Constrains OpenMP, PyTorch, OpenCV, OpenBLAS pools to physical cores. |
| `MODEL_BACKEND` | `auto` | Selects fastest available backend (`ncnn` → `tflite` → `onnx` → `pytorch`). |
| `MODEL_PATH` | — | Explicit override path to a custom `.pt`, `.onnx`, or `_ncnn_model` artifact. |
| `YOLO_IMGSZ` | `640` | Inference resolution (multiple of 32; `416` recommended for Pi 5). |
| `CONFIDENCE_THRESHOLD` | `0.3` | Minimum detection confidence cutoff. |
| `ENABLE_CAMERA` | `true` | Enables USB microscope live capture tab. |
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Set to `0.0.0.0` for headless LAN/Hotspot access. |

---

## 🔬 Model Compilation & Drift Verification

Compile weights to ARM-optimized runtimes and verify that quantization drift remains within strict statistical tolerances:

```bash
# 1. Install compilation tools (on development machine)
pip install -r requirements.txt -r requirements-export.txt

# 2. Compile models to NCNN, ONNX, and INT8 TFLite
python optimize_model.py export --formats ncnn onnx tflite --imgsz 416

# 3. Verify accuracy against PyTorch reference
python optimize_model.py verify --iou-tol 0.85 --conf-tol 0.1
```

---

## 📁 Project Structure

```
Marine-AI/
├── app.py                      # Primary Gradio web application & inference router
├── camera.py                   # Thread-safe USB microscope capture pipeline
├── optimize_model.py           # NCNN/ONNX/TFLite export & accuracy drift verifier
│
├── pyproject.toml              # Build system, tool configurations & dependencies
├── requirements.txt            # Core desktop / development dependencies
├── requirements-pi.txt         # Raspberry Pi 5 runtime dependencies
├── requirements-export.txt     # Model export toolchain
│
├── run_optimized.sh            # Production shell launcher for Raspberry Pi
├── create_offline_bundle.sh    # Offline wheelhouse builder for field deployment
├── install_offline.sh          # Offline installer for field devices
├── marine_ai.service           # systemd unit for headless auto-boot
│
├── models/
│   ├── best.pt                 # Trained production YOLOv8 model weights (Git LFS)
│   ├── exports.json            # Manifest of compiled artifacts & frozen resolutions
│   └── best_ncnn_model/        # Compiled NCNN model directory
│
├── scripts/
│   ├── pi_healthcheck.py       # Hardware diagnostic & benchmark utility
│   └── setup_pi_os.sh          # OS-level tuning (ZRAM, CPU governor, USB power)
│
├── docs/
│   ├── PI5_DEPLOYMENT.md       # Comprehensive Raspberry Pi 5 deployment guide
│   ├── Classification_table_YOLOv8.pdf # Taxonomic classification reference
│   └── audits/                 # Architectural assessments & historical audit reports
│
├── test_img/                   # Validation imagery & drift-comparator sample corpus
└── tests/
    ├── test_app.py             # Unit tests for app, camera, and backend resolution
    └── test_optimize_model.py  # Unit tests for drift comparator & export parser
```

---

## 🧪 Testing & Code Quality

Marine AI includes a full unit test suite and strict static analysis standards:

```bash
# Run unit test suite
python -m unittest discover -s tests -v

# Run static analysis and style checks
ruff check .
ruff format --check .
```

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
