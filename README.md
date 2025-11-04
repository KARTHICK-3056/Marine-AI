# 🌊 Marine AI - Embedded Intelligent Microscopy System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An AI-powered embedded microscopy system for automated detection, classification, and counting of marine microorganisms.**

Developed by **Team CodeFather** for Smart India Hackathon 2024

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Information](#model-information)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contact](#contact)

---

## 🌐 Overview

Marine biodiversity assessments traditionally rely on manual microscopic examination of planktonic organisms—a process that is **time-consuming** (10-20 minutes per sample), **labor-intensive**, **subjective**, and **not scalable**. 

Our solution is an **offline-first, low-cost embedded AI system** built on Raspberry Pi 5 that automates:
- ✅ **Detection** of marine microorganisms using YOLOv8
- ✅ **Classification** with high accuracy deep learning models
- ✅ **Counting** and statistical analysis
- ✅ **Real-time inference** on embedded hardware
- ✅ **User-friendly web interface** for easy operation

### Why This Matters
- ⏱️ **Reduces analysis time** from 20 minutes to 20 seconds
- 💰 **Cuts costs** by 10x compared to traditional methods
- 🚢 **Field-deployable** for ships and remote coastal labs
- 🇮🇳 **Indigenous technology** supporting Make in India and Digital India
- 🌍 **UN SDG alignment** (SDG 6: Clean Water, SDG 14: Life Below Water)

---

## ✨ Features

### Core Capabilities
- **Real-time Detection**: YOLOv8 detects multiple overlapping organisms
- **High Accuracy Classification**: Trained on marine zooplankton dataset
- **Automated Counting**: Enumerates organisms by species
- **Offline Operation**: Fully functional without internet connectivity
- **User-Friendly Dashboard**: Gradio-based web interface with visual results
- **Detailed Analytics**: Species diversity, abundance, confidence scores

### Technical Highlights
- 🖥️ **Embedded AI**: Designed for Raspberry Pi 5 (8GB)
- ⚡ **Optimized Inference**: Production-ready YOLO models
- 🎯 **Multi-Scale Detection**: Handles organisms from 2µm to 200µm
- 📊 **Visual Reports**: Color-coded results with statistics
- 🎨 **Professional UI**: Modern gradient-based interface

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────┐
│                  USB Digital Microscope                │
│                    (1080p/4K Imaging)                  │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│              Raspberry Pi 5 (8GB RAM)                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Image Preprocessing Pipeline             │  │
│  │  • Quality Assessment  • Normalization           │  │
│  └───────────────────┬──────────────────────────────┘  │
│                      │                                 │
│  ┌───────────────────▼──────────────────────────────┐  │
│  │     YOLOv8 Detection & Classification            │  │
│  │  • Multi-scale Detection  • Real-time Inference  │  │
│  └───────────────────┬──────────────────────────────┘  │
│                      │                                 │
│  ┌───────────────────▼──────────────────────────────┐  │
│  │    Post-Processing & Analytics Engine            │  │
│  │  • Counting  • Statistics  • Visualization       │  │
│  └───────────────────┬──────────────────────────────┘  │
└────────────────────┬─┴─────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌──────────────┐         ┌──────────────┐
│   Gradio     │         │   Export     │
│   Dashboard  │         │   Results    │
│   127.0.0.1  │         │   (Images)   │
└──────────────┘         └──────────────┘
```

---

## 🔧 Hardware Requirements

### Development Setup (Current)
- **Computer**: Windows/Linux/Mac with Python 3.9+
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **GPU**: Optional (CUDA-compatible for faster inference)

### Production Configuration (Raspberry Pi)
| Component | Specification | Cost (INR) |
|-----------|--------------|-----------|
| **Raspberry Pi 5** | 8GB RAM | ₹8,000 |
| **USB Microscope** | 1080p/4K Digital | ₹2,000 |
| **Power Supply** | 27W USB-C PSU | ₹800 |
| **Storage** | 64GB microSD Card | ₹600 |
| **Cooling** | Active Fan/Heatsink | ₹400 |
| **Case** | Protective Enclosure | ₹200 |
| **Total** | - | **₹12,000** |

---

## 📦 Installation

### Prerequisites
```bash
# Check Python version (3.9 or higher required)
python --version

# Check pip
pip --version
```

### Step 1: Clone or Download Project
```bash
# If using Git
git clone https://github.com/YourUsername/Marine-AI.git
cd Marine-AI

# Or download and extract to D:\Marine-AI
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Navigate to project directory
cd D:\Marine-AI

# Create virtual environment
python -m venv Marine-AI

# Activate it
# Windows Command Prompt:
Marine-AI\Scripts\activate

# Windows PowerShell:
Marine-AI\Scripts\Activate.ps1

# Linux/Mac:
source Marine-AI/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: Installation may take 5-10 minutes depending on your internet speed. PyTorch is the largest package (~2GB).

### Step 4: Verify Installation
```bash
# Check if all packages are installed
pip list

# Should see: ultralytics, gradio, torch, opencv-python, etc.
```

### Step 5: Place Model File
Ensure your trained model is in the correct location:
```
D:\Marine-AI\models\best.pt
```

---

## 🚀 Usage

### Quick Start

#### Method 1: Using Virtual Environment (Recommended)
```bash
# Open Command Prompt/PowerShell
cd D:\Marine-AI

# Activate virtual environment
Marine-AI\Scripts\activate

# Run application
python app.py
```

#### Method 2: Direct Python Execution
```bash
# From any directory
python D:\Marine-AI\app.py

# Or using Python 3 explicitly
python3 D:\Marine-AI\app.py
```

#### Method 3: From Activated Virtual Environment
```bash
# If already in (Marine-AI) environment
cd D:\Marine-AI
python app.py
```

### Expected Output
```
Loading YOLOv8 model from: D:\Marine-AI\models\best.pt
Model loaded successfully!
Found free port: 7860
Starting Clean Gradio App...
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live

To create a public link, set `share=True` in `launch()`.
```

### Accessing the Application

1. **Local Access** (Same Computer):
   - Open browser and go to: `http://127.0.0.1:7860`
   
2. **Network Access** (Other Devices on Same Network):
   - Use the public Gradio link shown in terminal
   - Valid for 72 hours

### Using the Interface

**Step-by-Step Workflow:**

1. **Upload Image**
   - Click "Select Marine Sample Image"
   - Choose an image from `D:\Marine-AI\test_img\` folder
   - Supported formats: JPG, PNG, JPEG

2. **Analyze Sample**
   - Click the "Analyze Microorganism sample" button
   - Wait 2-5 seconds for processing

3. **View Results**
   - **Before Detection**: Original image
   - **After Detection**: Annotated image with colored bounding boxes
   - **Analysis Summary**: 
     - Total detections count
     - Species diversity
     - Most common species
     - Detailed breakdown by species with counts and confidence scores

4. **Interpret Colors**
   - Each species has a unique color
   - Color legend shown in the detailed breakdown
   - Bounding boxes match the species color

5. **Download Results** (Optional)
   - Right-click on annotated image → "Save Image As"
   - Copy analysis summary for reports

---

## 🧠 Model Information

### Supported Marine Species (13 Classes)

| Species | Color Code | Description |
|---------|-----------|-------------|
| **Chaetognath** | 🔴 Red | Arrow worms, predatory zooplankton |
| **Larval Fish** | 🟢 Green | Early life stage fish |
| **Hydromedusa** | 🔵 Blue | Jellyfish-like organisms |
| **Lobate Ctenophore** | 🟡 Yellow | Comb jellies with lobes |
| **Pleurobrachia** | 🟣 Magenta | Sea gooseberry ctenophore |
| **Shrimp** | 🔷 Cyan | Decapod crustaceans |
| **Siphonophore** | 🟣 Purple | Colonial organisms |
| **Stomatopod Larva** | 🌊 Teal | Mantis shrimp larvae |
| **Thaliac** | 🟠 Orange | Salps and related organisms |
| **Polychaete Worm** | 💜 Indigo | Segmented marine worms |
| **Cumacean** | 🌸 Violet | Small crustaceans |
| **Ctenophore** | 🌳 Dark Green | Comb jellies |
| **Unknown** | ⚪ Gray | Unclassified organisms |

### Model Specifications

**YOLOv8 Detection Model**
- **Architecture**: YOLOv8 (Ultralytics)
- **Model File**: `D:\Marine-AI\models\best.pt`
- **Input Size**: Variable (auto-resized to 640x640)
- **Confidence Threshold**: 0.3 (30%)
- **Output**: Bounding boxes with class labels and confidence scores

**Performance**
- **Detection Speed**: ~2-5 seconds per image (CPU)
- **Accuracy**: High precision on trained dataset
- **Batch Processing**: Supported

---

## 📁 Project Structure

```
D:\Marine-AI\
│
├── app.py                      # Main Gradio application (Entry point)
├── requirements.txt            # Python dependencies list
├── README.md                   # This documentation file
│
├── models\
│   └── best.pt                 # Trained YOLOv8 model weights
│
├── test_img\                   # Test images directory
│   ├── sample1.jpg             # Example microscope images
│   ├── sample2.png
│   └── ...
│
└── Marine-AI\                  # Virtual environment (if created)
    ├── Scripts\
    ├── Lib\
    └── ...
```

### File Descriptions

- **`app.py`**: Main application with Gradio UI and YOLO detection logic
- **`requirements.txt`**: All Python package dependencies
- **`models\best.pt`**: Pre-trained YOLOv8 model (must be present)
- **`test_img\`**: Sample images for testing the system
- **`Marine-AI\`**: Virtual environment folder (optional but recommended)

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Model Not Found Error
```
ERROR: Model file not found at D:\Marine-AI\models\best.pt
```
**Solution:**
- Verify the model file exists at exactly `D:\Marine-AI\models\best.pt`
- Check file name is `best.pt` (case-sensitive on Linux)
- Ensure you have the trained model file

#### 2. Module Not Found Error
```
ModuleNotFoundError: No module named 'ultralytics'
```
**Solution:**
```bash
# Reinstall requirements
pip install -r requirements.txt

# Or install individually
pip install ultralytics gradio opencv-python torch
```

#### 3. Port Already in Use
```
OSError: [Errno 48] Address already in use
```
**Solution:**
- Close other applications using port 7860
- Or the app will automatically find a free port

#### 4. CUDA/GPU Errors (Optional GPU)
```
RuntimeError: CUDA out of memory
```
**Solution:**
- This is normal on CPU-only systems
- YOLOv8 will automatically use CPU
- Performance is still good for single images

#### 5. Gradio Public Link Not Working
**Solution:**
- Use local URL: `http://127.0.0.1:7860` instead
- Check firewall settings
- Restart the application

#### 6. Slow Inference Speed
**Solution:**
- Use smaller images (resize to 1920x1080 or smaller)
- Close unnecessary applications
- Consider using GPU if available
- On Raspberry Pi: ensure active cooling

### Getting Help

If you encounter issues:
1. Check the error message carefully
2. Verify all files are in correct locations
3. Ensure virtual environment is activated
4. Try reinstalling dependencies
5. Open an issue on GitHub with error logs

---


## 📊 Performance Metrics

### Current System (Development PC)
- **Detection Time**: 2-5 seconds per image
- **Supported Image Sizes**: Up to 4K resolution
- **Concurrent Users**: Single user (Gradio limitation)
- **Accuracy**: High (based on training dataset)

### Target System (Raspberry Pi 5)
- **Detection Time**: ~10-15 seconds per image (with optimization)
- **Power Consumption**: 10-12W
- **Offline Operation**: 100% functional
- **Storage**: SD card or external SSD

---

## 🌍 Impact & Applications

### Marine Research
- 🔬 Automated biodiversity monitoring
- 📊 Long-term ecological studies
- 🌡️ Climate change impact assessment
- 🗺️ Spatial distribution mapping

### Aquaculture & Fisheries
- 💧 Water quality monitoring
- 🦐 Plankton abundance tracking
- 🐟 Feed optimization
- ⚠️ Early warning for blooms

### Environmental Protection
- 🌊 Harmful algal bloom detection
- 🏭 Pollution indicator monitoring
- 🌴 Coastal ecosystem health
- 🛡️ Marine conservation efforts

### Education & Training
- 🎓 Affordable AI microscopy for universities
- 👨‍🎓 Hands-on embedded systems training
- 📚 Open-source research platform
- 🔬 STEM education tool

---

## 🤝 Contributing

We welcome contributions! Areas where you can help:

- 🐛 **Bug Reports**: Found an issue? Let us know
- 💡 **Feature Suggestions**: Ideas for improvements
- 📝 **Documentation**: Help improve guides
- 🔬 **Dataset Sharing**: Contribute marine organism images
- 💻 **Code**: Submit pull requests

---

## 👥 Team CodeFather

**Smart India Hackathon 2024**

Developed with 💙 for Marine Conservation

---

## 👥 Authors

**VIJAYA KARTHICK RAJA U M**  
📧 vkr3056@gmail.com  
🔗 [github.com/KARTHICK-3056](https://github.com/KARTHICK-3056)

**S.S.MADHAVAN**  
📧 ssmadhavan006@gmail.com  
🔗 [github.com/ssmadhavan006](https://github.com/ssmadhavan006)

**DIVYESH HARI G**  
📧 divyesh02208@gmail.com  
🔗 [github.com/DIVYESH-HARI](https://github.com/DIVYESH-HARI)

**G.K.AKASHGAUTHAM**  
📧 gkakash2006@gmail.com  
🔗 [github.com/Akashgautham](https://github.com/Akashgautham)

**K.RAKSHITHASRI**  
📧 rakshiekt@gmail.com  
🔗 [github.com/rakshithasri-k](https://github.com/rakshithasri06)

**M.N.RAKSHA**  
📧 rakshanathan006@gmail.com  
🔗 [github.com/raksha006](https://github.com/raksha006)

---

## 🙏 Acknowledgments

- **Ultralytics**: For the YOLOv8 framework
- **Gradio Team**: For the amazing web interface library
- **Marine Biology Community**: Dataset and validation support
- **Smart India Hackathon**: Platform and opportunity
- **Open Source Community**: For various tools and libraries

---

<div align="center">

### 🌊 **Protecting Marine Biodiversity Through AI Innovation** 🌊

**Made in India 🇮🇳 | For the Ocean 🌊 | Open Source 💻**

---

**[⭐ Star this project](https://github.com/YourUsername/Marine-AI)** | **[📖 Documentation](README.md)** | **[🐛 Report Bug](https://github.com/YourUsername/Marine-AI/issues)**

</div>
