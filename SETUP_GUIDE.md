# 🚀 Air Math Solver - Complete Setup Guide

This guide will walk you through setting up the Air Math Solver from scratch.

## 📋 Prerequisites Checklist

Before starting, ensure you have:
- [ ] Computer with Windows, macOS, or Linux
- [ ] Python 3.8 or higher installed
- [ ] Working webcam
- [ ] At least 3GB free disk space
- [ ] Internet connection (for downloads)

---

## Step 1: Verify Python Installation

### Check Python Version

Open your terminal/command prompt and run:

```bash
python --version
```
or
```bash
python3 --version
```

You should see Python 3.8 or higher. If not, download from [python.org](https://www.python.org/downloads/).

**Important**: During installation, check "Add Python to PATH"!

---

## Step 2: Set Up Project Directory

### Option A: Download the Project Files

1. Download the entire `air_math_solver` folder
2. Place it somewhere easy to access (e.g., Desktop or Documents)

### Option B: Navigate to Existing Project

```bash
cd /path/to/air_math_solver
```

Example:
```bash
# Windows
cd C:\Users\YourName\Desktop\air_math_solver

# macOS/Linux
cd ~/Desktop/air_math_solver
```

---

## Step 3: Create Virtual Environment (HIGHLY RECOMMENDED)

A virtual environment keeps project dependencies isolated.

### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your command line.

### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your command line.

**Troubleshooting**:
- If `venv\Scripts\activate` fails on Windows, try: `venv\Scripts\activate.bat`
- If permission denied on macOS/Linux, use: `chmod +x venv/bin/activate`

---

## Step 4: Install Python Dependencies

With your virtual environment activated:

```bash
pip install -r requirements.txt
```

This will install:
- opencv-python (Computer Vision)
- mediapipe (Hand Tracking)
- tensorflow (Deep Learning)
- customtkinter (GUI)
- numpy, scikit-learn, matplotlib, Pillow

**Installation Time**: 5-15 minutes depending on internet speed

**Common Errors & Fixes**:

### Error: "pip: command not found"
```bash
python -m pip install -r requirements.txt
```

### Error: "Could not find a version that satisfies..."
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Error: TensorFlow installation fails
For Apple M1/M2 Macs:
```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

For Windows with GPU:
```bash
pip install tensorflow-gpu
```

---

## Step 5: Download the Dataset

This is the most important step for training the model.

### Method 1: Manual Download (RECOMMENDED)

1. **Go to Kaggle**:
   - Visit: https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols
   
2. **Download Dataset**:
   - Click the "Download" button (you may need to create a free Kaggle account)
   - You'll get a ZIP file (~100MB)

3. **Extract Dataset**:
   - Extract the ZIP file
   - You should see folders named: 0, 1, 2, ..., 9, +, -, times, dot

4. **Place in Correct Location**:
   - Create folder: `air_math_solver/data/handwritten_math_symbols/`
   - Move all the extracted folders into this directory

**Final Structure**:
```
air_math_solver/
├── data/
│   └── handwritten_math_symbols/
│       ├── 0/
│       │   ├── img_0001.png
│       │   ├── img_0002.png
│       │   └── ...
│       ├── 1/
│       ├── 2/
│       ├── .../
│       ├── 9/
│       ├── +/
│       ├── -/
│       ├── times/
│       └── dot/
├── main_app.py
├── train_model.py
└── ...
```

### Method 2: Using Kaggle API (Advanced)

```bash
# Install Kaggle
pip install kaggle

# Set up API credentials
# 1. Go to https://www.kaggle.com/account
# 2. Scroll to "API" section
# 3. Click "Create New API Token"
# 4. Place kaggle.json in:
#    - Windows: C:\Users\YourName\.kaggle\kaggle.json
#    - macOS/Linux: ~/.kaggle/kaggle.json

# Download dataset
mkdir -p data
kaggle datasets download -d xainano/handwrittenmathsymbols
unzip handwrittenmathsymbols.zip -d data/handwritten_math_symbols/
```

---

## Step 6: Train the Model

Now we'll train the CNN to recognize handwritten symbols.

### Run Training Script

```bash
python train_model.py
```

### What to Expect:

**Console Output**:
```
==============================================================
DATASET SETUP INSTRUCTIONS
==============================================================
...
✓ Dataset found! Starting training...

Loading dataset...
Loaded 10000+ images across 14 classes

Class names: ['0' '1' '2' '3' '4' '5' '6' '7' '8' '9' '+' '-' 'x' '.']
Dataset shape: (10000, 45, 45, 1)

Training samples: 8000
Testing samples: 2000

Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
...

Training model...
Epoch 1/30
250/250 [==============================] - 45s 180ms/step - loss: 0.8234 - accuracy: 0.7345 - val_loss: 0.3456 - val_accuracy: 0.8934
...
Epoch 15/30
250/250 [==============================] - 42s 168ms/step - loss: 0.1234 - accuracy: 0.9678 - val_loss: 0.0987 - val_accuracy: 0.9756

✓ Training complete!
Test Accuracy: 97.56%
Model saved to models/math_symbol_classifier.h5
```

**Training Duration**: 10-30 minutes (depends on your CPU/GPU)

**Expected Accuracy**: 95-98%

**Files Created**:
- `models/math_symbol_classifier.h5` - Trained model
- `models/class_names.npy` - Class labels
- `models/training_history.png` - Accuracy/loss graphs

**Troubleshooting**:

### Error: "Dataset not found"
- Check that `data/handwritten_math_symbols/` exists
- Verify folders (0, 1, 2, ..., times, dot) are inside
- Check folder names exactly match (case-sensitive)

### Error: "Out of memory"
- Reduce batch size in `train_model.py`:
  ```python
  BATCH_SIZE = 16  # Change from 32 to 16
  ```

### Training too slow?
- First time is always slower (compiling)
- Consider using Google Colab for free GPU access

---

## Step 7: Run the Application

Finally! Time to test your Air Math Solver.

```bash
python main_app.py
```

### What You Should See:

A window opens with:
- **Left side**: Video feed displays (black initially)
- **Right side**: Controls and results panel
- **Title**: "✍️ Air Math Solver"

---

## Step 8: Test the Application

### Test 1: Camera Check

1. Click **"▶️ Start Camera"**
2. Your webcam light should turn on
3. You should see yourself in the left video feed
4. Button changes to **"⏸️ Stop Camera"** (red)

**Problem?** See "Camera Issues" section below.

### Test 2: Hand Detection

1. With camera running, hold your hand in front of the camera
2. You should see:
   - Green skeleton overlay on your hand
   - Dots at key points (21 landmarks)

**Problem?** See "Hand Detection Issues" section below.

### Test 3: Air Writing

1. Bring your thumb and index finger together (pinch gesture)
2. Green circle should appear on index finger
3. Move your hand - you should see white drawing in bottom panel
4. Release fingers - circle turns red, drawing stops

### Test 4: Solve Equation

1. Draw a simple equation: "5 + 3"
2. Release fingers when done
3. Click **"🔍 Solve Equation"**
4. Results panel should show:
   - Detected Equation: `5+3`
   - Answer: `8`
   - Confidence: ~95%+

🎉 **SUCCESS!** Your Air Math Solver is working!

---

## 🐛 Common Issues & Solutions

### Camera Issues

**Camera won't start**:
```python
# Try different camera index
# Edit main_app.py, line ~232:
self.video_capture = cv2.VideoCapture(1)  # Try 1 instead of 0
```

**Multiple cameras?**
- 0 = Built-in webcam
- 1 = External USB camera

**Permission denied** (macOS):
- System Preferences → Security & Privacy → Camera
- Enable for Terminal/Python

### Hand Detection Issues

**Hand not detected**:
- Ensure good lighting
- Keep hand fully in frame
- Try moving closer/farther
- Use palm facing camera

**Shaky/unstable tracking**:
- Keep hand still for a moment
- Reduce background clutter
- Improve lighting

### Recognition Issues

**Symbols not recognized**:
- Write larger (fill more of screen)
- Write slower
- Wait between symbols
- Clear canvas and retry

**Wrong predictions**:
- Model needs more training data
- Retrain with more epochs
- Write more clearly

### GUI Issues

**Window too large/small**:
Edit `main_app.py`, line ~49:
```python
self.root.geometry("1200x700")  # Adjust size
```

**Black screens**:
- Check camera is working
- Restart application
- Check webcam permissions

---

## 🎯 Quick Start Commands

Here's everything in one place:

```bash
# 1. Navigate to project
cd air_math_solver

# 2. Create virtual environment
python -m venv venv

# 3. Activate (Windows)
venv\Scripts\activate

# 3. Activate (macOS/Linux)
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download dataset (manual from Kaggle)
# https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols
# Extract to: data/handwritten_math_symbols/

# 6. Train model
python train_model.py

# 7. Run application
python main_app.py
```

---

## 📱 Alternative: Test Without Training

If you want to test the interface before training:

1. **Test Hand Tracking Only**:
```bash
python utils/hand_tracker.py
```

This opens a simple window showing hand tracking without classification.

2. **Use Pre-uploaded Images**:
- Skip camera mode
- Use "Upload Image" option
- Draw equation in Paint/Preview
- Upload and test segmentation

---

## 💡 Tips for Best Results

1. **Writing Technique**:
   - Write large symbols (fill 1/4 of screen)
   - Leave space between symbols
   - Keep hand steady while pinching

2. **Lighting**:
   - Face a window or light source
   - Avoid backlighting
   - Even lighting on hand

3. **Camera Position**:
   - Camera at eye level
   - 2-3 feet away
   - Plain background behind you

4. **Supported Equations**:
   - ✅ `5+3`, `12-7`, `4x6`
   - ✅ `3.5+2.1` (decimals)
   - ❌ `5÷2` (division not in dataset)
   - ❌ `2^3` (powers not in dataset)

---

## 🆘 Still Having Issues?

### Check This Checklist:

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated (see `(venv)`)
- [ ] All dependencies installed (no errors)
- [ ] Dataset in correct folder structure
- [ ] Model trained successfully (97%+ accuracy)
- [ ] Webcam working in other apps
- [ ] Good lighting conditions

### Test Each Component:

```bash
# Test imports
python -c "import cv2, mediapipe, tensorflow; print('All imports OK')"

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAIL')"

# Test model
python -c "import os; print('Model OK' if os.path.exists('models/math_symbol_classifier.h5') else 'Model MISSING')"
```

---

## 📚 Next Steps

Once working:

1. **Experiment**: Try different equations
2. **Improve**: Retrain with more epochs
3. **Extend**: Add new operators
4. **Document**: Take screenshots for your report
5. **Present**: Prepare demo for class

---

## 🎓 For CMSC 191 Submission

Make sure you have:
- [x] All code files
- [x] Trained model files
- [x] Requirements.txt
- [x] README.md
- [x] Screenshots of working app
- [x] Training history plot
- [x] Demo video (optional but recommended)

---

**You're all set! Happy equation solving! ✍️🔢**
