# ⚡ QUICK START CHEAT SHEET

## 🔥 First Time Setup (5 Commands)

```bash
# 1. Navigate to project folder
cd air_math_solver

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install everything
pip install -r requirements.txt

# 5. Verify setup
python verify_setup.py
```

---

## 📥 Dataset Setup

### Quick Method:
1. Go to: https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols
2. Click "Download" (create free account if needed)
3. Extract ZIP file
4. Copy folders (0, 1, 2, ..., times, dot) to:
   `air_math_solver/data/handwritten_math_symbols/`

### Verify:
```bash
# Should show 14 folders
ls data/handwritten_math_symbols/
# or Windows:
dir data\handwritten_math_symbols\
```

---

## 🤖 Train Model

```bash
python train_model.py
```

**Time**: 10-30 minutes  
**Expected**: 95%+ accuracy  
**Output**: `models/math_symbol_classifier.h5`

---

## 🚀 Run Application

```bash
python main_app.py
```

---

## 🎮 How to Use

1. **Start Camera** → Click green "▶️ Start Camera" button
2. **Pinch Fingers** → Thumb + index together = ✍️ writing mode
3. **Draw Equation** → Write "5 + 3" in air
4. **Solve** → Click "🔍 Solve Equation"
5. **View Answer** → Result shows "8"

---

## 🐛 Quick Fixes

### Camera won't start?
```python
# Edit main_app.py line ~232
cv2.VideoCapture(1)  # Try 1 instead of 0
```

### Model not found?
```bash
# Train first!
python train_model.py
```

### Import errors?
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
air_math_solver/
├── main_app.py          ← RUN THIS
├── train_model.py       ← Train model
├── verify_setup.py      ← Check setup
├── requirements.txt     ← Dependencies
│
├── utils/
│   ├── hand_tracker.py  ← Hand tracking
│   └── image_processor.py ← Recognition
│
├── models/              ← Created after training
│   └── math_symbol_classifier.h5
│
└── data/                ← Download dataset here
    └── handwritten_math_symbols/
        ├── 0/
        ├── 1/
        └── ...
```

---

## ⚡ Essential Commands

```bash
# Check Python version
python --version

# Activate virtual env (Windows)
venv\Scripts\activate

# Activate virtual env (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python verify_setup.py

# Train model
python train_model.py

# Run app
python main_app.py

# Test hand tracking only
python utils/hand_tracker.py
```

---

## 🎯 Supported Equations

✅ **Works:**
- `5+3` = 8
- `12-7` = 5
- `4x6` = 24
- `3.5+2.1` = 5.6

❌ **Not supported:**
- Division (÷)
- Powers (^)
- Parentheses ()

---

## 💡 Pro Tips

1. **Write BIG** - Fill 1/4 of screen per symbol
2. **Space symbols** - Leave gaps between numbers/operators
3. **Good lighting** - Face window/light
4. **Steady hand** - Hold still while pinching
5. **Plain background** - Avoid clutter behind you

---

## 📊 CMSC 191 Compliance

✅ 3 CV Techniques:
1. Hand Tracking (MediaPipe)
2. Contour Detection (OpenCV)
3. CNN Classification (TensorFlow)

✅ GUI: CustomTkinter (+5 bonus points)
✅ Multiple inputs: Webcam + Image upload
✅ User-friendly interface

---

## 🆘 Help!

1. Read SETUP_GUIDE.md (detailed)
2. Run `python verify_setup.py`
3. Check error messages carefully
4. Google specific errors
5. Ask classmates/instructor

---

**Made with ❤️ for CMSC 191**
