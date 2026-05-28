
## 📖 Project Overview

**Air Math Solver** is an interactive computer vision application that allows users to write mathematical equations in the air using hand gestures. The system tracks hand movements, recognizes handwritten symbols, and automatically solves the equation.

### 🎯 Key Features

- ✍️ **Air Writing**: Write equations in the air using hand gestures
- 🎥 **Real-time Processing**: Live webcam feed with hand tracking visualization
- 🔍 **Symbol Recognition**: Detects and classifies 0-9 and operators (+, -, ×, .)
- 🧮 **Automatic Solving**: Computes results instantly
- 🖥️ **Modern GUI**: Intuitive CustomTkinter interface
- 📁 **Multi-Input Support**: Webcam or image file upload

---

## 🔬 Computer Vision Techniques (CMSC 191 Requirements)

This project integrates **three distinct CV techniques**:

### 1️⃣ Hand Tracking / Trajectory Mapping (MediaPipe)
- **Technology**: MediaPipe Hands
- **Purpose**: Tracks 21 hand landmarks in real-time
- **Implementation**: Maps index finger tip coordinates to create drawing trajectory
- **Gesture Detection**: Recognizes "writing" gesture (thumb-index proximity)

### 2️⃣ Contour Detection & Segmentation (OpenCV)
- **Technology**: OpenCV contour finding algorithms
- **Purpose**: Isolates individual mathematical symbols from drawing
- **Implementation**: 
  - Preprocessing with Gaussian blur and binary thresholding
  - Morphological operations (closing, opening)
  - Contour detection and bounding box extraction
  - Left-to-right sorting for reading order

### 3️⃣ Image Classification (Deep Learning CNN)
- **Technology**: TensorFlow/Keras Convolutional Neural Network
- **Purpose**: Classifies each segmented symbol
- **Architecture**:
  - 3 Convolutional blocks with BatchNorm and Dropout
  - MaxPooling layers for feature extraction
  - Dense layers for classification
  - Softmax output for 14 classes (0-9, +, -, ×, .)

---

## 📦 Project Structure

```
air_math_solver/
├── main_app.py                 # Main GUI application
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── utils/
│   ├── __init__.py
│   ├── hand_tracker.py        # Hand tracking module (CV Technique #1)
│   └── image_processor.py     # Segmentation & classification (CV Technique #2 & #3)
│
├── models/                     # Saved models (created after training)
│   ├── math_symbol_classifier.h5
│   ├── class_names.npy
│   └── training_history.png
│
└── data/                       # Dataset folder
    └── handwritten_math_symbols/
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Webcam (for air writing mode)
- ~2GB disk space (for dataset and models)

### Step 1: Clone/Download Project

```bash
# If using git
git clone <repository-url>
cd air_math_solver

# Or simply navigate to the project folder
cd air_math_solver
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies installed:**
- `opencv-python` - Computer vision operations
- `mediapipe` - Hand tracking
- `tensorflow` - Deep learning model
- `numpy` - Numerical computations
- `customtkinter` - Modern GUI framework
- `Pillow` - Image processing
- `scikit-learn` - Machine learning utilities
- `matplotlib` - Visualization
- `kaggle` - Dataset download

### Step 4: Download Dataset

#### Option A: Manual Download (Recommended)

1. Go to [Kaggle Handwritten Math Symbols Dataset](https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols)
2. Download the dataset
3. Extract to `air_math_solver/data/handwritten_math_symbols/`

Expected structure:
```
data/handwritten_math_symbols/
├── 0/
├── 1/
├── 2/
├── .../
├── 9/
├── +/
├── -/
├── times/
└── dot/
```

#### Option B: Using Kaggle API

```bash
# Install Kaggle API
pip install kaggle

# Configure API credentials (follow Kaggle instructions)
# Download dataset
kaggle datasets download -d xainano/handwrittenmathsymbols
unzip handwrittenmathsymbols.zip -d data/handwritten_math_symbols/
```

### Step 5: Train the Model

```bash
python train_model.py
```

**Training details:**
- Duration: ~10-20 minutes (depending on hardware)
- Dataset: ~10,000+ images across 14 classes
- Validation accuracy: Expected >95%
- Output: `models/math_symbol_classifier.h5`

**Training output includes:**
- Model file (.h5)
- Class names mapping (.npy)
- Training history plot (.png)

---

## 🎮 Usage Guide

### Running the Application

```bash
python main_app.py
```

### Using Air Writing Mode

1. **Start Camera**: Click "▶️ Start Camera" button
2. **Writing Gesture**: 
   - Pinch thumb and index finger together
   - Green circle = Writing mode active
   - Red circle = Not writing
3. **Write Equation**: 
   - Move your hand to draw numbers and operators
   - Example: Draw "5", "+", "3"
4. **Finish Writing**: Release fingers (circle turns red)
5. **Solve**: Click "🔍 Solve Equation"
6. **View Result**: Answer appears in the results panel

### Using Image Upload Mode

1. Select "Upload Image" radio button
2. Click "📁 Upload Image"
3. Choose an image file containing handwritten equation
4. Click "🔍 Solve Equation"
5. View results

### Controls

- **🗑️ Clear Drawing**: Reset canvas and results
- **⏸️ Stop Camera**: Stop webcam feed
- **Input Mode Toggle**: Switch between webcam and upload

---

## 📊 Technical Details

### Hand Tracking Configuration

```python
HandTracker(
    max_hands=1,
    detection_confidence=0.7,
    tracking_confidence=0.5
)
```

- **Landmarks**: 21 hand keypoints tracked
- **Gesture Detection**: Thumb-index distance threshold: 0.1
- **Trajectory**: Deque with max 1000 points
- **Drawing Thickness**: 8 pixels

### Symbol Segmentation Parameters

```python
SymbolSegmenter(
    min_contour_area=200
)
```

- **Preprocessing**: Gaussian blur (5×5 kernel)
- **Thresholding**: Binary threshold at 30
- **Morphological Ops**: 3×3 kernel for closing/opening
- **Sorting**: Left-to-right by x-coordinate

### CNN Architecture

```
Input: 45×45×1 grayscale images

Conv2D(32, 3×3) + BatchNorm + MaxPool + Dropout(0.25)
Conv2D(64, 3×3) + BatchNorm + MaxPool + Dropout(0.25)
Conv2D(128, 3×3) + BatchNorm + MaxPool + Dropout(0.25)
Flatten
Dense(256) + BatchNorm + Dropout(0.5)
Dense(128) + Dropout(0.5)
Dense(14, softmax)

Total Parameters: ~1.5M
Optimizer: Adam (lr=0.001)
Loss: Sparse Categorical Crossentropy
```

### Supported Operations

- **Digits**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Operators**: + (addition), - (subtraction), × (multiplication), . (decimal)
- **Auto-conversion**: × → * for Python eval()

---

## 🎨 GUI Features

### Layout

- **Left Panel**: 
  - Webcam feed with hand tracking overlay
  - Drawing canvas showing trajectory
  
- **Right Panel**:
  - Input mode selection
  - Control buttons
  - Results display
  - Instructions

### Themes

- **Appearance**: Dark mode
- **Color Scheme**: Blue accent
- **Status Indicators**: 
  - 🟢 Green: Active/Running
  - 🔴 Red: Stopped/Inactive
  - ⚪ Gray: Neutral

---

## 🧪 Testing

### Manual Testing Checklist

- ✅ Camera initialization
- ✅ Hand detection accuracy
- ✅ Writing gesture recognition
- ✅ Symbol segmentation
- ✅ Classification accuracy
- ✅ Equation solving
- ✅ Image upload functionality
- ✅ Clear/reset operations

### Test Cases

**Simple Equations:**
- `5+3` → 8
- `9-4` → 5
- `2×6` → 12

**Complex Equations:**
- `12+34` → 46
- `100-25` → 75
- `3×7+2` → 23

**Decimal Numbers:**
- `3.5+2.5` → 6.0
- `10.2-5.1` → 5.1

---

## 🐛 Troubleshooting

### Common Issues

**❌ Camera not opening**
- Check webcam permissions
- Try different camera index: `cv2.VideoCapture(1)`
- Ensure no other app is using camera

**❌ Model not found**
- Run `train_model.py` first
- Check `models/` directory exists
- Verify file paths

**❌ Hand not detected**
- Ensure good lighting
- Keep hand in frame
- Adjust detection confidence

**❌ Low recognition accuracy**
- Write symbols clearly and large
- Wait for trajectory to register
- Retrain model with more epochs

**❌ Import errors**
- Verify all dependencies installed
- Check Python version (3.8+)
- Reinstall requirements

---

## 📝 Course Requirements Compliance

### ✅ CMSC 191 Requirements Met

| Requirement | Implementation | Status |
|------------|----------------|---------|
| 3+ CV Techniques | Hand Tracking, Contour Detection, CNN Classification | ✅ |
| GUI Interface | CustomTkinter (5 bonus points) | ✅ |
| Multiple Input Types | Webcam + Image Upload | ✅ |
| Practical Application | Math equation solver | ✅ |
| User-Friendly | Intuitive controls & instructions | ✅ |
| Visualization | Real-time feed + results display | ✅ |

### Bonus Points

- **+5 points**: GUI using Python library (CustomTkinter) ✅
- **Advanced Features**: Real-time processing, gesture detection ✅

---

## 🎓 Educational Value

### Learning Outcomes

1. **Hand Tracking**: Understanding pose estimation and landmark detection
2. **Image Segmentation**: Contour-based object isolation techniques
3. **Deep Learning**: CNN architecture and training for image classification
4. **Integration**: Combining multiple CV techniques into cohesive application
5. **GUI Development**: Creating user-friendly interfaces for CV apps

### Potential Extensions

- 🔢 Support for more operators (÷, ^, √)
- 📊 Multi-line equation support
- 🎯 Improved gesture recognition
- 📱 Mobile app version
- 🌐 Web-based deployment
- 📈 Equation history tracking
- 🎨 Custom drawing colors

---

## 👥 Credits

**CMSC 191 Final Project**

- **Dataset**: [Handwritten Math Symbols](https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols) by xainano on Kaggle
- **Hand Tracking**: MediaPipe by Google
- **GUI Framework**: CustomTkinter by Tom Schimansky
- **Deep Learning**: TensorFlow by Google

---

## 📄 License

This project is created for educational purposes as part of CMSC 191 coursework.

---

## 🆘 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review code comments
3. Test individual modules separately
4. Contact project team members

---

**Ready to solve equations in the air! ✍️➗🔢**
