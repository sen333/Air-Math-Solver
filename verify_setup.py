"""
verify_setup.py — checks every component before you run the app.
Run:  python verify_setup.py
"""

import sys, os

def hdr(text):
    print(f"\n{'='*58}\n  {text}\n{'='*58}")

def ok(msg):  print(f"  ✓  {msg}")
def fail(msg):print(f"  ✗  {msg}")
def warn(msg):print(f"  ⚠  {msg}")

# ── 1. Python version ────────────────────────────────────────────────────────
hdr("1 / 6  Python Version")
vi = sys.version_info
ver_str = f"{vi.major}.{vi.minor}.{vi.micro}"
if vi.major == 3 and vi.minor in (10, 11):
    ok(f"Python {ver_str}  ← ideal")
elif vi.major == 3 and vi.minor == 12:
    warn(f"Python {ver_str}  — supported, but run  python download_models.py  after pip install")
else:
    fail(f"Python {ver_str}  — need 3.10, 3.11, or 3.12")
python_ok = vi.major == 3 and vi.minor >= 10

# ── 2. Dependencies ──────────────────────────────────────────────────────────
hdr("2 / 6  Dependencies")
deps = [
    ("cv2",          "opencv-python"),
    ("mediapipe",    "mediapipe"),
    ("tensorflow",   "tensorflow"),
    ("numpy",        "numpy"),
    ("customtkinter","customtkinter"),
    ("PIL",          "Pillow"),
    ("sklearn",      "scikit-learn"),
    ("matplotlib",   "matplotlib"),
]
deps_ok = True
for mod, pkg in deps:
    try:
        m   = __import__(mod)
        ver = getattr(m, "__version__", "?")
        ok(f"{pkg:<22} {ver}")
    except ImportError:
        fail(f"{pkg:<22} NOT INSTALLED  →  pip install {pkg}")
        deps_ok = False

# ── 3. MediaPipe API detection ───────────────────────────────────────────────
hdr("3 / 6  MediaPipe API")
mp_ok = False
try:
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        ok("Legacy API (mp.solutions.hands) available — no model file needed")
        mp_ok = True
    else:
        warn("New Tasks API detected (mediapipe ≥ 0.10.14 / Python 3.12)")
        model_path = os.path.join("models", "hand_landmarker.task")
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / 1024 / 1024
            ok(f"hand_landmarker.task found ({size:.1f} MB)")
            mp_ok = True
        else:
            fail("hand_landmarker.task NOT found")
            print("        Run:  python download_models.py")
except Exception as e:
    fail(f"mediapipe import failed: {e}")

# ── 4. Project files ─────────────────────────────────────────────────────────
hdr("4 / 6  Project Files")
required = [
    "main_app.py",
    "train_model.py",
    "download_models.py",
    "requirements.txt",
    "utils/__init__.py",
    "utils/hand_tracker.py",
    "utils/image_processor.py",
]
files_ok = True
for f in required:
    if os.path.exists(f):
        ok(f)
    else:
        fail(f"{f}  MISSING")
        files_ok = False

# ── 5. Dataset ───────────────────────────────────────────────────────────────
hdr("5 / 6  Dataset")
ds_path  = os.path.join("data", "handwritten_math_symbols")
expected = ["0","1","2","3","4","5","6","7","8","9","+","-","times"]
ds_ok    = False
if not os.path.isdir(ds_path):
    fail(f"Folder not found: {ds_path}")
    print("        Download from: https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols")
else:
    missing = [c for c in expected if not os.path.isdir(os.path.join(ds_path, c))]
    total   = sum(
        len([f for f in os.listdir(os.path.join(ds_path, c))
             if f.lower().endswith((".png",".jpg",".jpeg"))])
        for c in expected if os.path.isdir(os.path.join(ds_path, c))
    )
    if missing:
        fail(f"Missing class folders: {', '.join(missing)}")
    elif total < 1000:
        warn(f"Only {total} images found (expected ≥ 10,000) — dataset may be incomplete")
    else:
        ok(f"All 14 class folders present  |  {total:,} images")
        ds_ok = True

# ── 6. Trained model ─────────────────────────────────────────────────────────
hdr("6 / 6  Trained CNN Model")
h5_path    = os.path.join("models", "math_symbol_classifier.h5")
names_path = os.path.join("models", "class_names.npy")
model_ok   = False
if os.path.exists(h5_path) and os.path.exists(names_path):
    mb = os.path.getsize(h5_path) / 1024 / 1024
    ok(f"math_symbol_classifier.h5  ({mb:.1f} MB)")
    ok("class_names.npy")
    model_ok = True
else:
    fail("Model not found — run:  python train_model.py")

# ── Summary ──────────────────────────────────────────────────────────────────
hdr("Summary")
checks = [
    ("Python version",  python_ok),
    ("Dependencies",    deps_ok),
    ("MediaPipe API",   mp_ok),
    ("Project files",   files_ok),
    ("Dataset",         ds_ok),
    ("Trained model",   model_ok),
]
all_ok = True
for name, result in checks:
    s = "✓ PASS" if result else "✗ FAIL"
    print(f"  {s}  {name}")
    if not result:
        all_ok = False

print()
if all_ok:
    print("  🎉  Everything ready — run:  python main_app.py\n")
else:
    print("  Fix the ✗ items above, then re-run this script.\n")
    if not ds_ok:
        print("  Dataset:  https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols")
    if not model_ok:
        print("  Model:    python train_model.py")
    if not mp_ok:
        print("  MediaPipe model: python download_models.py")
