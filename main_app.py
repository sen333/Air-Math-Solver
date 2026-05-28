"""
Air Math Solver - Main GUI Application
CMSC 191 Final Project

CV Techniques:
  1. Hand Tracking / Trajectory Mapping  (MediaPipe)
  2. Contour Detection & Segmentation    (OpenCV)
  3. Image Classification                (CNN / TensorFlow)
"""

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import time
import os
from tkinter import messagebox, filedialog
from utils import SymbolSegmenter, SymbolClassifier, EquationProcessor, get_hand_tracker

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class AirMathSolverGUI:
    """Main GUI Application using CustomTkinter"""

    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Air Math Solver – CMSC 191")
        self.root.geometry("1400x820")
        self.root.resizable(True, True)

        # App state
        self.is_running = False
        self.video_capture = None
        self.drawing_canvas = None    # ndarray shared between threads

        # CV components
        self.hand_tracker = None
        self.symbol_segmenter = SymbolSegmenter(min_contour_area=200)
        self.symbol_classifier = None
        self.equation_processor = EquationProcessor()

        self.input_mode = ctk.StringVar(value="webcam")

        self._setup_gui()

        # Delay model loading until after mainloop starts so messagebox works
        self.root.after(200, self._load_model)

    # ── GUI Construction ────────────────────────────────────────────────────

    def _setup_gui(self):
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self._setup_left_panel()
        self._setup_right_panel()

    def _setup_left_panel(self):
        lf = ctk.CTkFrame(self.root)
        lf.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        lf.grid_rowconfigure(1, weight=1)
        lf.grid_rowconfigure(2, weight=1)
        lf.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(lf, text="Air Math Solver",
                     font=ctk.CTkFont(size=26, weight="bold")
                     ).grid(row=0, column=0, pady=(12, 6))

        # Webcam feed
        wc_box = ctk.CTkFrame(lf)
        wc_box.grid(row=1, column=0, padx=16, pady=(6, 4), sticky="nsew")
        wc_box.grid_rowconfigure(0, weight=1)
        wc_box.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(wc_box, text="Webcam Feed  (hand tracking overlay)",
                     font=ctk.CTkFont(size=12)).grid(row=0, column=0,
                     sticky="nw", padx=8, pady=4)
        self.webcam_label = ctk.CTkLabel(wc_box, text="Camera not started",
                                         font=ctk.CTkFont(size=13))
        self.webcam_label.grid(row=1, column=0, padx=8, pady=8, sticky="nsew")

        # Drawing canvas
        dc_box = ctk.CTkFrame(lf)
        dc_box.grid(row=2, column=0, padx=16, pady=(4, 12), sticky="nsew")
        dc_box.grid_rowconfigure(0, weight=1)
        dc_box.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(dc_box, text="Drawing Canvas  (your handwriting)",
                     font=ctk.CTkFont(size=12)).grid(row=0, column=0,
                     sticky="nw", padx=8, pady=4)
        self.canvas_label = ctk.CTkLabel(dc_box, text="Draw here using air writing",
                                         font=ctk.CTkFont(size=13))
        self.canvas_label.grid(row=1, column=0, padx=8, pady=8, sticky="nsew")

    def _setup_right_panel(self):
        rf = ctk.CTkScrollableFrame(self.root)
        rf.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        rf.grid_columnconfigure(0, weight=1)

        # ── Controls ──────────────────────────────────────────────────────
        ctrl = ctk.CTkFrame(rf)
        ctrl.pack(padx=14, pady=14, fill="x")

        ctk.CTkLabel(ctrl, text="Controls",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(10, 8))

        # Input mode
        mode_box = ctk.CTkFrame(ctrl)
        mode_box.pack(fill="x", padx=8, pady=6)
        ctk.CTkLabel(mode_box, text="Input Mode:",
                     font=ctk.CTkFont(size=13)).pack(pady=4)
        ctk.CTkRadioButton(mode_box, text="Webcam (air writing)",
                           variable=self.input_mode, value="webcam").pack(pady=3)
        ctk.CTkRadioButton(mode_box, text="Upload image",
                           variable=self.input_mode, value="upload").pack(pady=3)

        ctk.CTkButton(ctrl, text="Upload Image",
                      command=self._upload_image, height=34
                      ).pack(fill="x", padx=8, pady=4)

        self.start_btn = ctk.CTkButton(
            ctrl, text="Start Camera", command=self._toggle_camera,
            height=40, font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#2F578A", hover_color="#232F72")
        self.start_btn.pack(fill="x", padx=8, pady=6)

        ctk.CTkButton(ctrl, text="Clear Drawing",
                      command=self._clear_drawing, height=34,
                      fg_color="#232F72", hover_color="#121358"
                      ).pack(fill="x", padx=8, pady=4)

        ctk.CTkButton(ctrl, text="Solve Equation",
                      command=self._solve_equation, height=40,
                      font=ctk.CTkFont(size=14, weight="bold"),
                      fg_color="#36ADA3", hover_color="#2F578A"
                      ).pack(fill="x", padx=8, pady=6)

        self.status_lbl = ctk.CTkLabel(ctrl, text="Camera: stopped",
                                       font=ctk.CTkFont(size=12),
                                       text_color="gray")
        self.status_lbl.pack(pady=8)

        self.gesture_lbl = ctk.CTkLabel(ctrl, text="Gesture: —",
                                        font=ctk.CTkFont(size=12),
                                        text_color="gray")
        self.gesture_lbl.pack(pady=2)

        # ── Results ───────────────────────────────────────────────────────
        res = ctk.CTkFrame(rf)
        res.pack(padx=14, pady=8, fill="x")

        ctk.CTkLabel(res, text="Results",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(10, 6))

        eq_box = ctk.CTkFrame(res)
        eq_box.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(eq_box, text="Detected Equation:",
                     font=ctk.CTkFont(size=13)).pack(pady=4)
        self.equation_lbl = ctk.CTkLabel(eq_box, text="—",
                                         font=ctk.CTkFont(size=22, weight="bold"),
                                         text_color="#2F578A")
        self.equation_lbl.pack(pady=6)

        ans_box = ctk.CTkFrame(res)
        ans_box.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(ans_box, text="Answer:",
                     font=ctk.CTkFont(size=13)).pack(pady=4)
        self.answer_lbl = ctk.CTkLabel(ans_box, text="—",
                                       font=ctk.CTkFont(size=30, weight="bold"),
                                       text_color="#36ADA3")
        self.answer_lbl.pack(pady=8)

        self.conf_lbl = ctk.CTkLabel(res, text="Avg. Confidence: —",
                                     font=ctk.CTkFont(size=11),
                                     text_color="gray")
        self.conf_lbl.pack(pady=4)

        # ── Instructions ─────────────────────────────────────────────────
        inst = ctk.CTkFrame(rf)
        inst.pack(padx=14, pady=8, fill="both", expand=True)

        ctk.CTkLabel(inst, text="How to Use",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(10, 4))
        ctk.CTkLabel(inst,
                     text=(
                         "1. Start the camera\n"
                         "2. Pinch thumb & index together\n"
                         "   → green dot = writing active\n"
                         "3. Move your hand to write\n"
                         "4. Release fingers between symbols\n"
                         "5. Click Solve Equation\n\n"
                         "• Write large & clear\n"
                         "• Leave space between symbols\n"
                         "• Supported: 0-9, +, -, ×, ."
                     ),
                     font=ctk.CTkFont(size=12),
                     justify="left").pack(padx=10, pady=6)

    # ── Model Loading ───────────────────────────────────────────────────────

    def _load_model(self):
        model_path = 'models/math_symbol_classifier.h5'
        names_path = 'models/class_names.npy'
        if os.path.exists(model_path) and os.path.exists(names_path):
            try:
                self.symbol_classifier = SymbolClassifier(model_path, names_path)
                print("✓ Model loaded")
            except Exception as e:
                messagebox.showerror("Model Error", f"Failed to load model:\n{e}")
        else:
            messagebox.showwarning(
                "Model Not Found",
                "No trained model found.\n\nRun  train_model.py  first, "
                "then restart the application."
            )

    # ── Camera Control ──────────────────────────────────────────────────────

    def _toggle_camera(self):
        if self.is_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        if self.input_mode.get() != "webcam":
            messagebox.showinfo("Mode", "Switch to 'Webcam' mode first.")
            return
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error",
                                 "Could not open webcam.\n"
                                 "Try index 1 in main_app.py if you have multiple cameras.")
            return
        self.video_capture = cap
        try:
            self.hand_tracker = get_hand_tracker(detection_confidence=0.7, tracking_confidence=0.7)
        except Exception as e:
            cap.release()
            messagebox.showerror("MediaPipe Error",
                                 f"Could not initialise hand tracker:\n{e}\n\n"
                                 "Python 3.12 users: run  python download_models.py  first.")
            return
        self.is_running = True
        self.start_btn.configure(text="Stop Camera", fg_color="#121358", hover_color="#232F72")
        self.status_lbl.configure(text="Camera: running", text_color="#36ADA3")

        t = threading.Thread(target=self._video_loop, daemon=True)
        t.start()
        print("✓ Camera started")

    def _stop_camera(self):
        self.is_running = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        if self.hand_tracker:
            self.hand_tracker.release()
            self.hand_tracker = None
        self.start_btn.configure(text="Start Camera", fg_color="#2F578A", hover_color="#232F72")
        self.status_lbl.configure(text="Camera: stopped", text_color="gray")
        self.gesture_lbl.configure(text="Gesture: —", text_color="gray")
        print("✓ Camera stopped")

    # ── Video Thread ────────────────────────────────────────────────────────

    def _video_loop(self):
        """
        Runs in a daemon thread.
        All Tkinter widget updates are dispatched to the main thread
        via root.after() to avoid thread-safety crashes.
        """
        target_fps = 30
        frame_interval = 1.0 / target_fps

        while self.is_running and self.video_capture:
            t0 = time.time()

            ret, frame = self.video_capture.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            annotated, canvas, is_writing = self.hand_tracker.process_frame(frame)

            # Store canvas for solve step (shared; both threads only write same array)
            self.drawing_canvas = canvas

            # Schedule GUI updates on main thread
            self.root.after(0, self._update_webcam_display, annotated.copy())
            self.root.after(0, self._update_canvas_display,
                            canvas.copy() if canvas is not None else None)
            self.root.after(0, self._update_gesture_label, is_writing)

            # Cap frame rate
            elapsed = time.time() - t0
            wait = frame_interval - elapsed
            if wait > 0:
                time.sleep(wait)

    # ── GUI Update Helpers (main thread only) ───────────────────────────────

    def _update_webcam_display(self, frame):
        img = Image.fromarray(cv2.cvtColor(
            cv2.resize(frame, (560, 360)), cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.webcam_label.configure(image=imgtk, text="")
        self.webcam_label.image = imgtk      # keep reference

    def _update_canvas_display(self, canvas):
        if canvas is None:
            return
        img = Image.fromarray(cv2.cvtColor(
            cv2.resize(canvas, (560, 360)), cv2.COLOR_GRAY2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas_label.configure(image=imgtk, text="")
        self.canvas_label.image = imgtk

    def _update_gesture_label(self, is_writing):
        if is_writing:
            self.gesture_lbl.configure(text="Gesture: WRITING", text_color="#36ADA3")
        else:
            self.gesture_lbl.configure(text="Gesture: idle", text_color="gray")

    # ── Actions ─────────────────────────────────────────────────────────────

    def _clear_drawing(self):
        if self.hand_tracker:
            self.hand_tracker.clear_drawing()
        self.drawing_canvas = None
        self.equation_lbl.configure(text="—")
        self.answer_lbl.configure(text="—")
        self.conf_lbl.configure(text="Avg. Confidence: —")
        print("✓ Drawing cleared")

    def _upload_image(self):
        path = filedialog.askopenfilename(
            title="Select handwritten equation image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("File Error", "Could not load image.")
            return
        self.drawing_canvas = img
        self._update_canvas_display(img)
        print(f"✓ Image loaded: {path}")

    def _solve_equation(self):
        if self.drawing_canvas is None:
            messagebox.showinfo("No Drawing", "Draw an equation first.")
            return
        if self.symbol_classifier is None:
            messagebox.showerror("Model Missing",
                                 "No model loaded. Run train_model.py first.")
            return

        try:
            canvas = self.drawing_canvas.copy()

            # CV Technique #2 – Contour Detection & Segmentation
            symbols, boxes, vis = self.symbol_segmenter.segment_symbols(canvas)
            if not symbols:
                messagebox.showwarning("Nothing Detected",
                                       "No symbols found. Write larger and more clearly.")
                return
            print(f"✓ Detected {len(symbols)} symbol(s)")

            # CV Technique #3 – Image Classification (batched)
            predictions = self.symbol_classifier.classify_symbols(symbols)

            display_eq, result, conf_scores = \
                self.equation_processor.process_predictions(predictions)

            self.equation_lbl.configure(
                text=display_eq if display_eq else "—")
            self.answer_lbl.configure(text=str(result))
            avg_conf = np.mean(conf_scores) * 100 if conf_scores else 0.0
            self.conf_lbl.configure(text=f"Avg. Confidence: {avg_conf:.1f}%")

            print(f"✓ {display_eq} = {result}  (conf {avg_conf:.1f}%)")

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))
            print(f"✗ {e}")

    # ── Window Lifecycle ────────────────────────────────────────────────────

    def _on_close(self):
        self._stop_camera()
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()


# ── Entry Point ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Air Math Solver – CMSC 191 Final Project")
    print("  CV Techniques: Hand Tracking | Contour Detection | CNN")
    print("=" * 60)
    AirMathSolverGUI().run()


if __name__ == "__main__":
    main()
