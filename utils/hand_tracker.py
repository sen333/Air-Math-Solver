"""
Hand Tracking Module — supports both MediaPipe API generations
CV Technique #1: Hand Tracking / Trajectory Mapping

• mediapipe < 0.10.14  (Python 3.10/3.11)  → mp.solutions.hands  (legacy API)
• mediapipe ≥ 0.10.14  (Python 3.12+)      → mp.tasks HandLandmarker (Tasks API)

The Tasks API requires models/hand_landmarker.task — run download_models.py first.
"""

import cv2
import numpy as np
from collections import deque

# ── Detect which MediaPipe API is available ──────────────────────────────────
import mediapipe as mp

try:
    _test = mp.solutions.hands          # AttributeError on new mediapipe
    MEDIAPIPE_API = "legacy"
except AttributeError:
    MEDIAPIPE_API = "tasks"

# Hard-coded hand connections (same in both APIs)
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


class HandTracker:
    """
    Tracks hand landmarks and maps index-finger trajectory onto a canvas.
    Works transparently with both the legacy and Tasks MediaPipe APIs.
    """

    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.5):
        self._max_hands = max_hands
        self._det_conf   = detection_confidence
        self._trk_conf   = tracking_confidence

        self.trajectory_points = deque(maxlen=1000)
        self.drawing_canvas    = None

        if MEDIAPIPE_API == "legacy":
            self._init_legacy()
        else:
            self._init_tasks()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_legacy(self):
        """Set up mp.solutions.hands (mediapipe < 0.10.14)"""
        self._mp_hands     = mp.solutions.hands
        self._mp_drawing   = mp.solutions.drawing_utils
        self._mp_styles    = mp.solutions.drawing_styles
        self._detector     = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self._max_hands,
            min_detection_confidence=self._det_conf,
            min_tracking_confidence=self._trk_conf,
        )
        print("  HandTracker: using MediaPipe legacy API (mp.solutions.hands)")

    def _init_tasks(self):
        """Set up mp.tasks HandLandmarker (mediapipe ≥ 0.10.14)"""
        import os
        from mediapipe.tasks.python import vision as mpv
        from mediapipe.tasks import python as mpt

        model_path = os.path.join("models", "hand_landmarker.task")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Tasks API model not found: {model_path}\n"
                "Run  python download_models.py  to download it."
            )

        options = mpv.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=model_path),
            num_hands=self._max_hands,
            min_hand_detection_confidence=self._det_conf,
            min_tracking_confidence=self._trk_conf,
        )
        self._detector = mpv.HandLandmarker.create_from_options(options)
        print("  HandTracker: using MediaPipe Tasks API (HandLandmarker)")

    # ── Canvas helpers ────────────────────────────────────────────────────────

    def reset_canvas(self, width, height):
        self.drawing_canvas = np.zeros((height, width), dtype=np.uint8)
        self.trajectory_points.clear()

    def clear_drawing(self):
        if self.drawing_canvas is not None:
            h, w = self.drawing_canvas.shape[:2]
            self.reset_canvas(w, h)

    def get_drawing_canvas(self):
        return self.drawing_canvas.copy() if self.drawing_canvas is not None else None

    # ── Gesture helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _pinch_distance(lm_list):
        """Normalised thumb-tip (4) to index-tip (8) distance"""
        t, i = lm_list[4], lm_list[8]
        return np.hypot(t.x - i.x, t.y - i.y)

    @staticmethod
    def _tip_coords(lm_list, frame_shape):
        h, w = frame_shape[:2]
        tip = lm_list[8]
        return int(tip.x * w), int(tip.y * h)

    # ── Trajectory ────────────────────────────────────────────────────────────

    def update_trajectory(self, x, y, is_writing):
        if is_writing:
            self.trajectory_points.append((x, y))
            if len(self.trajectory_points) > 1:
                cv2.line(self.drawing_canvas,
                         self.trajectory_points[-2], (x, y),
                         color=255, thickness=8)
        else:
            self.trajectory_points.clear()  # break stroke so no jump on re-pinch

    # ── Drawing overlay ───────────────────────────────────────────────────────

    def _draw_landmarks(self, frame, lm_list):
        """Draw skeleton overlay manually (works for both APIs)"""
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]

        for a, b in _HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (80, 200, 80), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 255, 0), -1)

    # ── Main process method ───────────────────────────────────────────────────

    def process_frame(self, frame):
        """
        Process one BGR video frame.

        Returns:
            annotated_frame  – frame with hand skeleton drawn on it
            drawing_canvas   – accumulated trajectory (grayscale)
            is_writing       – True while pinch gesture is active
        """
        if self.drawing_canvas is None:
            h, w = frame.shape[:2]
            self.reset_canvas(w, h)

        annotated = frame.copy()
        is_writing = False

        if MEDIAPIPE_API == "legacy":
            landmarks_list = self._detect_legacy(frame)
        else:
            landmarks_list = self._detect_tasks(frame)

        for lm_list in landmarks_list:
            self._draw_landmarks(annotated, lm_list)

            x, y       = self._tip_coords(lm_list, frame.shape)
            is_writing = self._pinch_distance(lm_list) < 0.06
            self.update_trajectory(x, y, is_writing)

            # Indicator dot: green = writing, red = idle
            cv2.circle(annotated, (x, y), 12,
                       (0, 255, 0) if is_writing else (0, 0, 255), -1)

        return annotated, self.drawing_canvas, is_writing

    # ── API-specific detection ─────────────────────────────────────────────────

    def _detect_legacy(self, frame):
        """Return list of landmark lists using legacy mp.solutions.hands"""
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._detector.process(rgb)
        if not result.multi_hand_landmarks:
            return []
        return [h.landmark for h in result.multi_hand_landmarks]

    def _detect_tasks(self, frame):
        """Return list of landmark lists using mp.tasks HandLandmarker"""
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._detector.detect(mp_image)
        return result.hand_landmarks   # list of list[NormalizedLandmark]

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def release(self):
        if MEDIAPIPE_API == "legacy":
            self._detector.close()
        else:
            self._detector.close()


# ── Standalone test ───────────────────────────────────────────────────────────
def test_hand_tracker():
    cap     = cv2.VideoCapture(0)
    tracker = HandTracker()
    print("Pinch thumb+index to write | 'c' clear | 'q' quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        annotated, canvas, writing = tracker.process_frame(frame)

        if canvas is not None:
            overlay  = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)

        cv2.putText(annotated,
                    "WRITING" if writing else "idle",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0) if writing else (0,0,255), 2)
        cv2.imshow("Hand Tracker", annotated)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('c'):
            tracker.clear_drawing()

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()


if __name__ == "__main__":
    test_hand_tracker()
