"""
Image Processing Module
CV Technique #2: Contour Detection & Segmentation
CV Technique #3: Image Classification
Isolates individual symbols and classifies them using CNN
"""

import cv2
import numpy as np
from tensorflow import keras


class SymbolSegmenter:
    """Segment individual symbols from a drawn equation using contour detection"""

    def __init__(self, min_contour_area=200):
        self.min_contour_area = min_contour_area

    def preprocess_image(self, image):
        """Blur → binary threshold → morphological clean-up"""
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary

    def find_symbol_contours(self, binary_image):
        """Return external contours that exceed the minimum area threshold"""
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        valid_contours, bounding_boxes = [], []
        for c in contours:
            if cv2.contourArea(c) > self.min_contour_area:
                valid_contours.append(c)
                bounding_boxes.append(cv2.boundingRect(c))
        return valid_contours, bounding_boxes

    def sort_left_to_right(self, bounding_boxes):
        """Sort bounding boxes by x-coordinate (reading order)"""
        return sorted(bounding_boxes, key=lambda b: b[0])

    def extract_symbol_roi(self, image, bounding_box, target_size=45):
        """Crop symbol, pad to square, resize to target_size, normalise to [0,1]"""
        x, y, w, h = bounding_box
        roi = image[y:y + h, x:x + w]

        # Pad to square so aspect ratio is preserved
        max_dim = max(w, h)
        # Add a tiny little bit of padding (e.g. 10%) so it doesn't touch the absolute edge
        pad = int(max_dim * 0.15)
        max_dim += pad * 2
        
        square = np.zeros((max_dim, max_dim), dtype=np.uint8)
        y_off = (max_dim - h) // 2
        x_off = (max_dim - w) // 2
        square[y_off:y_off + h, x_off:x_off + w] = roi

        resized = cv2.resize(square, (target_size, target_size))
        
        # INVERT colors! The CNN data was black ink on white background!
        inverted = cv2.bitwise_not(resized)
        return inverted.astype('float32') / 255.0

    def segment_symbols(self, image):
        """
        Full pipeline: preprocess → find contours → sort → extract ROIs.

        Returns:
            symbols       – list of (45,45) float32 arrays, normalised
            sorted_boxes  – matching bounding boxes in reading order
            visualization – BGR image with bounding-box overlays
        """
        binary = self.preprocess_image(image)
        _, bounding_boxes = self.find_symbol_contours(binary)

        if not bounding_boxes:
            return [], [], image

        sorted_boxes = self.sort_left_to_right(bounding_boxes)
        symbols = [self.extract_symbol_roi(image, box) for box in sorted_boxes]

        # Build visualisation
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for i, (x, y, w, h) in enumerate(sorted_boxes):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis, str(i + 1), (x, max(y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return symbols, sorted_boxes, vis


class SymbolClassifier:
    """Classify segmented symbols using a trained CNN"""

    def __init__(self, model_path, class_names_path):
        self.model = keras.models.load_model(model_path)
        self.class_names = np.load(class_names_path, allow_pickle=True)

    def classify_symbols(self, symbol_images):
        """
        Classify a list of symbols in a single batched predict call (fast).

        Returns:
            list of (predicted_class, confidence) tuples
        """
        if not symbol_images:
            return []

        # Stack into one batch: shape (N, 45, 45, 1)
        batch = np.array(symbol_images).reshape(-1, 45, 45, 1)
        preds = self.model.predict(batch, verbose=0)  # one call for the whole batch

        results = []
        for pred in preds:
            idx = int(np.argmax(pred))
            results.append((self.class_names[idx], float(pred[idx])))
        return results


class EquationProcessor:
    """Build a display string and an eval string, then solve"""

    def __init__(self):
        self._valid_digits = set('0123456789')
        self._valid_eval_ops = set('+-*/')

    def build_strings(self, predictions):
        """
        Convert predictions to two strings:
          display_eq – human-readable (uses × for multiplication)
          eval_eq    – Python-evaluable (uses * for multiplication)
        """
        display_parts = []
        eval_parts = []

        for symbol, _ in predictions:
            if symbol == '=':
                continue
            if symbol == 'x':
                display_parts.append('×')
                eval_parts.append('*')
            else:
                display_parts.append(str(symbol))
                eval_parts.append(str(symbol))

        return ''.join(display_parts), ''.join(eval_parts)

    def validate_eval_equation(self, equation):
        """Basic sanity check before passing to eval()"""
        if not equation:
            return False, "Empty equation"

        valid_chars = self._valid_digits | self._valid_eval_ops | {'.'}
        for ch in equation:
            if ch not in valid_chars:
                return False, f"Invalid character: '{ch}'"

        # No consecutive operators
        ops = list(self._valid_eval_ops)
        for i in range(len(equation) - 1):
            if equation[i] in ops and equation[i + 1] in ops:
                return False, "Consecutive operators"

        return True, ""

    def solve(self, eval_eq):
        """Evaluate the equation string and return a rounded result or error string"""
        ok, msg = self.validate_eval_equation(eval_eq)
        if not ok:
            return f"Error: {msg}"
        try:
            result = eval(eval_eq)             # safe: only digits + +-*./ are allowed
            return round(result, 4)
        except ZeroDivisionError:
            return "Error: division by zero"
        except Exception as e:
            return f"Error: {e}"

    def process_predictions(self, predictions):
        """
        Full pipeline: predictions → (display_eq, eval_eq) → result.

        Returns:
            display_eq       – human-readable equation string (× for multiply)
            result           – numeric result or error string
            confidence_scores – list of per-symbol confidences
        """
        display_eq, eval_eq = self.build_strings(predictions)
        result = self.solve(eval_eq)
        confidence_scores = [conf for _, conf in predictions]
        return display_eq, result, confidence_scores


if __name__ == "__main__":
    print("image_processor.py — SymbolSegmenter / SymbolClassifier / EquationProcessor")
