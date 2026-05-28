"""
Utility modules for Air Math Solver.
HandTracker is imported lazily to avoid crashing when mediapipe has a
protobuf conflict with tensorflow (a known issue on Python 3.12 with
certain package version combinations).
"""

from .image_processor import SymbolSegmenter, SymbolClassifier, EquationProcessor

def get_hand_tracker(*args, **kwargs):
    """Lazy import so mediapipe errors surface at call time, not import time."""
    from .hand_tracker import HandTracker
    return HandTracker(*args, **kwargs)

__all__ = [
    "SymbolSegmenter",
    "SymbolClassifier",
    "EquationProcessor",
    "get_hand_tracker",
]
