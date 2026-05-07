"""
inference.py
------------
Standalone inference pipeline.
Usage:
    from utils.inference import BreastCancerPredictor
    predictor = BreastCancerPredictor("model/best_model.keras")
    result    = predictor.predict("path/to/image.png")
    print(result)   # {'label': 'Malignant', 'confidence': 0.91, 'probability': 0.91}
"""

import numpy as np
import tensorflow as tf
from utils.data_utils import preprocess_image

THRESHOLD = 0.50
LABELS    = {0: "Benign", 1: "Malignant"}


class BreastCancerPredictor:
    """Wraps a saved Keras model for single-image inference."""

    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        print(f"[inference] Model loaded from {model_path}")

    def predict(self, image_path: str, threshold: float = THRESHOLD) -> dict:
        """
        Parameters
        ----------
        image_path : str   Path to the image file.
        threshold  : float Decision boundary (default 0.5).

        Returns
        -------
        dict with keys: label, confidence, probability, raw_score
        """
        img   = preprocess_image(image_path)
        score = float(self.model.predict(img, verbose=0)[0][0])

        label      = LABELS[int(score >= threshold)]
        confidence = score if score >= threshold else 1 - score

        return {
            "label":      label,
            "confidence": round(confidence * 100, 2),   # percentage
            "probability": round(score, 4),
            "raw_score":  score,
        }


# ── Quick CLI test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m utils.inference <model_path> <image_path>")
        sys.exit(1)

    predictor = BreastCancerPredictor(sys.argv[1])
    result    = predictor.predict(sys.argv[2])
    print(f"\nPrediction : {result['label']}")
    print(f"Confidence : {result['confidence']}%")
    print(f"Raw score  : {result['raw_score']:.4f}")
