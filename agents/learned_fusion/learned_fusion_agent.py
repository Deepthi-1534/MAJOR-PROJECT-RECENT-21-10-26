
"""learned_fusion_agent.py
Learned fusion driven by the trained meta-classifier.
Independent of the rule-based FusionAgent.
"""

import numpy as np
import joblib

from .features import (
    FEATURE_NAMES,
    ALL_FEATURE_NAMES,
    candidates_to_matrix,
)


# --------------------------------------------------
# Final decision labels
# --------------------------------------------------

STATUS_DETECTED = "DETECTED"
STATUS_CLEAR = "CLEAR"
STATUS_REVIEW = "REVIEW"

LABELS = {
    STATUS_DETECTED: "Camouflaged Object",
    STATUS_CLEAR: "Not Camouflaged",
    STATUS_REVIEW: "Needs Review",
}

SUBLABELS = {
    "H0": "Camouflaged object",
    "H1": "Ordinary object (not camouflaged)",
    "H2": "No object",
}


class LearnedFusionAgent:
    """
    Combines detection-agent evidence and cognitive outputs
    using a trained meta-classifier.

    Public methods:
        predict_proba_batch()
        predict_proba()
        fuse_many()
        fuse_candidate()
        analyze()
    """

    def __init__(
        self,
        model_path,
        detect_threshold=None,
        clear_threshold=None,
    ):
        # Load the trained model bundle.
        bundle = joblib.load(model_path)

        names = bundle["feature_names"]

        # Verify compatibility with the model's training features.
        if names == ALL_FEATURE_NAMES:
            self.use_context = True
        elif names == FEATURE_NAMES:
            self.use_context = False
        else:
            raise ValueError(
                "Model was trained with different features. "
                "Rebuild the dataset and retrain."
            )

        self.names = names
        self.model = bundle["model"]
        self.calibrator = bundle["calibrator"]
        self.algo = bundle["algo"]

        self.detect_t = (
            detect_threshold
            if detect_threshold is not None
            else bundle["detect_threshold"]
        )

        self.clear_t = (
            clear_threshold
            if clear_threshold is not None
            else bundle["clear_threshold"]
        )

        if not 0 <= self.clear_t < self.detect_t <= 1:
            raise ValueError(
                "Thresholds must satisfy "
                "0 <= clear_threshold < detect_threshold <= 1."
            )

    # --------------------------------------------------
    # Probability prediction
    # --------------------------------------------------

    @staticmethod
    def _logit(p):
        """Convert probabilities into logits for calibration."""
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def predict_proba_batch(self, matrix):
        """Return calibrated camouflage probabilities for a batch."""
        raw = self.model.predict_proba(matrix)[:, 1]

        calibrated = self.calibrator.predict_proba(
            self._logit(raw)
        )[:, 1]

        return calibrated

    def predict_proba(self, features):
        """Return the calibrated probability for one feature vector."""
        matrix = np.asarray(features, dtype=float).reshape(1, -1)
        return float(self.predict_proba_batch(matrix)[0])

    # --------------------------------------------------
    # Explainability
    # --------------------------------------------------

    def _top_drivers(self, features, k=3):
        """
        Return the strongest feature contributions for logistic
        regression models. Tree-based models return no drivers.
        """
        if self.algo != "lr":
            return []

        scaler = self.model[0]
        classifier = self.model[-1]

        transformed = scaler.transform(
            np.asarray(features, dtype=float).reshape(1, -1)
        )[0]

        contributions = classifier.coef_[0] * transformed
        order = np.argsort(-np.abs(contributions))[:k]

        return [
            f"{self.names[i]} ({contributions[i]:+.2f})"
            for i in order
        ]

    # --------------------------------------------------
    # Candidate validation
    # --------------------------------------------------

    @staticmethod
    def _reject_if_implausible(candidate):
        """
        Reject malformed or whole-image bounding boxes.

        Assumes bbox is [x1, y1, x2, y2] and image dimensions
        are available as image_width/image_height when provided.

        Adjust this validation if your project uses another
        bounding-box format.
        """
        bbox = candidate.get("bbox")

        if bbox is None:
            return None

        try:
            if len(bbox) != 4:
                raise ValueError

            x1, y1, x2, y2 = map(float, bbox)

            if not np.all(np.isfinite([x1, y1, x2, y2])):
                raise ValueError

            if x2 <= x1 or y2 <= y1:
                return "Invalid bounding box: non-positive area."

            width = candidate.get("image_width")
            height = candidate.get("image_height")

            if width and height:
                box_area = (x2 - x1) * (y2 - y1)
                image_area = float(width) * float(height)

                if image_area > 0 and box_area / image_area >= 0.98:
                    return "Rejected implausible whole-image bounding box."

        except (TypeError, ValueError):
            return "Invalid bounding-box format."

        return None

    # --------------------------------------------------
    # Final decision
    # --------------------------------------------------

    def _decide(self, candidate, features, probability):
        """Convert learned probability into a final status."""

        rejection = self._reject_if_implausible(candidate)

        if rejection is not None:
            return {
                "final_status": STATUS_REVIEW,
                "final_label": LABELS[STATUS_REVIEW],
                "final_sublabel": "",
                "final_suspicion": 0.5,
                "final_bbox": candidate.get("bbox"),
                "final_class": candidate.get("class"),
                "fusion_method": f"learned-{self.algo}",
                "fusion_signals": {
                    "learned_probability": round(
                        float(probability), 4
                    )
                },
                "fusion_reasons": [rejection],
            }

        # Use the same feature definition as training.
        n_present_index = FEATURE_NAMES.index("n_present")
        n_present = features[n_present_index]

        reasons = []

        # No supporting agent evidence means the result is uncertain.
        if n_present == 0:
            probability = 0.5
            status = STATUS_REVIEW
            sublabel = ""

            reasons.append(
                "No detection agent reported evidence for this candidate."
            )

        elif probability >= self.detect_t:
            status = STATUS_DETECTED
            sublabel = SUBLABELS["H0"]

            reasons.append(
                f"Learned probability {probability:.2f} "
                f">= detection threshold {self.detect_t:.2f}."
            )

        elif probability <= self.clear_t:
            status = STATUS_CLEAR
            sublabel = SUBLABELS.get(
                candidate.get("bayes_hypothesis"),
                "Not camouflaged",
            )

            reasons.append(
                f"Learned probability {probability:.2f} "
                f"<= clear threshold {self.clear_t:.2f}."
            )

        else:
            status = STATUS_REVIEW
            sublabel = ""

            reasons.append(
                f"Learned probability {probability:.2f} falls "
                f"between thresholds {self.clear_t:.2f} and "
                f"{self.detect_t:.2f}."
            )

        drivers = self._top_drivers(features)

        if drivers:
            reasons.append(
                "Main drivers: " + ", ".join(drivers)
            )

        return {
            "final_status": status,
            "final_label": LABELS[status],
            "final_sublabel": sublabel,
            "final_suspicion": round(float(probability), 4),
            "final_bbox": candidate.get("bbox"),
            "final_class": candidate.get("class"),
            "fusion_method": f"learned-{self.algo}",
            "fusion_signals": {
                "learned_probability": round(
                    float(probability), 4
                )
            },
            "fusion_reasons": reasons,
        }

    # --------------------------------------------------
    # Batch and single-candidate inference
    # --------------------------------------------------

    def fuse_many(self, candidates):
        """
        Score candidates together so context features can compare
        candidates from the same image.
        """
        if not candidates:
            return []

        matrix = candidates_to_matrix(
            candidates,
            self.use_context,
        )

        probabilities = self.predict_proba_batch(matrix)

        return [
            self._decide(
                candidate,
                list(matrix[i]),
                probabilities[i],
            )
            for i, candidate in enumerate(candidates)
        ]

    def fuse_candidate(self, candidate):
        """Score one candidate."""
        return self.fuse_many([candidate])[0]

    def analyze(self, candidates):
        """Return each original candidate with its learned result."""
        results = self.fuse_many(candidates)

        return [
            {**candidate, **result}
            for candidate, result in zip(candidates, results)
        ]
