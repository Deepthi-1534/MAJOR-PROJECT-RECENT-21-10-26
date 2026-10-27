"""cognitive_agent.py

Integrates hypothesis testing, Bayesian belief updating,
and metacognitive monitoring.

Accepts numeric evidence scores or agent-result dictionaries
containing a recognized numeric score field.
"""

import math
from numbers import Real
from typing import Dict, Optional, Any

from .common import (
    HYPS,
    NAMES,
    DEFAULT_MODELS,
    load_calibrated_models,
    Evidence,
    CognitiveResult,
    clip01,
)
from .hypothesis_testing import hypothesis_test
from .bayesian_update import bayesian_update
from .metacognition import metacognitive_monitor


class CognitiveAgent:
    """Cognitive reasoning layer for multi-agent camouflage detection."""

    # Fields that may contain a score produced by an upstream agent.
    # These are checked in order; the first valid numeric field is used.
    SCORE_FIELDS = (
        "score",
        "suspicion_score",
        "camouflage_score",
        "confidence",
        "average_confidence",
        "probability",
        "prob",
    )

    def __init__(
        self,
        models=None,
        prior: Optional[Dict[str, float]] = None,
        alpha: float = 0.05,
        bf_threshold: float = 3.0,
        accept_threshold: float = 0.5,
    ):
        # explicit models > models/cognitive_models.json (from calibrate.py) > built-in assumptions
        self.models = models or load_calibrated_models() or DEFAULT_MODELS
        self.prior = prior
        self.alpha = alpha
        self.bf = bf_threshold
        self.accept = accept_threshold

    @classmethod
    def _extract_score(cls, value: Any, agent_name: str):
        """Extract a normalized numeric score from supported input types.

        Returns None when the upstream agent did not report evidence.
        Raises ValueError for unsupported or ambiguous dictionary formats.
        """

        if value is None:
            return None

        # Booleans are numeric subclasses in Python, but are not valid scores.
        if isinstance(value, bool):
            raise ValueError(
                f"{agent_name} evidence must be a numeric score, "
                "not a boolean."
            )

        if isinstance(value, Real):
            score = float(value)

        elif isinstance(value, dict):
            score = None

            for key in cls.SCORE_FIELDS:
                candidate = value.get(key)

                if (
                    isinstance(candidate, Real)
                    and not isinstance(candidate, bool)
                ):
                    score = float(candidate)
                    break

            if score is None:
                available_keys = ", ".join(map(str, value.keys()))

                raise ValueError(
                    f"Cannot extract a numeric {agent_name} score from "
                    f"dictionary keys [{available_keys}]. "
                    f"Expected one of: {', '.join(cls.SCORE_FIELDS)}. "
                    "Inspect the upstream agent output and map its "
                    "correct score field explicitly."
                )

        else:
            raise TypeError(
                f"{agent_name} evidence must be a number, a supported "
                f"result dictionary, or None; received "
                f"{type(value).__name__}."
            )

        if not math.isfinite(score):
            raise ValueError(
                f"{agent_name} evidence score must be finite; "
                f"received {score}."
            )

        # The cognitive model expects evidence scores in [0, 1].
        return clip01(score)

    def reason(
        self,
        spatial: Optional[float] = None,
        spectral: Optional[float] = None,
        feature: Optional[float] = None,
        conflict: float = 0.0,
        uncertainty: float = 0.0,
        reliabilities: Optional[Dict[str, float]] = None,
    ) -> CognitiveResult:
        """Evaluate evidence using hypothesis testing, Bayes updating,
        and metacognitive monitoring.
        """

        # Convert supported upstream outputs to numeric evidence.
        spatial_score = self._extract_score(spatial, "spatial")
        spectral_score = self._extract_score(spectral, "spectral")
        feature_score = self._extract_score(feature, "feature")

        # Conflict and uncertainty must also be scalar values.
        conflict_score = self._extract_score(conflict, "conflict")
        uncertainty_score = self._extract_score(
            uncertainty, "uncertainty"
        )

        # Defaults for the optional scalar inputs.
        conflict_score = (
            0.0 if conflict_score is None else conflict_score
        )
        uncertainty_score = (
            0.0 if uncertainty_score is None else uncertainty_score
        )

        default_rel = clip01(1.0 - uncertainty_score)
        rel = reliabilities or {}

        evidence_scores = (
            ("spatial", spatial_score),
            ("spectral", spectral_score),
            ("feature", feature_score),
        )

        ev = [
            Evidence(
                name,
                score,
                clip01(float(rel.get(name, default_rel))),
            )
            for name, score in evidence_scores
            if score is not None
        ]

        # No evidence: do not invent a camouflage detection.
        if not ev:
            uniform = {
                hypothesis: 1.0 / len(HYPS)
                for hypothesis in HYPS
            }

            return CognitiveResult(
                decision="UNCERTAIN - REVIEW",
                hypothesis=None,
                camouflage_detected=False,
                suspicion_score=round(uniform["H0"], 4),
                posterior=uniform,
                raw_posterior=uniform,
                flags=["NO_EVIDENCE"],
                explanation=[
                    "No agent reported usable evidence for this candidate."
                ],
            )

        # Stage 1: hypothesis testing.
        test = hypothesis_test(
            ev,
            self.models,
            self.alpha,
            self.bf,
        )

        # Stage 2: Bayesian belief updating.
        raw, trace = bayesian_update(
            ev,
            self.models,
            self.prior,
        )

        # Stage 3: metacognitive monitoring.
        adj, force_review, flags, notes = metacognitive_monitor(
            raw,
            trace,
            ev,
            conflict_score,
            uncertainty_score,
            test,
        )

        # Select the highest-posterior hypothesis.
        top = max(HYPS, key=lambda hypothesis: adj[hypothesis])

        if force_review or adj[top] < self.accept:
            decision = "UNCERTAIN - REVIEW"
            hypothesis = None
        else:
            decision = NAMES[top]
            hypothesis = top

        explanation = [
            (
                f"Hypothesis test: best={test['best']}, "
                f"BF over {test['runner_up']} = "
                f"{math.exp(min(test['log_bf_best_vs_second'], 50)):.1f} "
                f"({'decisive' if test['supported'] else 'not decisive'}); "
                f"consistent with {test['consistent'] or 'none'}."
            ),
            "Bayesian posterior: "
            + ", ".join(
                f"P({h})={raw[h]:.3f}" for h in HYPS
            )
            + ".",
            f"Metacognition flags: {flags or 'none'}.",
        ] + notes

        return CognitiveResult(
            decision=decision,
            hypothesis=hypothesis,
            camouflage_detected=(hypothesis == "H0"),
            suspicion_score=round(adj["H0"], 4),
            posterior={
                h: round(v, 4) for h, v in adj.items()
            },
            raw_posterior={
                h: round(v, 4) for h, v in raw.items()
            },
            test=test,
            belief_trace=trace,
            flags=flags,
            explanation=explanation,
        )