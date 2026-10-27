"""calibration.py - fit p(score | Hk) from labelled data (use in evaluate.py)."""
import math
from typing import Dict, List
from .common import HYPS, Gauss, DEFAULT_MODELS


def fit_models(samples: Dict[str, List[float]], labels: Dict[str, List[str]]):
    """
    samples[agent] = list of scores
    labels[agent]  = list of "H0" / "H1" / "H2" ground-truth labels (same length)
    Falls back to DEFAULT_MODELS when a hypothesis has fewer than 2 samples.
    """
    fitted = {}
    for agent, xs in samples.items():
        fitted[agent] = {}
        for h in HYPS:
            v = [x for x, y in zip(xs, labels[agent]) if y == h]
            if len(v) < 2:
                fitted[agent][h] = DEFAULT_MODELS.get(agent, DEFAULT_MODELS["spatial"])[h]
                continue
            mu = sum(v) / len(v)
            sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1))
            fitted[agent][h] = Gauss(mu, max(sd, 0.05))
    return fitted


def fit_from_candidates(rows, min_sd: float = 0.03, min_samples: int = 5, p_floor: float = 0.02):
    """
    Fit the cognitive models from labelled fusion candidates (rows of data/fusion/*.csv, built
    from the VAL split - never the test split).

      row["label"]            1 = the candidate is a real camouflaged target, 0 = it is not
      row["<a>_present"]      1 if agent <a> reported evidence for the candidate
      row["<a>_evidence"]     the agent's evidence score

    H0 (camouflaged)  <- label 1.
    H1 / H2           <- label 0. The dataset contains no separately labelled "ordinary object", so
                         both non-camouflaged hypotheses are fitted from the same negatives.
    Also fits PRESENCE likelihoods P(agent reports | Hk) with add-one smoothing.
    A (agent, hypothesis) pair with fewer than `min_samples` scores keeps the default model.
    """
    from .common import PRESENCE_KEY, AGENT_NAMES
    groups = {"H0": [r for r in rows if int(float(r["label"])) == 1],
              "H1": [r for r in rows if int(float(r["label"])) == 0],
              "H2": [r for r in rows if int(float(r["label"])) == 0]}
    models, presence = {}, {}
    for agent in AGENT_NAMES:
        models[agent], presence[agent] = {}, {}
        for h in HYPS:
            rs = groups[h]
            seen = [float(r[f"{agent}_evidence"]) for r in rs if float(r[f"{agent}_present"]) > 0]
            p = (len(seen) + 1.0) / (len(rs) + 2.0)
            presence[agent][h] = min(max(p, p_floor), 1.0 - p_floor)
            if len(seen) < min_samples:
                models[agent][h] = DEFAULT_MODELS[agent][h]
                continue
            mu = sum(seen) / len(seen)
            sd = math.sqrt(sum((x - mu) ** 2 for x in seen) / (len(seen) - 1))
            models[agent][h] = Gauss(mu, max(sd, min_sd))
    models[PRESENCE_KEY] = presence
    return models