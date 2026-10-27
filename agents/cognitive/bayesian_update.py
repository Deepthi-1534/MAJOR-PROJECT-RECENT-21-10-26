"""bayesian_update.py - Stage 2: sequential categorical belief updating over H0/H1/H2."""
import math
from typing import Dict, List, Optional
from .common import HYPS, EPS, Evidence, clip01, log_norm_pdf, normalize_log, presence_loglik, PRESENCE_KEY


def bayesian_update(evidence: List[Evidence], models,
                    prior: Optional[Dict[str, float]] = None):
    """
    log P(Hk | x1..xi) = log P(Hk) + sum_j w_j * log p(x_j | Hk), then normalise.
    w_j = agent reliability, so an unreliable agent moves the belief less.
    Most informative agent first. Returns (posterior, trace).
    """
    prior = prior or {h: 1.0 / len(HYPS) for h in HYPS}
    logp = {h: math.log(prior[h] + EPS) for h in HYPS}
    belief = normalize_log(logp)
    trace = [{"step": "prior", "agent": "-", "weight": 1.0, "spread": 0.0,
              "belief": {h: round(v, 4) for h, v in belief.items()}}]

    # Which agents reported at all (only with calibrated models; otherwise adds nothing).
    if models.get(PRESENCE_KEY):
        pres = presence_loglik(models, [e.name for e in evidence])
        for h in HYPS:
            logp[h] += pres[h]
        belief = normalize_log(logp)
        trace.append({"step": "presence", "agent": "all", "weight": 1.0, "spread": 0.0,
                      "belief": {h: round(v, 4) for h, v in belief.items()}})

    items = []
    for e in evidence:
        ll = {h: log_norm_pdf(e.score, models[e.name][h].mu, models[e.name][h].sd) for h in HYPS}
        spread = (max(ll.values()) - min(ll.values())) * clip01(e.reliability)
        items.append((spread, e, ll))
    items.sort(key=lambda t: t[0], reverse=True)

    for spread, e, ll in items:
        w = clip01(e.reliability)
        for h in HYPS:
            logp[h] += w * ll[h]
        belief = normalize_log(logp)
        trace.append({"step": "update", "agent": e.name, "score": round(e.score, 3),
                      "weight": round(w, 3), "spread": round(spread, 3),
                      "belief": {h: round(v, 4) for h, v in belief.items()}})
    return belief, trace