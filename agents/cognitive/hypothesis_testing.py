"""hypothesis_testing.py - Stage 1: compare H0 / H1 / H2 using likelihoods and Bayes factors."""
import math
from typing import List
from .common import HYPS, Evidence, log_norm_pdf, chi2_sf, presence_loglik


def hypothesis_test(evidence: List[Evidence], models, alpha: float = 0.05,
                    bf_threshold: float = 3.0) -> dict:
    """
    - log-likelihood of the evidence under each hypothesis
    - pairwise log Bayes factors
    - best hypothesis is 'supported' only if BF over the runner-up >= bf_threshold
    - goodness-of-fit chi-square per hypothesis (p < alpha => evidence inconsistent with it)
    """
    loglik = presence_loglik(models, [e.name for e in evidence])   # zeros unless calibrated
    chi2 = {h: 0.0 for h in HYPS}
    for e in evidence:
        for h in HYPS:
            g = models[e.name][h]
            loglik[h] += log_norm_pdf(e.score, g.mu, g.sd)
            chi2[h] += ((e.score - g.mu) / max(g.sd, 1e-3)) ** 2

    k = len(evidence)
    gof_p = {h: chi2_sf(chi2[h], k) for h in HYPS}
    ranked = sorted(HYPS, key=lambda h: loglik[h], reverse=True)
    best, second = ranked[0], ranked[1]
    log_bf = loglik[best] - loglik[second]

    return {
        "loglik": {h: round(v, 3) for h, v in loglik.items()},
        "log_bayes_factor_pairwise": {
            f"{a} vs {b}": round(loglik[a] - loglik[b], 3)
            for a, b in (("H0", "H1"), ("H0", "H2"), ("H1", "H2"))},
        "best": best,
        "runner_up": second,
        "log_bf_best_vs_second": log_bf,
        "supported": log_bf >= math.log(bf_threshold),
        "gof_p": {h: round(v, 4) for h, v in gof_p.items()},
        "consistent": [h for h in HYPS if gof_p[h] >= alpha],
        "alpha": alpha,
    }