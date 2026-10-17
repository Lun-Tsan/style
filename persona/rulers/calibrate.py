"""Out-of-fold calibration on contiguous date blocks: per-bag-size thresholds and isotonic LLR calibration.

Calibrator maps the NB target-LLR (-margin) of a single turn to calibrated log-odds of 'written by T'
(PIPELINE_SPEC §3.4). It is fitted on OOF training scores only."""
from __future__ import annotations

import numpy as np

from .. import config as C
from .stats import Isotonic, brier, cllr, tune_threshold


class Calibrator:
    def __init__(self):
        self.iso = Isotonic()
        self.fit_stats = {}

    def fit(self, llr_T, y_T):
        """llr_T: NB target-LLR per turn (OOF); y_T: 1 if written by T."""
        llr_T = np.asarray(llr_T, float)
        y_T = np.asarray(y_T, int)
        self.iso.fit(llr_T, y_T)
        lo = self.iso.logodds(llr_T)
        self.fit_stats = {"n": int(len(y_T)), "cllr_raw": cllr(llr_T, y_T), "cllr_cal": cllr(lo, y_T),
                          "brier_raw_sigmoid": brier(1 / (1 + np.exp(-llr_T)), y_T),
                          "brier_cal": brier(self.iso.prob(llr_T), y_T)}
        return self

    def logodds(self, llr_T):
        return self.iso.logodds(np.asarray(llr_T, float))

    def prob(self, llr_T):
        return self.iso.prob(np.asarray(llr_T, float))

    def evaluate(self, llr_T, y_T):
        llr_T = np.asarray(llr_T, float)
        y_T = np.asarray(y_T, int)
        return {"n": int(len(y_T)), "cllr_raw": cllr(llr_T, y_T), "cllr_cal": cllr(self.logodds(llr_T), y_T),
                "brier_raw_sigmoid": brier(1 / (1 + np.exp(-llr_T)), y_T), "brier_cal": brier(self.prob(llr_T), y_T)}


def per_k_threshold(scores_pos, scores_neg):
    """Threshold on partner-likeness bag scores from OOF training bags (positive = S bags)."""
    s = np.array(list(scores_pos) + list(scores_neg), float)
    y = np.array([1] * len(scores_pos) + [0] * len(scores_neg))
    thr, _, _ = tune_threshold(s, y)
    return thr
