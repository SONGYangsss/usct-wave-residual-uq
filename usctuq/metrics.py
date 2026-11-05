"""Phantom-level conformal calibration and the evaluation endpoints (numpy only).

Errors and widths are (cases, pixels) arrays over the breast mask, in normalised units (x 100 = m/s).
Calibration: per case the level-quantile of |e| / u, then the finite-sample level-quantile of those
case scores over the calibration cases. The interval is c +- q u.
"""
import numpy as np

ALPHA, LEVEL, FLOOR, C_SCALE = 0.10, 0.90, 1e-9, 100.0
ECE_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
SEED = 20261010


def finite_quantile(v, level):
    return float(np.quantile(v, min(1.0, np.ceil((len(v) + 1) * level) / len(v))))


def calibrate(err, u, level=LEVEL):
    return finite_quantile(np.quantile(err / np.maximum(u, FLOOR), level, axis=1), level)


def coverage(err, u, q):
    """Per-case fraction of pixels inside the interval."""
    return np.mean(err <= q * np.maximum(u, FLOOR), axis=1)


def interval_score(err, hw):
    """Per-case 90% interval score in m/s."""
    return np.mean(2.0 * hw + (2.0 / ALPHA) * np.maximum(err - hw, 0.0), axis=1) * C_SCALE


def evaluate(cal, test_id, ood, level=LEVEL):
    """cal/test_id/ood = (err, u). Fixed-split endpoints, with the per-case OOD arrays kept."""
    q = calibrate(*cal, level)
    ci, co = coverage(*test_id, q), coverage(*ood, q)
    hw = q * ood[1]
    return {"q": q, "cov_id": ci.mean(), "cov_ood": co.mean(), "succ_id": (ci >= level).mean(),
            "succ_ood": (co >= level).mean(), "n_succ_ood": int((co >= level).sum()),
            "hw_ood": np.median(hw) * C_SCALE, "hw_id": np.median(q * test_id[1]) * C_SCALE,
            "is_ood": interval_score(ood[0], hw).mean(), "_cov_id": ci, "_cov_ood": co,
            "_is_ood": interval_score(ood[0], hw)}


def ece(cal, ood, levels=ECE_LEVELS):
    """OOD ECE: mean |mean per-case coverage - level| over nominal levels, recalibrating at each level."""
    return float(np.mean([abs(coverage(*ood, calibrate(*cal, l)).mean() - l) for l in levels]))


def resample(pool, ood, n_cal=40, draws=1000, seed=SEED, mode="permutation"):
    """Calibration-set resampling: q from 40 pool cases, OOD endpoints per draw.

    mode='permutation': 1000 permutations of the pool, first 40 calibrate (80-case pool, all methods).
    mode='choice':      40 cases drawn without replacement per draw (159-case pool, latent scales).
    """
    rng = np.random.default_rng(seed)
    out = {"q": [], "cov_ood": [], "succ_ood": [], "is_ood": [], "hw_ood": []}
    for _ in range(draws):
        a = rng.permutation(len(pool[0]))[:n_cal] if mode == "permutation" else rng.choice(len(pool[0]), n_cal, replace=False)
        q = calibrate(pool[0][a], pool[1][a])
        co = coverage(*ood, q)
        hw = q * np.maximum(ood[1], FLOOR)
        out["q"].append(q); out["cov_ood"].append(co.mean()); out["succ_ood"].append((co >= LEVEL).mean())
        out["is_ood"].append(interval_score(ood[0], hw).mean()); out["hw_ood"].append(np.median(hw) * C_SCALE)
    return {k: np.array(v) for k, v in out.items()}


def iqr(v):
    v = np.asarray(v, float)
    return [float(np.median(v)), float(np.quantile(v, 0.25)), float(np.quantile(v, 0.75))]


def ranks(x):
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x)); r[o] = np.arange(len(x))
    return r


def spearman(x, y):
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def auroc(score, label):
    s1, s0 = score[label], score[~label]
    return float((s1[:, None] > s0[None, :]).mean() + 0.5 * (s1[:, None] == s0[None, :]).mean())


def worst_quarter(rmse):
    return rmse >= np.quantile(rmse, 0.75)


def bootstrap(n, draws=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(draws)]


def ci95(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def partial_spearman(a_phys, a_res, rmse, cal, sel):
    """Rank association of a_phys with case RMSE beyond a_res: Spearman of the residuals of log-log
    fits on a_res (fitted on the calibration cases), evaluated on the cases in sel."""
    la, lr, lm = np.log(a_phys), np.log(a_res), np.log(rmse)
    b, a = np.polyfit(lr[cal], la[cal], 1)
    bm, am = np.polyfit(lr[cal], lm[cal], 1)
    return spearman((la - (a + b * lr))[sel], (lm - (am + bm * lr))[sel])
