"""Reproduce the tables and the numbers quoted in the paper from the run files (numpy only).

    python scripts/evaluate.py --runs <data>/runs --out results

Run files are written by reconstruct.py (recon_*.npz) and run_baselines.py (baselines_*.npz).
Each section prints the rows of one table or paragraph and stores them in results/<section>.json.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from usctuq import metrics as M  # noqa: E402

C = M.C_SCALE
S3 = ("calib", "test_id", "ood")


class Recon:
    """A reconstruction run restricted to the mask 'training std > thr' (0.02 = the breast mask)."""

    def __init__(self, path, thr=0.02):
        z = np.load(path)
        m = z["std_pix"] > thr
        self.splits = [s for s in ("calib", "test_id", "ood", "train") if f"cr_{s}" in z.files]
        self.w = z["w"][m].astype(float)
        self.w /= np.median(self.w)
        self.err = {s: np.abs(z[f"cr_{s}"] - z[f"cg_{s}"])[:, m].astype(float) for s in self.splits}
        self.u = {s: z[f"u_{s}"][:, m].astype(float) for s in self.splits}
        self.scale = {"a_res": {s: z[f"ares_{s}"].astype(float) for s in self.splits},
                      "a_phys": {s: np.median(z[f"u_{s}"][:, m], 1).astype(float) for s in self.splits}}
        opt = lambda k: {s: z[f"{k}_{s}"].astype(float) for s in self.splits if f"{k}_{s}" in z.files}
        self.efwd, self.dens, self.wcase = opt("efwd"), opt("dens"), opt("wcase")
        self.wcase = {s: (v[:, m] / np.median(v[:, m], 1, keepdims=True)) for s, v in self.wcase.items()}

    def pair(self, kind, s):
        """(err, width) of an interval construction on split s."""
        e, n = self.err[s], len(self.err[s])
        if kind == "raw_map":
            return e, self.u[s]
        if kind == "constant":
            return e, np.ones_like(e)
        if kind == "w_only":
            return e, np.broadcast_to(self.w, e.shape)
        a = self.scale[kind.replace("_only", "").replace("_w", "")][s][:, None]      # a_res_w, a_phys_only, ...
        return e, (a * self.w[None] if kind.endswith("_w") else np.broadcast_to(a, (n, len(self.w))))

    def rmse(self, s):
        return np.sqrt((self.err[s] ** 2).mean(1)) * C


class Baselines:
    def __init__(self, path):
        self.z = np.load(path)
        self.methods = [m for m in ("ensemble", "dropout", "diffusion") if f"{m}_rmse_ood" in self.z.files]

    def pair(self, m, s):
        return self.z[f"{m}_err_{s}"].astype(float), self.z[f"{m}_u_{s}"].astype(float)

    def case(self, m, key, splits=("test_id", "ood")):
        return np.concatenate([self.z[f"{m}_{key}_{s}"] for s in splits]).astype(float)


def fixed(pairs):
    return M.evaluate(pairs["calib"], pairs["test_id"], pairs["ood"])


def pool80(pairs):
    return tuple(np.concatenate([pairs["calib"][i], pairs["test_id"][i]]) for i in (0, 1))


def pool159(R, kind):
    return tuple(np.concatenate([R.pair(kind, s)[i] for s in ("calib", "train", "test_id")]) for i in (0, 1))


def clean(d):
    if isinstance(d, dict):
        return {k: clean(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, (list, tuple)):
        return [clean(v) for v in d]
    return float(d) if isinstance(d, (np.floating, np.integer)) else d


def triage_scores(R, B):
    """Case-level uncertainty and own case RMSE over the 81 test phantoms."""
    out = {k: (np.r_[R.scale[k]["test_id"], R.scale[k]["ood"]], np.r_[R.rmse("test_id"), R.rmse("ood")])
           for k in ("a_res", "a_phys")}
    for m in B.methods:
        out[m] = (B.case(m, "spread"), B.case(m, "rmse"))
    return out


GRP = np.r_[np.zeros(40, bool), np.ones(41, bool)]          # False = ID test, True = OOD


# ====================================================================== sections
def main_table(R, B):
    """Table: unified pipeline comparison (fixed split, 80-pool draws, triage AUROC)."""
    pairs = {m: {s: B.pair(m, s) for s in S3} for m in B.methods}
    pairs.update({k: {s: R.pair(k, s) for s in S3} for k in ("raw_map", "a_res_w", "a_phys_w")})
    tri = triage_scores(R, B)
    out = {}
    for name, p in pairs.items():
        f = fixed(p)
        dr = M.resample(pool80(p), p["ood"], mode="permutation")
        key = {"a_res_w": "a_res", "a_phys_w": "a_phys"}.get(name, name)
        auc = M.auroc(tri[key][0], M.worst_quarter(tri[key][1])) if key in tri else None
        rm = lambda s: float(np.sqrt((p[s][0] ** 2).mean()) * C)
        out[name] = {**clean(f), "rmse_id": rm("test_id"), "rmse_ood": rm("ood"), "pece_ood": M.ece(p["calib"], p["ood"]),
                     "draws_cov_ood": M.iqr(dr["cov_ood"]), "draws_is_ood": float(np.median(dr["is_ood"])),
                     "triage_auroc": auc, "_is_draws": dr["is_ood"], "_cov_draws": dr["cov_ood"]}
        o = out[name]
        print(f"  {name:10s} RMSE {o['rmse_id']:.1f}->{o['rmse_ood']:.1f}  cov {o['cov_id']:.3f}->{o['cov_ood']:.3f}  "
              f"succ {o['succ_id']:.3f}->{o['succ_ood']:.3f}  HW {o['hw_ood']:.0f}  IS {o['is_ood']:.0f}  P-ECE {o['pece_ood']:.3f}  "
              f"draws {o['draws_cov_ood'][0]:.3f} [{o['draws_cov_ood'][1]:.3f},{o['draws_cov_ood'][2]:.3f}]"
              + (f"  AUROC {auc:.2f}" if auc is not None else ""))
    for k in ("a_res_w", "a_phys_w"):
        out[k]["frac_draws_ensemble_lower_is"] = float(np.mean(out["ensemble"]["_is_draws"] < out[k]["_is_draws"]))
    return clean(out)


def scales(R, B):
    """Residual norm vs back-projected scale: fixed split, 159-pool draws, bootstrap, ranking, matched width."""
    out = {}
    draws = {}
    for k in ("a_res_w", "a_phys_w"):
        f = fixed({s: R.pair(k, s) for s in S3})
        d = M.resample(pool159(R, k), R.pair(k, "ood"), mode="choice")
        draws[k] = d
        out[k] = {**clean(f), "draws_cov_ood": M.iqr(d["cov_ood"]), "draws_succ_ood": float(np.median(d["succ_ood"])),
                  "draws_is_ood": float(np.median(d["is_ood"])), "cv_q": float(np.std(d["q"]) / np.mean(d["q"])),
                  "fixed_percentile": float(np.mean(d["cov_ood"] <= f["cov_ood"]))}
        print(f"  {k:9s} fixed cov {f['cov_id']:.3f}->{f['cov_ood']:.3f} succ {f['succ_id']:.3f}->{f['succ_ood']:.3f} "
              f"({f['n_succ_ood']}/41) HW {f['hw_ood']:.0f} IS {f['is_ood']:.0f} | 159-pool cov {out[k]['draws_cov_ood'][0]:.3f} "
              f"[{out[k]['draws_cov_ood'][1]:.3f},{out[k]['draws_cov_ood'][2]:.3f}] succ {out[k]['draws_succ_ood']:.3f} "
              f"CV(q) {out[k]['cv_q']:.3f}  fixed split at percentile {out[k]['fixed_percentile']:.2f}")
    out["frac_draws_res_cov_above_phys"] = float(np.mean(draws["a_res_w"]["cov_ood"] > draws["a_phys_w"]["cov_ood"]))
    print(f"  a_res w above a_phys w in {out['frac_draws_res_cov_above_phys']:.1%} of draws")

    # ranking: Spearman with case RMSE, partial Spearman of a_phys given a_res
    ar = np.concatenate([R.scale["a_res"][s] for s in S3]); ap = np.concatenate([R.scale["a_phys"][s] for s in S3])
    rm = np.concatenate([R.rmse(s) for s in S3]); cal = np.r_[np.ones(40, bool), np.zeros(81, bool)]
    test = ~cal; ood = np.r_[np.zeros(80, bool), np.ones(41, bool)]
    for k, a in (("a_res", ar), ("a_phys", ap)):
        out[f"spearman_{k}"] = [M.spearman(a[test], rm[test]), M.spearman(a[test][~GRP], rm[test][~GRP]),
                                M.spearman(a[ood], rm[ood])]
    out["partial_spearman_test81_ood41"] = [M.partial_spearman(ap, ar, rm, cal, test), M.partial_spearman(ap, ar, rm, cal, ood)]
    print("  Spearman with case RMSE (all 81 / ID / OOD): a_res {:.3f} {:.3f} {:.3f}  a_phys {:.3f} {:.3f} {:.3f}".format(
        *out["spearman_a_res"], *out["spearman_a_phys"]))
    print("  partial Spearman of a_phys given a_res: test81 {:+.2f}  OOD {:+.2f}".format(*out["partial_spearman_test81_ood41"]))

    # phantom bootstrap over the 41 OOD cases, calibration fixed
    per = {k: fixed({s: R.pair(k, s) for s in S3}) for k in ("a_res_w", "a_phys_w")}
    per["ensemble"] = fixed({s: B.pair("ensemble", s) for s in S3})
    cv = {k: v["_cov_ood"] for k, v in per.items()}; isc = {k: v["_is_ood"] for k, v in per.items()}
    bs = M.bootstrap(41)
    boot = lambda f: [float(f(np.arange(41)))] + M.ci95([f(b) for b in bs])
    succ = lambda c: (c >= M.LEVEL)
    out["bootstrap"] = {
        "a_res_w_cov": boot(lambda b: cv["a_res_w"][b].mean()), "a_res_w_succ": boot(lambda b: succ(cv["a_res_w"][b]).mean()),
        "res_minus_ens_cov": boot(lambda b: cv["a_res_w"][b].mean() - cv["ensemble"][b].mean()),
        "res_minus_ens_succ": boot(lambda b: succ(cv["a_res_w"][b]).mean() - succ(cv["ensemble"][b]).mean()),
        "res_minus_ens_is": boot(lambda b: isc["a_res_w"][b].mean() - isc["ensemble"][b].mean()),
        "phys_minus_res_cov": boot(lambda b: cv["a_phys_w"][b].mean() - cv["a_res_w"][b].mean()),
        "phys_minus_res_succ": boot(lambda b: succ(cv["a_phys_w"][b]).mean() - succ(cv["a_res_w"][b]).mean()),
        "phys_minus_res_is": boot(lambda b: isc["a_phys_w"][b].mean() - isc["a_res_w"][b].mean())}
    for k, v in out["bootstrap"].items():
        print(f"  bootstrap {k:20s} {v[0]:+.3f}  CI [{v[1]:+.3f}, {v[2]:+.3f}]")

    # coverage-width frontier: scale each calibrated q by kappa, compare at a matched median half-width
    kap = np.exp(np.linspace(np.log(0.3), np.log(3.0), 121))
    frontier = {}
    for k, (e, u) in (("a_res_w", R.pair("a_res_w", "ood")), ("a_phys_w", R.pair("a_phys_w", "ood")),
                      ("ensemble", B.pair("ensemble", "ood"))):
        q0 = per[k]["q"]
        hw = np.array([np.median(x * q0 * u) * C for x in kap]); cov = np.array([np.mean(e <= x * q0 * u, 1).mean() for x in kap])
        frontier[k] = (hw, cov)
    at = per["a_phys_w"]["hw_ood"]
    out["matched_width"] = {"hw": float(at), **{k: float(np.interp(at, h[np.argsort(h)], c[np.argsort(h)]))
                                                 for k, (h, c) in frontier.items()}}
    print("  at matched median half-width {hw:.0f} m/s: OOD coverage a_res w {a_res_w:.3f}  a_phys w {a_phys_w:.3f}  "
          "ensemble {ensemble:.3f}".format(**out["matched_width"]))
    levels = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
    out["reliability_ood"] = {"levels": list(levels)}
    for k in ("a_res_w", "a_phys_w"):
        out["reliability_ood"][k] = [M.evaluate(R.pair(k, "calib"), R.pair(k, "test_id"), R.pair(k, "ood"), l)["cov_ood"] for l in levels]
    out["reliability_ood"]["ensemble"] = [M.evaluate(B.pair("ensemble", "calib"), B.pair("ensemble", "test_id"),
                                                     B.pair("ensemble", "ood"), l)["cov_ood"] for l in levels]
    return clean(out)


def ablation(R):
    """Interval constructions on the common latent reconstruction (fixed split and 159-pool draws)."""
    out = {}
    for k in ("constant", "w_only", "a_res_only", "a_res_w", "a_phys_only", "a_phys_w", "raw_map"):
        f = fixed({s: R.pair(k, s) for s in S3})
        d = M.resample(pool159(R, k), R.pair(k, "ood"), mode="choice")
        e, u = R.pair(k, "ood")
        within = [M.spearman(u[i], e[i]) for i in range(len(e)) if np.std(u[i]) > 1e-12]
        out[k] = {**clean(f), "draws_cov_ood": M.iqr(d["cov_ood"]), "draws_is_ood": float(np.median(d["is_ood"])),
                  "within_case_spearman_ood": float(np.median(within)) if within else None,
                  "across_case_spearman_ood": M.spearman(u.mean(1), e.mean(1)) if np.std(u.mean(1)) > 1e-12 else None}
        o = out[k]
        print(f"  {k:11s} fixed cov {o['cov_id']:.3f}->{o['cov_ood']:.3f} succ {o['n_succ_ood']}/41 IS {o['is_ood']:.0f} "
              f"HW {o['hw_ood']:.0f} | draws cov {o['draws_cov_ood'][0]:.3f} IS {o['draws_is_ood']:.0f} | "
              f"OOD ranking within {o['within_case_spearman_ood']}  across {o['across_case_spearman_ood']}")
    return clean(out)


def triage(R, B, same=None):
    """Case triage: Spearman and worst-quarter AUROC, paired bootstrap against the residual norm."""
    tri = triage_scores(R, B)
    out = {}
    bs = M.bootstrap(81)
    ar, rr = tri["a_res"]; lr = M.worst_quarter(rr)
    for k, (a, r) in tri.items():
        lab = M.worst_quarter(r)
        o = {"spearman": [M.spearman(a, r), M.spearman(a[~GRP], r[~GRP]), M.spearman(a[GRP], r[GRP])],
             "auroc": M.auroc(a, lab), "worst_quarter_n_ood": int(lab[GRP].sum())}
        if k == "a_res":
            ok = [b for b in bs if not (lab[b].all() or (~lab[b]).all())]
            o["auroc_ci"] = M.ci95([M.auroc(a[b], lab[b]) for b in ok])
        else:
            ok = [b for b in bs if not (lr[b].all() or (~lr[b]).all() or lab[b].all() or (~lab[b]).all())]
            o["res_minus_this_auroc_ci"] = M.ci95([M.auroc(ar[b], lr[b]) - M.auroc(a[b], lab[b]) for b in ok])
        out[k] = o
        print(f"  {k:9s} Spearman {o['spearman'][0]:+.3f} (ID {o['spearman'][1]:+.3f}, OOD {o['spearman'][2]:+.3f})  "
              f"AUROC {o['auroc']:.3f}  " + (f"CI {np.round(o['auroc_ci'], 3)}" if "auroc_ci" in o
                                             else f"a_res minus this CI {np.round(o['res_minus_this_auroc_ci'], 2)}"))

    # silent-failure signature: low residual but failed interval
    f = fixed({s: R.pair("a_res_w", s) for s in S3})
    cov81 = np.r_[f["_cov_id"], f["_cov_ood"]]
    low = ar < np.median(R.scale["a_res"]["calib"])
    out["low_residual_cases"] = {"n": int(low.sum()), "n_failed": int((cov81[low] < M.LEVEL).sum()),
                                 "min_coverage": float(cov81[low].min()), "n_ood_above_median": int((~low[GRP]).sum())}
    print("  low-residual cases (a_res below calibration median): {n}, failed {n_failed}, min coverage {min_coverage:.3f}, "
          "OOD above median {n_ood_above_median}/41".format(**out["low_residual_cases"]))

    if same is not None:                                    # each baseline's own reconstruction
        out["same_reconstruction"] = {}
        for m in same.methods:
            r, s, a = same.case(m, "rmse"), same.case(m, "spread"), same.case(m, "ares")
            lab = M.worst_quarter(r)
            ok = [b for b in bs if not (lab[b].all() or (~lab[b]).all())]
            o = {"auroc_spread": M.auroc(s, lab), "auroc_a_res": M.auroc(a, lab),
                 "diff_ci": M.ci95([M.auroc(a[b], lab[b]) - M.auroc(s[b], lab[b]) for b in ok]),
                 "spearman_id": [M.spearman(s[~GRP], r[~GRP]), M.spearman(a[~GRP], r[~GRP])],
                 "spearman_ood": [M.spearman(s[GRP], r[GRP]), M.spearman(a[GRP], r[GRP])],
                 "median_a_res_id": float(np.median(a[~GRP]))}
            out["same_reconstruction"][m] = o
            print(f"  same recon {m:9s} AUROC spread {o['auroc_spread']:.3f} a_res {o['auroc_a_res']:.3f} CI {np.round(o['diff_ci'], 2)}"
                  f" | Spearman ID {o['spearman_id'][0]:+.3f}/{o['spearman_id'][1]:+.3f} OOD {o['spearman_ood'][0]:+.3f}/"
                  f"{o['spearman_ood'][1]:+.3f} | median ID a_res {o['median_a_res_id']:.2f}")
    return clean(out)


def density_and_forward_error(R):
    """OOD-cutoff sweep, and forward error versus density (overlap band of e_fwd)."""
    out = {"cutoff": {}}
    dens = np.r_[R.dens["test_id"], R.dens["ood"]]
    for k in ("a_res_w", "a_phys_w"):
        f = fixed({s: R.pair(k, s) for s in S3}); cov = np.r_[f["_cov_id"], f["_cov_ood"]]
        out["cutoff"][k] = {}
        for qq in (0.60, 0.70, 0.80, 0.90):
            t = float(np.quantile(np.r_[R.dens["calib"], dens], qq)); sel = dens >= t
            out["cutoff"][k][str(qq)] = {"threshold": t, "n": int(sel.sum()), "cov": float(cov[sel].mean())}
        print(f"  {k:9s} coverage above density quantile .6/.7/.8/.9: " +
              " ".join(f"{v['cov']:.3f} (n={v['n']})" for v in out["cutoff"][k].values()))
    ef = np.r_[R.efwd["test_id"], R.efwd["ood"]]
    mae = np.r_[R.err["test_id"].mean(1), R.err["ood"].mean(1)] * C
    out["spearman"] = {"efwd_density": M.spearman(ef, dens), "mae_density": M.spearman(mae, dens), "mae_efwd": M.spearman(mae, ef)}
    lo, hi = R.efwd["ood"].min(), R.efwd["test_id"].max(); band = (ef >= lo) & (ef <= hi)
    out["overlap_band"] = {"band": [float(lo), float(hi)], "n_id": int((band & ~GRP).sum()), "n_ood": int((band & GRP).sum()),
                           "efwd_mean": [float(ef[band & ~GRP].mean()), float(ef[band & GRP].mean())],
                           "mae": [float(mae[band & ~GRP].mean()), float(mae[band & GRP].mean())]}
    for k in ("a_res_w", "a_phys_w"):
        f = fixed({s: R.pair(k, s) for s in S3}); cov = np.r_[f["_cov_id"], f["_cov_ood"]]
        out["overlap_band"][f"cov_{k}"] = [float(cov[band & ~GRP].mean()), float(cov[band & GRP].mean())]
    print("  Spearman: e_fwd~density {efwd_density:.2f}  MAE~density {mae_density:.2f}  MAE~e_fwd {mae_efwd:.2f}".format(**out["spearman"]))
    b = out["overlap_band"]
    print(f"  overlap band [{b['band'][0]:.3f},{b['band'][1]:.3f}] ID n={b['n_id']} OOD n={b['n_ood']}  e_fwd {b['efwd_mean'][0]:.3f}/"
          f"{b['efwd_mean'][1]:.3f}  MAE {b['mae'][0]:.1f}/{b['mae'][1]:.1f}  cov a_res w {b['cov_a_res_w'][0]:.3f}/"
          f"{b['cov_a_res_w'][1]:.3f}  a_phys w {b['cov_a_phys_w'][0]:.3f}/{b['cov_a_phys_w'][1]:.3f}")
    return clean(out)


def controls(runs):
    """Table: main and control surrogates (80-pool draws, partial Spearman, triage, worst forward-error third)."""
    out = {}
    for path in [os.path.join(runs, "recon_main.npz")] + sorted(glob.glob(os.path.join(runs, "recon_fno_*.npz"))):
        tag = os.path.basename(path)[6:-4]
        R = Recon(path)
        rmse_ood = float(np.sqrt((R.err["ood"] ** 2).mean()) * C)
        o = {"efwd_ood_median": float(np.median(R.efwd["ood"])), "rmse_ood": rmse_ood}
        for k in ("a_res_w", "a_phys_w"):
            p = {s: R.pair(k, s) for s in S3}
            d = M.resample(pool80(p), p["ood"], mode="permutation")
            f = fixed(p); cov = np.r_[f["_cov_id"], f["_cov_ood"]]
            ef = np.r_[R.efwd["test_id"], R.efwd["ood"]]
            top = np.digitize(ef, np.quantile(ef, [1 / 3, 2 / 3])) == 2
            o[k] = {"draws_cov_ood": float(np.median(d["cov_ood"])), "draws_is_ood": float(np.median(d["is_ood"])),
                    "worst_efwd_third_cov": float(cov[top].mean())}
        ar = np.concatenate([R.scale["a_res"][s] for s in S3]); ap = np.concatenate([R.scale["a_phys"][s] for s in S3])
        rm = np.concatenate([R.rmse(s) for s in S3]); cal = np.r_[np.ones(40, bool), np.zeros(81, bool)]
        o["partial_spearman"] = M.partial_spearman(ap, ar, rm, cal, ~cal)
        o["triage_auroc_a_res"] = M.auroc(ar[~cal], M.worst_quarter(rm[~cal]))
        o["spearman_a_res_a_phys"] = [M.spearman(ar[~cal], rm[~cal]), M.spearman(ap[~cal], rm[~cal])]
        out[tag] = o
        print(f"  {tag:16s} e_fwd {o['efwd_ood_median']:.3f}  RMSE {rmse_ood:.1f}  cov {o['a_res_w']['draws_cov_ood']:.3f}/"
              f"{o['a_phys_w']['draws_cov_ood']:.3f}  IS {o['a_res_w']['draws_is_ood']:.0f}/{o['a_phys_w']['draws_is_ood']:.0f}  "
              f"partial {o['partial_spearman']:+.2f}  AUROC {o['triage_auroc_a_res']:.2f}  "
              f"worst third {o['a_res_w']['worst_efwd_third_cov']:.2f}")
    return out


def acquisition(runs, R0, B0):
    """Table: transducer gain/phase errors and noise (clean q fixed, and recalibrated 80-pool draws)."""
    out = {}
    conds = ["clean"] + [os.path.basename(p)[6:-4] for p in sorted(glob.glob(os.path.join(runs, "recon_acq_*.npz")))]
    for c in conds:
        R = R0 if c == "clean" else Recon(os.path.join(runs, f"recon_{c}.npz"))
        fb = os.path.join(runs, f"baselines_{c}.npz")
        B = B0 if c == "clean" else (Baselines(fb) if os.path.exists(fb) else None)    # 40/30/20 dB: scales only
        o = {"rmse_ood": float(np.sqrt((R.err["ood"] ** 2).mean()) * C),
             "ratio_a_res": float(np.median(R.scale["a_res"]["ood"]) /                  # OOD over the 80 ID phantoms
                                  np.median(np.r_[R.scale["a_res"]["calib"], R.scale["a_res"]["test_id"]]))}
        for k in ("a_res_w", "a_phys_w") + (("ensemble",) if B else ()):
            clean_pairs = {s: (B0.pair("ensemble", s) if k == "ensemble" else R0.pair(k, s)) for s in S3}
            pert = {s: (B.pair("ensemble", s) if k == "ensemble" else R.pair(k, s)) for s in S3}
            q = M.calibrate(*clean_pairs["calib"])
            co = M.coverage(*pert["ood"], q)
            d = M.resample(pool80(pert), pert["ood"], mode="permutation")
            o[k] = {"clean_q_cov_ood": float(co.mean()),
                    "clean_q_hw_id": float(np.median(q * pert["test_id"][1]) * C),
                    "recal_cov_ood": float(np.median(d["cov_ood"])), "recal_is_ood": float(np.median(d["is_ood"])),
                    "recal_hw_ood": float(np.median(d["hw_ood"])), "recal_succ_ood": float(np.median(d["succ_ood"])),
                    "_cov": d["cov_ood"], "_is": d["is_ood"]}
        o["frac_draws_phys_cov_above_res"] = float(np.mean(o["a_phys_w"]["_cov"] > o["a_res_w"]["_cov"]))
        o["frac_draws_phys_is_below_res"] = float(np.mean(o["a_phys_w"]["_is"] < o["a_res_w"]["_is"]))
        rm = np.r_[R.rmse("test_id"), R.rmse("ood")]
        o["triage_auroc_a_res"] = M.auroc(np.r_[R.scale["a_res"]["test_id"], R.scale["a_res"]["ood"]], M.worst_quarter(rm))
        if B:
            o["triage_auroc_ensemble"] = M.auroc(B.case("ensemble", "spread"), M.worst_quarter(B.case("ensemble", "rmse")))
        out[c] = clean(o)
        col = lambda key, fmt: " ".join(fmt.format(o[k][key]) for k in ("a_res_w", "a_phys_w", "ensemble") if k in o)
        print(f"  {c:16s} clean q {col('clean_q_cov_ood', '{:.3f}')} | recal {col('recal_cov_ood', '{:.3f}')} | "
              f"IS {col('recal_is_ood', '{:.0f}')} | ratio {o['ratio_a_res']:.2f}  RMSE {o['rmse_ood']:.1f}  "
              f"ID HW clean q {o['a_res_w']['clean_q_hw_id']:.1f}  AUROC {o['triage_auroc_a_res']:.3f}"
              + (f"/{o['triage_auroc_ensemble']:.3f}" if B else "")
              + f"  phys>res cov {o['frac_draws_phys_cov_above_res']:.2f} IS {o['frac_draws_phys_is_below_res']:.2f}")
    return out


def sensitivity(runs, R0):
    """PCA dimension x mask threshold, and a case-specific relative width."""
    out = {"pca": {}}
    for path in sorted(glob.glob(os.path.join(runs, "recon_pca*.npz"))):
        for thr in (0.005, 0.01, 0.02, 0.03, 0.04):
            R = Recon(path, thr)
            o = {"n_pix": int(len(R.w))}
            for k in ("a_res_w", "a_phys_w"):
                f = fixed({s: R.pair(k, s) for s in S3})
                o[k] = {"cov_ood": float(f["cov_ood"]), "n_succ_ood": f["n_succ_ood"]}
            out["pca"][f"{os.path.basename(path)[6:-4]}_thr{thr}"] = o
            print(f"  {os.path.basename(path)[6:-4]:7s} mask {thr:.3f} ({o['n_pix']} px)  a_res w {o['a_res_w']['cov_ood']:.3f} "
                  f"({o['a_res_w']['n_succ_ood']}/41)  a_phys w {o['a_phys_w']['cov_ood']:.3f} ({o['a_phys_w']['n_succ_ood']}/41)")
    if R0.wcase:
        out["case_w"] = {"median_corr_with_shared": float(np.median([np.corrcoef(v[i], R0.w)[0, 1] for v in R0.wcase.values()
                                                                    for i in range(len(v))]))}
        for k in ("a_res", "a_phys"):
            p = {s: (R0.err[s], R0.scale[k][s][:, None] * R0.wcase[s]) for s in S3}
            f = fixed(p)
            pool = lambda width: tuple(np.concatenate([(R0.err[s], R0.scale[k][s][:, None] * width(s))[i]
                                                       for s in ("calib", "train")]) for i in (0, 1))
            shared = M.resample(pool(lambda s: R0.w[None]), (R0.err["ood"], R0.scale[k]["ood"][:, None] * R0.w[None]),
                                mode="choice")
            case = M.resample(pool(lambda s: R0.wcase[s]), p["ood"], mode="choice")
            out["case_w"][k] = {"cov_ood": float(f["cov_ood"]), "n_succ_ood": f["n_succ_ood"], "is_ood": float(f["is_ood"]),
                                "draws119_cov_ood": [float(np.median(shared["cov_ood"])), float(np.median(case["cov_ood"]))]}
            print(f"  case-specific w with {k}: fixed OOD cov {f['cov_ood']:.3f}  success {f['n_succ_ood']}/41  "
                  f"IS {f['is_ood']:.0f} | 119-pool draws, shared -> case w: {out['case_w'][k]['draws119_cov_ood'][0]:.3f} -> "
                  f"{out['case_w'][k]['draws119_cov_ood'][1]:.3f}")
        print(f"  median correlation of case-specific and shared w: {out['case_w']['median_corr_with_shared']:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(os.environ.get("USCTUQ_DATA", "data"), "runs"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", default="", help="comma-separated sections, e.g. main_table,triage")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R, B = Recon(os.path.join(a.runs, "recon_main.npz")), Baselines(os.path.join(a.runs, "baselines_main.npz"))
    same = os.path.join(a.runs, "baselines_same_recon.npz")
    sections = [("main_table", lambda: main_table(R, B)), ("scales", lambda: scales(R, B)), ("ablation", lambda: ablation(R)),
                ("triage", lambda: triage(R, B, Baselines(same) if os.path.exists(same) else None)),
                ("density_forward_error", lambda: density_and_forward_error(R)), ("controls", lambda: controls(a.runs)),
                ("acquisition", lambda: acquisition(a.runs, R, B)), ("sensitivity", lambda: sensitivity(a.runs, R))]
    for name, fn in sections:
        if a.only and name not in a.only.split(","):
            continue
        print(f"\n== {name}")
        with open(os.path.join(a.out, f"{name}.json"), "w") as f:
            json.dump(fn(), f, indent=1)


if __name__ == "__main__":
    main()
