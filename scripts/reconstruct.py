"""Latent reconstruction of the evaluation phantoms and the case-level scales (GPU).

    python scripts/reconstruct.py --name main --splits train,calib,test_id,ood --case-w
    python scripts/reconstruct.py --name fno_lowdens_s0 --fno checkpoints/fno_lowdens_s0.weights.npz
    python scripts/reconstruct.py --name acq_gain10 --perturb gain:0.10,10
    python scripts/reconstruct.py --name acq_noise00 --perturb noise:0
    python scripts/reconstruct.py --name pca128 --pca-dim 128 --mask-std 0.005

Writes <data>/runs/recon_<name>.npz with, on the stored pixels (training std > --mask-std inside the tank),
the reconstruction cr, the ground truth cg, the back-projected residual map u, the residual norm ares,
the forward error efwd and the density score dens per split, and the shared relative width w.
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from usctuq import data, fno, prior, recon  # noqa: E402

for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="data folder (default: $USCTUQ_DATA or ./data)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--fno", default=None, help="surrogate weights (default: checkpoints/fno_main.weights.npz)")
    ap.add_argument("--splits", default="calib,test_id,ood")
    ap.add_argument("--perturb", default="none", help="none | gain:<fraction>,<degrees> | noise:<snr dB>")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--mask-std", type=float, default=prior.MASK_STD, help="store pixels with training std above this")
    ap.add_argument("--case-w", action="store_true", help="also linearise w at each case estimate")
    ap.add_argument("--steps", type=int, default=recon.STEPS)
    ap.add_argument("--limit", type=int, default=0, help="first N phantoms per split (quick check)")
    a = ap.parse_args()

    ddir = data.data_dir(a.data)
    ck = a.fno or os.path.join(ddir, "checkpoints", "fno_main.weights.npz")
    tr, ev = data.load_train(ddir), data.load_eval(ddir)
    norm = data.MeasNorm.load(ck.replace(".weights.npz", ".meta.npz"))
    pr = prior.fit_prior(tr["cn"], a.pca_dim)
    del tr
    pix = pr.pixels(a.mask_std)
    rc = recon.Reconstructor(fno.load_fno(ck), pr)
    print(f"[setup] surrogate {os.path.basename(ck)}  PCA d={pr.d}  stored pixels {len(pix)}  perturbation {a.perturb}")

    y = data.perturb(ev["meas"], norm, a.perturb)          # whole evaluation set: one noise realisation
    split = np.load(os.path.join(ddir, data.SPLIT_FILE))
    out = {"pix": pix, "std_pix": pr.std.reshape(-1)[pix], "w": rc.width(pix)}
    for s in a.splits.split(","):
        rows = ev["rows"][s][:a.limit or None]
        ydev = y[rows] - norm.ybar[None]
        r = rc.reconstruct(ydev, a.steps)
        flat = lambda m: m.reshape(len(m), -1)[:, pix].astype("float32")
        out.update({f"cr_{s}": flat(r["c"]), f"cg_{s}": flat(ev["cn"][rows][..., 0]), f"u_{s}": flat(r["u"]),
                    f"ares_{s}": r["a_res"].astype("float32"),
                    f"efwd_{s}": rc.forward_error(ev["cn"][rows], norm.to_y(ev["meas"][rows]), norm.ybar).astype("float32"),
                    f"dens_{s}": split[f"{s}_score"][:len(rows)], f"ids_{s}": ev["ids"][rows]})
        if a.case_w:
            out[f"wcase_{s}"] = np.stack([rc.width(pix, at=c) for c in r["c"]])
        e = out[f"cr_{s}"] - out[f"cg_{s}"]
        print(f"[{s}] N={len(rows)}  RMSE {np.sqrt((e ** 2).mean()) * data.C_SCALE:.1f} m/s  median a_res {np.median(r['a_res']):.3f}")
    path = os.path.join(ddir, "runs", f"recon_{a.name}.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)
    print(f"[save] {path}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)    # TF 2.10 on Windows can crash during CUDA teardown after a successful run
