"""Learned image-domain baselines: MC dropout, deep ensemble and conditional diffusion (GPU).

    python scripts/run_baselines.py --name main
    python scripts/run_baselines.py --name acq_gain10 --perturb gain:0.10,10 --methods unet

All baselines take the back-projection b = J(mean)^T M (y - ybar) of the main surrogate as input and are
trained on the 583 low-density training maps. Missing checkpoints are trained first (U-Net: 5 members,
60 epochs; diffusion: 200 epochs) and saved to checkpoints/. Writes <data>/runs/baselines_<name>.npz with,
on the breast mask, the absolute error and the native spread of each method, and per case the RMSE,
the mean spread (m/s) and the residual norm a_res of the method's mean reconstruction.
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from usctuq import baselines as bl, data, fno, prior, recon  # noqa: E402

for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--methods", default="unet,diffusion")
    ap.add_argument("--splits", default="calib,test_id,ood")
    ap.add_argument("--perturb", default="none")
    ap.add_argument("--seed", type=int, default=20261010, help="dropout masks and diffusion sampling")
    a = ap.parse_args()

    ddir = data.data_dir(a.data)
    ckdir = os.path.join(ddir, "checkpoints")
    ck = os.path.join(ckdir, "fno_main.weights.npz")
    tr, ev = data.load_train(ddir), data.load_eval(ddir)
    norm = data.MeasNorm.load(ck.replace(".weights.npz", ".meta.npz"))
    pr = prior.fit_prior(tr["cn"])
    pix = pr.pixels()
    rc = recon.Reconstructor(fno.load_fno(ck), pr)

    low = prior.low_density(tr["cn"])
    b_train = rc.backproject(norm.to_y(tr["meas"][low]) - norm.ybar[None])
    bm, bs = float(b_train.mean()), float(b_train.std() + 1e-9)
    scale = lambda b: ((b - bm) / bs)[..., None].astype("float32")
    x_train, c_train = scale(b_train), tr["cn"][low]
    del tr
    y = data.perturb(ev["meas"], norm, a.perturb)
    sets = {}
    for s in a.splits.split(","):
        rows = ev["rows"][s]
        ydev = y[rows] - norm.ybar[None]
        sets[s] = (scale(rc.backproject(ydev)), ev["cn"][rows][..., 0], ydev)

    out = {"pix": pix}
    flat = lambda m: m.reshape(len(m), -1)[:, pix].astype("float32")

    def record(method, s, mu, sd):
        _, truth, ydev = sets[s]
        e, u = flat(np.abs(mu - truth)), flat(sd)
        out.update({f"{method}_err_{s}": e, f"{method}_u_{s}": u,
                    f"{method}_rmse_{s}": np.sqrt((e.astype("float64") ** 2).mean(1)) * data.C_SCALE,
                    f"{method}_spread_{s}": u.astype("float64").mean(1) * data.C_SCALE,
                    f"{method}_ares_{s}": rc.a_res_of(mu, ydev)})
        print(f"  {method:9s} {s:7s} RMSE {np.sqrt((e.astype('float64') ** 2).mean()) * data.C_SCALE:.1f} m/s")

    if "unet" in a.methods:
        members = []
        for i in range(bl.N_ENS):
            m, path = bl.unet(1000 + i), os.path.join(ckdir, f"unet_m{i}")
            if os.path.exists(path + ".index"):
                m.load_weights(path).expect_partial()
            else:
                m.fit(x_train, c_train, batch_size=bl.UNET_BATCH, epochs=bl.UNET_EPOCHS, verbose=0)
                m.save_weights(path)
            members.append(m)
        tf.random.set_seed(a.seed)
        for s, (x, _, _) in sets.items():
            record("dropout", s, *bl.mean_std([members[0]] * bl.K_MC, True, x))
            record("ensemble", s, *bl.mean_std(members, False, x))
    if "diffusion" in a.methods:
        net, path = bl.diffusion_net(), os.path.join(ckdir, "diffusion")
        if os.path.exists(path + ".index"):
            net.load_weights(path).expect_partial()
        else:
            bl.train_diffusion(net, c_train, x_train)
            net.save_weights(path)
        sample = bl.diffusion_sampler(net)
        np.random.seed(a.seed); tf.random.set_seed(a.seed)
        for s, (x, _, _) in sets.items():
            record("diffusion", s, *sample(x))
    path = os.path.join(ddir, "runs", f"baselines_{a.name}.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)
    print(f"[save] {path}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
