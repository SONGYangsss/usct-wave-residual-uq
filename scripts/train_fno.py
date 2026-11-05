"""Train the forward surrogate F-hat on (sound-speed map, receiver matrix) pairs (GPU).

    python scripts/train_fno.py --subset all --select test_id --name fno_main      # main surrogate of the paper
    python scripts/train_fno.py --subset lowdens --seed 0 --name fno_lowdens_s0     # control surrogates
    python scripts/train_fno.py --subset matched --seed 1 --name fno_matched_s1
    python scripts/train_fno.py --subset all --seed 0 --name fno_full_s0

--subset all: the 799 training maps; lowdens: the 583 low-density maps (the prior population);
matched: 583 maps drawn at random from all 799 (fixed draw). Loss: masked squared error on the
normalised scattered field (the training mean ybar is subtracted), with the exact symmetries of the
ring as augmentation. --select test_id keeps the epoch with the lowest ID-test forward error, as for
the main surrogate of the paper; --select final keeps the last epoch, as for every control.
Writes checkpoints/<name>.weights.npz and checkpoints/<name>.meta.npz (measurement normalisation).
"""
import argparse
import os
import random
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from usctuq import data, fno, prior  # noqa: E402

for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
MATCHED_SEED = 20261010


def ring_symmetries(src_ij):
    """The rotations/reflections of the image that map the ring elements onto themselves.

    Returns (image op, element permutation) pairs; ops whose nearest-element match is not exact are dropped
    (the ring centre sits half a pixel off the grid centre, so 4 of the 8 survive).
    """
    out = []
    for k in range(4):
        for flip in (False, True):
            op = (lambda a, k=k, f=flip: np.fliplr(np.rot90(a, k)) if f else np.rot90(a, k))
            lab = -np.ones((480, 480), int)
            lab[src_ij[:, 0], src_ij[:, 1]] = np.arange(256)
            lt = op(lab)
            ys, xs = np.where(lt >= 0)
            d = ((src_ij[:, None, :] - np.stack([ys, xs], 1)[None].astype(float)) ** 2).sum(-1)
            nn = d.argmin(1)
            perm = lt[ys, xs][nn]
            if len(np.unique(perm)) == 256 and np.sqrt(d[np.arange(256), nn]).max() < 3.0:
                out.append((op, perm))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--subset", choices=("all", "lowdens", "matched"), default="all")
    ap.add_argument("--select", choices=("final", "test_id"), default="final")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); tf.random.set_seed(a.seed)

    ddir = data.data_dir(a.data)
    tr, ev = data.load_train(ddir), data.load_eval(ddir)
    low = prior.low_density(tr["cn"])
    sel = {"all": np.arange(len(tr["cn"])), "lowdens": low,
           "matched": np.sort(np.random.default_rng(MATCHED_SEED).choice(len(tr["cn"]), len(low), replace=False))}[a.subset]
    cn = tr["cn"][sel]
    norm = data.MeasNorm.fit(tr["meas"][sel])
    y = norm.to_y(tr["meas"][sel])                     # total field; ybar is subtracted after the symmetry
    y_id = norm.to_y(ev["meas"][ev["rows"]["test_id"]])
    cn_id = ev["cn"][ev["rows"]["test_id"]]
    sym = ring_symmetries(np.round(ev["src_ij"]).astype(int))
    mask = data.ring_mask()
    print(f"[data] {a.subset}: {len(sel)} training maps, {len(sym)} exact ring symmetries, scale {norm.scale:.2f}")

    model = fno.build_fno()
    steps = int(np.ceil(len(cn) / a.batch))
    opt = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(a.lr, a.epochs * steps))
    mask_t = tf.constant(mask[None])

    @tf.function
    def train_step(x, t):
        with tf.GradientTape() as tape:
            se = tf.reduce_sum((model(x, training=True) - t) ** 2, -1) * mask_t
            loss = tf.reduce_sum(se) / (tf.reduce_sum(mask_t) * tf.cast(tf.shape(x)[0], tf.float32))
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss

    def rel_err():
        num = den = 0.0
        for s in range(0, len(cn_id), 16):
            p = model(cn_id[s:s + 16], training=False).numpy() + norm.ybar
            num += (((p - y_id[s:s + 16]) ** 2).sum(-1) * mask).sum()
            den += ((y_id[s:s + 16] ** 2).sum(-1) * mask).sum()
        return float(np.sqrt(num / den))

    out = os.path.join(ddir, "checkpoints", a.name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rng, best = np.random.default_rng(a.seed), np.inf
    for ep in range(a.epochs):
        perm, losses = rng.permutation(len(cn)), []
        for b in range(0, len(cn), a.batch):
            xb, yb = [], []
            for r in perm[b:b + a.batch]:
                op, q = sym[rng.integers(len(sym))]
                xb.append(op(cn[r, :, :, 0])[..., None]); yb.append(y[r][q][:, q] - norm.ybar)
            losses.append(float(train_step(tf.constant(np.asarray(xb, "float32")), tf.constant(np.asarray(yb, "float32")))))
        if ep % 10 == 0 or ep == a.epochs - 1:
            e = rel_err()
            tag = ""
            if a.select == "test_id" and e < best:
                best = e; fno.save_fno(model, out + ".weights.npz"); tag = " (saved)"
            print(f"[epoch {ep:3d}] loss {np.mean(losses):.4f}  ID-test forward error {e:.3f}{tag}", flush=True)
    if a.select == "final":
        fno.save_fno(model, out + ".weights.npz")
    norm.save(out + ".meta.npz", subset=sel, seed=a.seed, select=a.select)
    print(f"[save] {out}.weights.npz")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
