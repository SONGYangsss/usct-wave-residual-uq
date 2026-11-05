"""Build the measurement files from OpenBreastUS phantoms and OpenWaves (CPU; only needed to regenerate data).

    python scripts/prepare_data.py stage   --openbreastus <dir> --split test --ids-from data/ow_f300.npz --out <staging dir>
    (run OpenWaves on <staging dir> at Simulation Frequency 3.0, i.e. 300 kHz)
    python scripts/prepare_data.py extract --openbreastus <dir> --split test --wavefields <OpenWaves output>/test \
                                           --map <staging dir>/map.npy --out data/ow_f300.npz
    python scripts/prepare_data.py split   --openbreastus <dir> --eval data/ow_f300.npz --out data/ow_f300_split.npz

stage    writes one speed map per phantom as test_<i>.mat (variable `slice`, 1 x 480 x 480). OpenWaves reads
         test_1, test_2, ... contiguously for either OpenBreastUS split, so map.npy records the phantom ids.
extract  keeps the 256 x 256 receiver matrix M[s, r] (field of source s at element r) of each output file,
         with the speed map downsampled to 240 x 240.
split    density score D = fraction of the map above 1510 m/s. The OOD cut is the 0.80 quantile over the 800
         OpenBreastUS test phantoms; maps below it are split 50/25/25 into train/calib/test_id (seed 0).
"""
import argparse
import glob
import os
import re

import numpy as np
from scipy.io import loadmat, savemat

DENSE_SOS, OOD_Q, SEED = 1510.0, 0.80, 0


def load_speed(ob, split):
    """OpenBreastUS speed maps (N, 480, 480) in MATLAB orientation."""
    if split == "test":
        return loadmat(os.path.join(ob, "breast_test_speed.mat"))["breast_test"]
    import h5py
    with h5py.File(os.path.join(ob, "breast_train_speed.mat"), "r") as f:
        a = np.array(f["breast_train"])
    return np.transpose(np.moveaxis(a, int(np.argmax(a.shape)), 0), (0, 2, 1))


def load_field(path):
    """OpenWaves output variable x (256 sources, 480, 480), MATLAB v5 or v7.3."""
    try:
        return loadmat(path)["x"]
    except (NotImplementedError, ValueError):
        import h5py
        with h5py.File(path, "r") as f:
            a = f["x"][...]
        if a.dtype.names:
            a = a["real"] + 1j * a["imag"]
        return np.transpose(a, (2, 1, 0))


def stage(a):
    sp = load_speed(a.openbreastus, a.split)
    ids = np.load(a.ids_from)["idx"].astype(int) if a.ids_from.endswith(".npz") else np.load(a.ids_from).astype(int)
    os.makedirs(a.out, exist_ok=True)
    for seq, pid in enumerate(ids, start=1):
        savemat(os.path.join(a.out, f"test_{seq}.mat"), {"slice": np.asarray(sp[pid - 1], np.float64).reshape(1, 480, 480)})
    np.save(os.path.join(a.out, "map.npy"), ids)
    print(f"[stage] {len(ids)} phantoms -> {a.out}")


def extract(a):
    files = sorted(glob.glob(os.path.join(a.wavefields, "*.mat")), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[-1]))
    sp, ids = load_speed(a.openbreastus, a.split), np.load(a.map).astype(int)
    first = load_field(files[0])
    src = np.stack(np.unravel_index(np.abs(first).reshape(256, -1).argmax(1), first.shape[1:]), 1)   # source singularities
    speed, meas, idx = [], [], []
    for p in files:
        pid = ids[int(re.findall(r"\d+", os.path.basename(p))[-1]) - 1]
        meas.append(load_field(p)[:, src[:, 0], src[:, 1]].astype(np.complex64))
        speed.append(sp[pid - 1][::2, ::2].astype(np.float32)); idx.append(pid)
    np.savez_compressed(a.out, speed=np.stack(speed), meas=np.stack(meas), src_ij=src, idx=np.array(idx))
    print(f"[extract] {len(files)} phantoms -> {a.out}")


def split(a):
    ref = (load_speed(a.openbreastus, "test") > DENSE_SOS).reshape(800, -1).mean(1).astype(np.float32)
    cut = float(np.quantile(ref, OOD_Q))
    ids = np.load(a.eval)["idx"].astype(int)
    score = ref[ids - 1]
    ind, ood = ids[score < cut], ids[score >= cut]
    perm = np.random.default_rng(SEED).permutation(len(ind))
    nc = nt = int(round(0.25 * len(ind)))
    parts = {"calib": ind[perm[:nc]], "test_id": ind[perm[nc:nc + nt]], "train": ind[perm[nc + nt:]], "ood": ood}
    parts = {k: np.sort(v) for k, v in parts.items()}
    np.savez(a.out, cut=cut, **parts, **{f"{k}_score": ref[v - 1] for k, v in parts.items()})
    print(f"[split] cut {cut:.4f}: " + ", ".join(f"{k} {len(v)}" for k, v in parts.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("stage", "extract", "split"))
    ap.add_argument("--openbreastus", required=True, help="folder with breast_train_speed.mat and breast_test_speed.mat")
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--ids-from", help="npz with idx, or .npy of phantom ids (stage)")
    ap.add_argument("--wavefields", help="OpenWaves output folder (extract)")
    ap.add_argument("--map", help="map.npy written by stage (extract)")
    ap.add_argument("--eval", help="evaluation npz (split)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"stage": stage, "extract": extract, "split": split}[a.step](a)
