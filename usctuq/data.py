"""Data loading, measurement normalisation and acquisition perturbations.

Each data file is an npz with
    speed  (N, H, W) float32   sound-speed maps in m/s (OpenBreastUS phantoms)
    meas   (N, 256, 256) complex64   receiver matrix M[s, r] at 300 kHz (OpenWaves)
    idx    (N,) int            OpenBreastUS phantom ids (1-based)
    src_ij (256, 2) int        ring-element positions on the 480 x 480 simulation grid
"""
import os
import numpy as np

C_WATER, C_SCALE = 1480.0, 100.0   # maps are normalised as (c - C_WATER) / C_SCALE
GRID = 256                         # reconstruction grid
SELF_W = 2                         # receivers within 2 elements of the source are masked
DENSE_SOS = 1510.0                 # m/s, threshold of the density score
TANK_R = 118                       # px, radius of the region inside the ring on the 256 grid

EVAL_FILE, SPLIT_FILE = "ow_f300.npz", "ow_f300_split.npz"
TRAIN_FILES = ("ow_train.npz", "ow_train_b.npz")


def data_dir(path=None):
    return path or os.environ.get("USCTUQ_DATA", "data")


def to_grid(speed):
    """(N, H, W) m/s -> (N, 256, 256, 1) normalised maps (bilinear, as in all published runs)."""
    import tensorflow as tf
    cn = (speed.astype("float32") - C_WATER) / C_SCALE
    return tf.image.resize(cn[..., None], (GRID, GRID), method="bilinear").numpy()


def load_npz(path):
    z = np.load(path)
    return {"cn": to_grid(z["speed"]), "meas": z["meas"], "ids": z["idx"].astype(int),
            "src_ij": z["src_ij"], "speed": z["speed"]}


def load_train(ddir):
    """The 799 FNO training maps (two files generated from the OpenBreastUS training split)."""
    parts = [load_npz(os.path.join(ddir, f)) for f in TRAIN_FILES]
    return {k: np.concatenate([p[k] for p in parts], 0) for k in ("cn", "meas", "ids", "speed")}


def load_eval(ddir):
    """The 200 evaluation maps and their row indices per partition (train/calib/test_id/ood)."""
    ev = load_npz(os.path.join(ddir, EVAL_FILE))
    split = np.load(os.path.join(ddir, SPLIT_FILE))
    pos = {int(p): k for k, p in enumerate(ev["ids"])}
    ev["rows"] = {k: np.array([pos[int(p)] for p in split[k]], int) for k in ("train", "calib", "test_id", "ood")}
    return ev


def ring_mask():
    """(256, 256) mask over (source, receiver) pairs that drops the source singularity."""
    i, j = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    d = np.minimum(np.abs(i - j), GRID - np.abs(i - j))
    return (d > SELF_W).astype("float32")


def tank_mask():
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    return ((yy - GRID // 2) ** 2 + (xx - GRID // 2) ** 2) <= TANK_R ** 2


def dense_fraction(cn):
    """Fraction of the tank above 1510 m/s on the 256 grid; < 0.12 selects the 583 low-density maps."""
    tank = tank_mask()
    return ((cn[..., 0] > (DENSE_SOS - C_WATER) / C_SCALE) & tank[None]).reshape(len(cn), -1).sum(1) / float(tank.sum())


def density_score(speed):
    """Density score D_j: fraction of pixels of the stored speed map above 1510 m/s."""
    return (np.asarray(speed) > DENSE_SOS).reshape(len(speed), -1).mean(1)


class MeasNorm:
    """Real/imag scaling of the receiver matrix and the mean training field ybar."""

    def __init__(self, scale, ybar):
        self.scale, self.ybar = float(scale), ybar.astype("float32")

    @classmethod
    def fit(cls, meas_train):
        m = ring_mask()
        mr, mi = np.real(meas_train).astype("float32"), np.imag(meas_train).astype("float32")
        scale = float(np.std(np.stack([mr, mi])[:, :, m > 0]))
        return cls(scale, cls._to_y(meas_train, scale).mean(0))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["M_SCALE"], z["ybarN"])

    def save(self, path, **extra):
        np.savez(path, M_SCALE=self.scale, ybarN=self.ybar, **extra)

    @staticmethod
    def _to_y(meas, scale):
        return np.stack([np.real(meas) / scale, np.imag(meas) / scale], -1).astype("float32")

    def to_y(self, meas):
        """complex (N, 256, 256) -> (N, 256, 256, 2) float32 total field."""
        return self._to_y(meas, self.scale)


def parse_perturbation(spec):
    """'none' | 'gain:<fraction>,<degrees>' | 'noise:<snr dB>'."""
    if not spec or spec == "none":
        return None
    kind, val = spec.split(":")
    if kind == "gain":
        amp, deg = (float(v) for v in val.split(","))
        return ("gain", amp, deg)
    if kind == "noise":
        return ("noise", float(val))
    raise ValueError(f"unknown perturbation {spec}")


def perturb(meas, norm, spec, gain_seed=20261010, noise_seed=0):
    """Normalised measurements of the whole evaluation set under an acquisition perturbation.

    gain:  per-element complex gain g_e = (1 + a n_e) exp(i phi n'_e), applied as M[s,r] g_s g_r,
           one seeded realisation shared by all phantoms (a device property).
    noise: complex Gaussian noise scaled to each phantom's scattered-field RMS on the ring mask,
           one seeded realisation for all levels.
    """
    p = parse_perturbation(spec)
    if p is None:
        return norm.to_y(meas)
    if p[0] == "gain":
        rng = np.random.default_rng(gain_seed)
        na, nphi = rng.standard_normal(256), rng.standard_normal(256)
        g = (1 + p[1] * na) * np.exp(1j * np.deg2rad(p[2]) * nphi)
        return norm.to_y(meas * (g[:, None] * g[None, :])[None].astype(np.complex64))
    y = norm.to_y(meas)
    m = ring_mask()[None, :, :, None]
    rms = np.sqrt((((y - norm.ybar[None]) * m) ** 2).reshape(len(y), -1).mean(1))
    noise = np.random.default_rng(noise_seed).standard_normal(y.shape).astype("float32")
    return (y + (10 ** (-p[1] / 20.0) * rms[:, None, None, None]) * noise).astype("float32")
