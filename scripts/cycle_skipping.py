"""Cycle-skipping control with a full nonlinear Helmholtz solver (CPU, independent of F-hat).

    python scripts/cycle_skipping.py [--plot cycle_skipping.png]

A 5% sound-speed inclusion (radius 20 mm, background 1500 m/s) sits inside a 128-element ring. For
inclusion contrasts m from -2% to 12% the solver generates the receiver data, and the misfit to the data
at the true 5% contrast is computed over all receivers and for the single receiver opposite the source,
at 0.20 MHz, 0.55 MHz and the sum over 0.20/0.35/0.55 MHz. Prints the false minima of the
single-receiver 0.55 MHz misfit and saves the curves to cycle_skipping.npz.
"""
import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

N, L, C0 = 140, 0.12, 1500.0           # grid, domain size (m), background sound speed (m/s)
DX = L / N
SPONGE, BETA = 16, 4.0                 # absorbing layer width (px) and strength
R_RING, R_INCL, M_TRUE = 0.052, 0.020, 0.05

ax = (np.arange(N) - (N - 1) / 2) * DX
xg, yg = np.meshgrid(ax, ax)
ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
edge = np.minimum.reduce([ii, jj, N - 1 - ii, N - 1 - jj])          # distance to the nearest grid edge (px)
ramp = np.where(edge < SPONGE, ((SPONGE - edge) / SPONGE) ** 2, 0.0).ravel()
d1 = sp.diags([np.ones(N - 1), -2 * np.ones(N), np.ones(N - 1)], [-1, 0, 1]) / DX ** 2
LAP = (sp.kron(sp.identity(N), d1) + sp.kron(d1, sp.identity(N))).tocsr()
ang = 2 * np.pi * np.arange(128) / 128
recv = np.stack([R_RING * np.cos(ang), R_RING * np.sin(ang)], 1)


def to_idx(p):
    j = np.clip(np.round(p[:, 0] / DX + (N - 1) / 2).astype(int), 0, N - 1)
    i = np.clip(np.round(p[:, 1] / DX + (N - 1) / 2).astype(int), 0, N - 1)
    return i * N + j


RECV, SRC = to_idx(recv), to_idx(np.array([[0.0, -R_RING]]))[0]
OPP = int(np.argmin(((recv - np.array([0.0, R_RING])) ** 2).sum(1)))       # receiver opposite the source
DISK = (np.sqrt(xg ** 2 + yg ** 2) < R_INCL).ravel().astype(float)


def field(m, f):
    """Solve (lap + k^2) p = -s with k = omega / c(x) and an absorbing sponge; return the whole field."""
    k2 = (2 * np.pi * f / (C0 * (1.0 + m * DISK))) ** 2 * (1 + 1j * BETA * ramp)
    s = np.zeros(N * N, complex); s[SRC] = 1.0 / DX ** 2
    return spla.spsolve((LAP + sp.diags(k2)).tocsc(), -s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    ms = np.linspace(-0.02, 0.12, 81)
    it = int(np.argmin(np.abs(ms - M_TRUE)))
    data = {f: np.stack([field(m, f)[RECV] for m in ms]) for f in (0.20e6, 0.35e6, 0.55e6)}
    norm = lambda x: x / max(x.max(), 1e-30)
    full = lambda D: np.sum(np.abs(D - D[it]) ** 2, axis=1)
    J = {"full_0.20": norm(full(data[0.20e6])), "full_0.55": norm(full(data[0.55e6])),
         "full_multi": norm(sum(full(D) for D in data.values())),
         "single_0.20": norm(np.abs(data[0.20e6][:, OPP] - data[0.20e6][it, OPP]) ** 2),
         "single_0.55": norm(np.abs(data[0.55e6][:, OPP] - data[0.55e6][it, OPP]) ** 2)}
    j = J["single_0.55"]
    false_min = [i for i in range(1, len(ms) - 1) if j[i] < j[i - 1] and j[i] < j[i + 1]
                 and abs(ms[i] - M_TRUE) > 0.012 and j[i] < 0.15]
    print("false minima of the single-receiver 0.55 MHz misfit (J < 0.15): " +
          ", ".join(f"{ms[i] * 100:.1f}% (J = {j[i]:.2f})" for i in false_min))
    print("near-zero false minima (J < 0.01): " + ", ".join(f"{ms[i] * 100:.1f}%" for i in false_min if j[i] < 0.01))
    np.savez("cycle_skipping.npz", m=ms, false_minima=ms[false_min], **J)
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        axs[0].imshow(np.real(field(0.0, 0.4e6)).reshape(N, N), cmap="RdBu_r")
        axs[0].set_title("wavefield, 0.40 MHz")
        for k, axi in (("full", axs[1]), ("single", axs[2])):
            for key, v in J.items():
                if key.startswith(k):
                    axi.plot(ms * 100, v, label=key.split("_")[1])
            axi.axvline(M_TRUE * 100, ls="--", c="k"); axi.set_xlabel("inclusion contrast m (%)"); axi.legend()
        axs[1].set_title("all 128 receivers"); axs[2].set_title("receiver opposite the source")
        axs[2].plot(ms[false_min] * 100, j[false_min], "v", c="C3")
        fig.tight_layout(); fig.savefig(a.plot, dpi=150)


if __name__ == "__main__":
    main()
