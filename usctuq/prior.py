"""PCA latent prior g(z) = mean + V z, fitted on the 583 low-density training maps.

The breast mask Omega (pixels where the training maps vary, std > 0.02, inside the tank) is defined by the
same maps. Coverage, half-width and the scales are all computed on Omega.
"""
import numpy as np
import tensorflow as tf

from .data import dense_fraction, tank_mask

DFRAC_IN = 0.12      # training maps with less than 12% dense tissue form the prior population
MASK_STD = 0.02      # breast mask threshold on the training std (normalised units)


class Prior:
    def __init__(self, mean, basis, std):
        self.mean = mean.astype("float32")             # (256, 256)
        self.basis = basis.astype("float32")           # (d, 256, 256)
        self.std = std.astype("float32")               # (256, 256) training std
        self.d = len(basis)
        self.mean_t = tf.constant(self.mean[None, :, :, None])
        self.basis_t = tf.constant(self.basis)

    def image(self, z):
        """(B, d) -> (B, 256, 256, 1)."""
        return (tf.einsum("bd,dij->bij", z, self.basis_t) + self.mean_t[..., 0])[..., None]

    def pixels(self, threshold=MASK_STD):
        """Flat indices of the pixels with training std above the threshold inside the tank."""
        return np.where(((self.std > threshold) & tank_mask()).reshape(-1))[0]

    def save(self, path):
        np.savez(path, mean=self.mean, basis=self.basis, std=self.std)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["mean"], z["basis"], z["std"])


def low_density(cn_train):
    return np.where(dense_fraction(cn_train) < DFRAC_IN)[0]


def fit_prior(cn_train, d=64):
    """PCA through the eigendecomposition of the Gram matrix (TF linalg, as in the published runs)."""
    sel = low_density(cn_train)
    x = cn_train[sel].reshape(len(sel), -1).astype("float32")
    mean = x.mean(0)
    xc = tf.constant(x - mean)
    _, v = tf.linalg.eigh(tf.matmul(xc, xc, transpose_b=True))
    v = tf.reverse(v, [1])
    basis = tf.matmul(xc, v[:, :d], transpose_a=True) / tf.sqrt(float(max(len(sel) - 1, 1)))
    return Prior(mean.reshape(256, 256), tf.reshape(tf.transpose(basis), (d, 256, 256)).numpy(),
                 cn_train[sel][..., 0].std(0))
