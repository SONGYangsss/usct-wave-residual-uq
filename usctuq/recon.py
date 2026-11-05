"""Latent reconstruction through F-hat and the case-level data-consistency scales.

For a measurement deviation y - ybar (normalised, ring-masked) the latent fit solves
    z* = argmin_z  ||M (F(g(z)) - (y - ybar))||^2 + ridge ||z||^2,    c* = g(z*)
and gives, per case,
    a_res   = ||M (F(c*) - (y - ybar))|| / ||M (y - ybar)||          residual norm (default scale)
    u_phys  = |J(c*)^T M ((y - ybar) - F(c*))|                      back-projected residual map
    a_phys  = median of u_phys over the breast mask                  back-projected scale (variant)
The relative width w is the latent posterior std, linearised at the prior mean and shared by all cases.
"""
import numpy as np
import tensorflow as tf

from .data import ring_mask

STEPS, LR, RIDGE, CHUNK = 400, 0.06, 1e-3, 10


class Reconstructor:
    def __init__(self, fno, prior):
        self.fno, self.prior = fno, prior
        self.mask_np = ring_mask()
        self.mask = tf.constant(self.mask_np[None, :, :, None])

    # ------------------------------------------------------------------ latent fit
    @tf.function(reduce_retracing=True)
    def _grad(self, z, target, b):
        with tf.GradientTape() as tape:
            r = (self.fno(self.prior.image(z), training=False) - target) * self.mask
            loss = tf.reduce_sum(r ** 2) / b + RIDGE * tf.reduce_sum(z ** 2)
        return tape.gradient(loss, z)

    @tf.function(reduce_retracing=True)
    def _vjp(self, c, cotangent):
        with tf.GradientTape() as tape:
            tape.watch(c)
            pred = self.fno(c, training=False)
        return pred, tape.gradient(pred, c, output_gradients=cotangent)

    def fit(self, ydev, steps=STEPS):
        """ydev (B, 256, 256, 2) -> c*, a_res, u_phys for one chunk.

        The loss averages the data misfit over the chunk and sums the ridge term, with chunks of 10
        phantoms in every published run.
        """
        target = tf.constant(ydev)
        z = tf.Variable(tf.zeros((len(ydev), self.prior.d)))
        opt = tf.keras.optimizers.Adam(LR)
        b = tf.constant(float(len(ydev)))
        for _ in range(steps):
            opt.apply_gradients([(self._grad(z, target, b), z)])
        c = self.prior.image(z)
        pred = self.fno(c, training=False)
        _, g = self._vjp(c, (target - pred) * self.mask)
        a_res = tf.sqrt(tf.reduce_sum(((pred - target) * self.mask) ** 2, [1, 2, 3]) /
                        tf.reduce_sum((target * self.mask) ** 2, [1, 2, 3]))
        return c.numpy()[..., 0], a_res.numpy(), tf.abs(g[..., 0]).numpy(), z.numpy()

    def reconstruct(self, ydev, steps=STEPS, log=print):
        out = {k: [] for k in ("c", "a_res", "u", "z")}
        for s in range(0, len(ydev), CHUNK):
            for k, v in zip(out, self.fit(ydev[s:s + CHUNK], steps)):
                out[k].append(v)
            log(f"    phantoms {s}-{min(s + CHUNK, len(ydev))} of {len(ydev)}")
        return {k: np.concatenate(v) for k, v in out.items()}

    # ------------------------------------------------------------------ other quantities
    def a_res_of(self, c, ydev):
        """Residual norm of any reconstruction c (B, 256, 256), e.g. a baseline's mean map."""
        out = []
        for s in range(0, len(c), CHUNK):
            pred = self.fno(tf.constant(c[s:s + CHUNK, :, :, None]), training=False)
            t = ydev[s:s + CHUNK]
            out.append(np.sqrt((((pred.numpy() - t) * self.mask_np[None, :, :, None]) ** 2).sum((1, 2, 3)) /
                               ((t * self.mask_np[None, :, :, None]) ** 2).sum((1, 2, 3))))
        return np.concatenate(out)

    def forward_error(self, c_true, y_total, ybar):
        """e_fwd = ||M (y - F(c_true) - ybar)|| / ||M y|| on the total normalised field."""
        m3, out = self.mask_np[None, :, :, None], []
        for s in range(0, len(c_true), CHUNK):
            yp = self.fno(c_true[s:s + CHUNK], training=False).numpy() + ybar
            y = y_total[s:s + CHUNK]
            num = np.sqrt((((y - yp) * m3) ** 2).reshape(len(y), -1).sum(1))
            out.append(num / np.maximum(np.sqrt(((y * m3) ** 2).reshape(len(y), -1).sum(1)), 1e-12))
        return np.concatenate(out)

    def backproject(self, ydev):
        """b = J(mean)^T M (y - ybar): input of the image-domain baselines."""
        out = np.empty((len(ydev), 256, 256), np.float32)
        for s in range(0, len(ydev), CHUNK):
            x = tf.tile(self.prior.mean_t, [len(ydev[s:s + CHUNK]), 1, 1, 1])
            _, g = self._vjp(x, tf.constant(ydev[s:s + CHUNK]) * self.mask)
            out[s:s + CHUNK] = g[..., 0].numpy()
        return out

    def width(self, pix, at=None, batch=8):
        """sigma_w on the pixels pix: sqrt(diag(V (J^T J + ridge I)^-1 V^T)), J = dF(g(z))/dz.

        at=None linearises at the prior mean (the shared w of the paper); otherwise at the map `at`.
        """
        prim = self.prior.mean_t if at is None else tf.constant(at[None, :, :, None])
        cols = []
        for s in range(0, self.prior.d, batch):
            tang = tf.constant(self.prior.basis[s:s + batch, :, :, None])
            x = tf.tile(prim, [len(tang), 1, 1, 1])
            with tf.autodiff.ForwardAccumulator(x, tang) as acc:
                p = self.fno(x, training=False)
            cols.append((acc.jvp(p).numpy() * self.mask_np[None, :, :, None]).reshape(len(tang), -1))
        jac = tf.constant(np.concatenate(cols, 0).T.astype(np.float64))
        cov = tf.linalg.inv(tf.matmul(jac, jac, transpose_a=True) + RIDGE * tf.eye(self.prior.d, dtype=tf.float64))
        v = tf.constant(self.prior.basis.reshape(self.prior.d, -1).T[pix].astype(np.float64))
        return np.sqrt(np.maximum(tf.reduce_sum(tf.matmul(v, cov) * v, 1).numpy(), 0)).astype("float32")
