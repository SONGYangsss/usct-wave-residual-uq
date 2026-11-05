"""Learned forward surrogate F-hat: sound-speed map (256, 256, 1) -> receiver matrix (256, 256, 2).

A light convolutional encoder-decoder brings the map to the 256 x 256 (source, receiver) grid, followed by
Fourier neural operator blocks. Both measurement axes are periodic on the ring, so the FFT has no edges.
The network predicts the deviation of the normalised field from the training mean ybar.
"""
import numpy as np
import tensorflow as tf

L = tf.keras.layers
BASE, MODES, BLOCKS = 32, 32, 4


class SpectralConv2d(L.Layer):
    """Keeps the lowest m1 x m2 Fourier modes and mixes channels with learned complex weights."""

    def __init__(self, cout, m1, m2, **kw):
        super().__init__(**kw)
        self.cout, self.m1, self.m2 = cout, m1, m2

    def build(self, shape):
        cin = int(shape[-1])
        init = tf.random_normal_initializer(stddev=(2.0 / (cin * self.cout)) ** 0.5)
        sh = (self.m1, self.m2, cin, self.cout)
        self.w1r, self.w1i, self.w2r, self.w2i = (self.add_weight(name=n, shape=sh, initializer=init)
                                                  for n in ("w1r", "w1i", "w2r", "w2i"))

    def call(self, x):
        h, w = int(x.shape[1]), int(x.shape[2])
        xf = tf.signal.rfft2d(tf.transpose(x, [0, 3, 1, 2]))
        w1, w2 = tf.complex(self.w1r, self.w1i), tf.complex(self.w2r, self.w2i)
        top = tf.einsum("bcxy,xyco->boxy", xf[:, :, :self.m1, :self.m2], w1)
        bot = tf.einsum("bcxy,xyco->boxy", xf[:, :, h - self.m1:, :self.m2], w2)
        pad_w = w // 2 + 1 - self.m2
        top = tf.pad(top, [[0, 0], [0, 0], [0, h - self.m1], [0, pad_w]])
        bot = tf.pad(bot, [[0, 0], [0, 0], [h - self.m1, 0], [0, pad_w]])
        return tf.transpose(tf.signal.irfft2d(top + bot, fft_length=[h, w]), [0, 2, 3, 1])


def build_fno(base=BASE, modes=MODES, blocks=BLOCKS):
    inp = L.Input((256, 256, 1))
    x = inp
    for ch in (base, base * 2):                       # encoder 256 -> 64
        x = L.ReLU()(L.Conv2D(ch, 3, strides=2, padding="same")(x))
    x = L.ReLU()(L.Conv2D(base * 2, 3, padding="same")(x))
    for ch in (base, base):                           # decoder back to the 256 measurement grid
        x = L.ReLU()(L.Conv2DTranspose(ch, 3, strides=2, padding="same")(x))
    x = L.Conv2D(base, 1, padding="same")(x)
    for _ in range(blocks):
        x = L.ReLU()(SpectralConv2d(base, modes, modes)(x) + L.Conv2D(base, 1, padding="same")(x))
    return tf.keras.Model(inp, L.Conv2D(2, 1, padding="same")(x))


def save_fno(model, path):
    np.savez(path, **{f"w{i}": w for i, w in enumerate(model.get_weights())})


def load_fno(path):
    model = build_fno()
    z = np.load(path)
    model.set_weights([z[f"w{i}"] for i in range(len(model.get_weights()))])
    return model
