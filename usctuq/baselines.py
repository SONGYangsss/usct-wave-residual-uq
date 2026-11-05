"""Image-domain learned baselines. Input: the normalised back-projection b = J(mean)^T M (y - ybar).

U-Net (5 independently initialised members): the deep ensemble uses all five, MC dropout uses member 0
with K = 20 stochastic passes. Conditional diffusion: cosine-schedule DDPM with x0 prediction,
multi-scale conditioning on b, EMA weights and K = 10 stochastic DDIM samples (20 steps).
"""
import numpy as np
import tensorflow as tf

L = tf.keras.layers
N_ENS, K_MC, UNET_EPOCHS, UNET_BATCH = 5, 20, 60, 16
T, TDIM, K_DIFF, N_STEP, DIFF_EPOCHS = 1000, 128, 10, 20, 200


# ---------------------------------------------------------------- U-Net (dropout and ensemble)
def _block(x, ch, drop):
    x = L.Conv2D(ch, 3, padding="same", activation="elu")(x)
    x = L.Conv2D(ch, 3, padding="same", activation="elu")(x)
    return L.SpatialDropout2D(drop)(x)


def unet(seed, drop=0.2):
    tf.random.set_seed(seed)
    inp = L.Input((256, 256, 1))
    c1 = _block(inp, 32, drop); p1 = L.MaxPool2D()(c1)
    c2 = _block(p1, 64, drop); p2 = L.MaxPool2D()(c2)
    c3 = _block(p2, 128, drop)
    d2 = _block(L.Concatenate()([L.UpSampling2D()(c3), c2]), 64, drop)
    d1 = _block(L.Concatenate()([L.UpSampling2D()(d2), c1]), 32, drop)
    m = tf.keras.Model(inp, L.Conv2D(1, 1)(d1))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m


def mean_std(models, training, x, batch=20):
    """Pixelwise mean and std over a list of stochastic or ensemble passes."""
    s = np.zeros((len(x), 256, 256)); s2 = np.zeros_like(s)
    for m in models:
        for i in range(0, len(x), batch):
            p = m(x[i:i + batch], training=training).numpy()[..., 0].astype(np.float64)
            s[i:i + batch] += p; s2[i:i + batch] += p * p
    mu = s / len(models)
    return mu.astype(np.float32), np.sqrt(np.maximum(s2 / len(models) - mu * mu, 0)).astype(np.float32)


# ---------------------------------------------------------------- conditional diffusion
_t = np.arange(T + 1)
_f = np.cos(((_t / T) + 0.008) / 1.008 * np.pi / 2) ** 2
ALPHA_BAR = np.clip((_f / _f[0]).astype(np.float32), 1e-5, 1.0)
STEPS = np.unique(np.linspace(0, T - 1, N_STEP).round().astype(int))


def time_embedding(t):
    half = TDIM // 2
    fr = tf.exp(-tf.math.log(10000.0) * tf.range(half, dtype=tf.float32) / half)
    a = tf.cast(t[:, None], tf.float32) * fr[None, :]
    return tf.concat([tf.sin(a), tf.cos(a)], 1)


def _dblock(h, te, ch):
    h = L.Conv2D(ch, 3, padding="same")(h)
    tp = L.Reshape((1, 1, ch))(L.Dense(ch)(te))
    h = L.Activation("swish")(L.Add()([h, tp]))
    return L.Conv2D(ch, 3, padding="same", activation="swish")(h)


def diffusion_net():
    # statement order kept as in the published code: TF checkpoints restore layers in this order
    ix, ib, it = L.Input((256, 256, 1)), L.Input((256, 256, 1)), L.Input((TDIM,))
    te = L.Dense(TDIM, activation="swish")(it)
    te = L.Dense(TDIM)(te)
    b1 = ib; b2 = L.AveragePooling2D()(b1); b3 = L.AveragePooling2D()(b2); b4 = L.AveragePooling2D()(b3)
    c1 = _dblock(L.Concatenate()([ix, b1]), te, 48); p1 = L.MaxPool2D()(c1)
    c2 = _dblock(L.Concatenate()([p1, b2]), te, 96); p2 = L.MaxPool2D()(c2)
    c3 = _dblock(L.Concatenate()([p2, b3]), te, 192); p3 = L.MaxPool2D()(c3)
    bo = _dblock(L.Concatenate()([p3, b4]), te, 256)
    d3 = _dblock(L.Concatenate()([L.UpSampling2D()(bo), c3]), te, 192)
    d2 = _dblock(L.Concatenate()([L.UpSampling2D()(d3), c2]), te, 96)
    d1 = _dblock(L.Concatenate()([L.UpSampling2D()(d2), c1]), te, 48)
    out = L.Lambda(lambda z: 1.1 * tf.tanh(z))(L.Conv2D(1, 3, padding="same")(d1))
    return tf.keras.Model([ix, ib, it], out)


def train_diffusion(model, x0, b, epochs=DIFF_EPOCHS, batch=8, ema_decay=0.999, log=print):
    opt = tf.keras.optimizers.Adam(2e-4)
    ema = [tf.Variable(w, trainable=False) for w in model.weights]
    abar = tf.constant(ALPHA_BAR)

    @tf.function
    def step(x0b, bb):
        t = tf.random.uniform([tf.shape(x0b)[0]], 1, T, dtype=tf.int32)
        ac = tf.gather(abar, t)[:, None, None, None]
        xt = tf.sqrt(ac) * x0b + tf.sqrt(1.0 - ac) * tf.random.normal(tf.shape(x0b))
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean((model([xt, bb, time_embedding(t)], training=True) - x0b) ** 2)
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        for e, w in zip(ema, model.weights):
            e.assign(ema_decay * e + (1.0 - ema_decay) * w)
        return loss

    nb = max(len(x0) // batch, 1)
    for ep in range(epochs):
        idx, tot = np.random.permutation(len(x0)), 0.0
        for i in range(nb):
            s = idx[i * batch:(i + 1) * batch]
            tot += float(step(tf.constant(x0[s]), tf.constant(b[s])))
        if ep % 20 == 0 or ep == epochs - 1:
            log(f"  diffusion epoch {ep} loss {tot / nb:.4f}")
    for w, e in zip(model.weights, ema):
        w.assign(e)


def diffusion_sampler(model, eta=1.0):
    @tf.function
    def trajectory(b, x):
        for i in range(len(STEPS) - 1, 0, -1):
            tc, tp = int(STEPS[i]), int(STEPS[i - 1])
            x0h = model([x, b, time_embedding(tf.fill([tf.shape(x)[0]], tc))], training=False)
            ac, ap = float(ALPHA_BAR[tc]), float(ALPHA_BAR[tp])
            eps = (x - ac ** 0.5 * x0h) / max((1.0 - ac) ** 0.5, 1e-2)
            sig = eta * ((1.0 - ap) / (1.0 - ac)) ** 0.5 * (1.0 - ac / ap) ** 0.5 if ap < 1.0 else 0.0
            x = ap ** 0.5 * x0h + max(1.0 - ap - sig * sig, 0.0) ** 0.5 * eps + sig * tf.random.normal(tf.shape(x))
        return x

    def sample(b, k=K_DIFF, batch=12):
        """b (N, 256, 256, 1) -> pixelwise mean and std over k posterior samples."""
        cond = np.repeat(b, k, axis=0)
        res = np.zeros((len(cond), 256, 256), np.float32)
        for s in range(0, len(cond), batch):
            bb = cond[s:s + batch]
            x = np.random.randn(len(bb), 256, 256, 1).astype(np.float32)
            res[s:s + batch] = np.nan_to_num(trajectory(tf.constant(bb), tf.constant(x)).numpy()[..., 0])
        res = res.reshape(-1, k, 256, 256)
        return res.mean(1), res.std(1)

    return sample
