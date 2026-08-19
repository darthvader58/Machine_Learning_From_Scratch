"""A linear autoencoder rotating onto the PCA subspace, from `neural_networks/autoencoder.ipynb`.

The notebook's headline result is that a linear autoencoder with bottleneck $k$ reaches
PCA's reconstruction error and spans the same $k$-dimensional subspace. This replays that
run: the `Autoencoder` class is copied from the notebook, the data is the same UCI digits
loaded and split the same way (seed 0), and the model is the same one the notebook trains
for k = 2 - `Autoencoder([64, 2, 64], learning_rate=0.006, activation=False, seed=0)` fitted
for 200 epochs. The only change to the class is a `self.frames` list appended to inside
`fit`; it stores the decoder and the loss after every epoch so the animation can replay the
real training run. It does not touch the maths.

Everything measured is measured in the full 64-dimensional pixel space. The picture is a
3-D view of it: the digits are drawn in the coordinate frame of the first three principal
components, where the PCA plane the model is chasing is exactly the horizontal PC1-PC2
plane (grey, fixed). The decoder's learned subspace is a 2-D subspace of R^64; its shadow
in that same three-component frame is the orange plane, and it swings down onto the grey
one as training proceeds. The two side panels track the real numbers: the reconstruction
error falling onto PCA's optimum, and the largest principal angle between the two
subspaces falling from ~86 degrees to the 4.19 degrees the notebook reports for k = 2.

The point written across the bottom is the one the notebook is careful about: the
autoencoder does not recover the principal components, only their span. Any invertible
2x2 A gives W_d W_e = (W_d A^-1)(A W_e), so the axes inside the subspace are free. That is
why the comparison is principal angles between subspaces and not W_d against V_k.

Run from the repo root:
    /opt/anaconda3/envs/tf_mps/bin/python animations/mpl_scenes/autoencoder_pca.py
"""
import os
import sys

import numpy as np

# animations/mpl_style.py holds the look shared by every GIF in this repo
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mpl_style import COLORS, GREY, FIGSIZE, DPI, save_gif  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402  (registers the 3d projection)
from ucimlrepo import fetch_ucirepo  # noqa: E402

np.random.seed(0)          # every seed is fixed so the GIF is reproducible


# ---------------------------------------------------------------------------
# Data - the notebook's loading cell, unchanged
# ---------------------------------------------------------------------------
digits = fetch_ucirepo(id=80)

X_all = digits.data.features.values.astype(float) / 16.0
y_all = digits.data.targets.values.ravel()

rng = np.random.default_rng(0)
order = rng.permutation(len(X_all))
split = int(0.8 * len(X_all))

train_idx, test_idx = order[:split], order[split:]
X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]


def recon_loss(X, R):
    # mean over images of the squared distance summed over the 64 pixels
    return np.mean(np.sum((X - R) ** 2, axis=1))


# ---------------------------------------------------------------------------
# The notebook's class, copied across. The only addition is the block marked
# "recorded for the animation" - it saves the decoder after every epoch.
# ---------------------------------------------------------------------------
class Autoencoder:
    def __init__(self, layer_sizes, learning_rate=0.006, activation=True, seed=0):
        r = np.random.default_rng(seed)

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        self.middle = len(layer_sizes) // 2
        self.weights = []
        self.biases = []
        self.history = {'train_loss': [], 'test_loss': []}
        self.frames = []

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He initialisation: sqrt(2 / fan_in)
            self.weights.append(r.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out))

        # Adam moment estimates, one pair per parameter array
        self.mw = [np.zeros_like(w) for w in self.weights]
        self.vw = [np.zeros_like(w) for w in self.weights]
        self.mb = [np.zeros_like(b) for b in self.biases]
        self.vb = [np.zeros_like(b) for b in self.biases]
        self.step = 0

    def _act(self, z, i):
        # no ReLU at the output layer or at the bottleneck
        if not self.activation or i == len(self.weights) - 1 or i == self.middle - 1:
            return z
        return np.maximum(0, z)

    def forward(self, X):
        self.activations = [X]
        self.zs = []
        a = X

        for i in range(len(self.weights)):
            z = a @ self.weights[i] + self.biases[i]
            a = self._act(z, i)
            self.zs.append(z)
            self.activations.append(a)

        return a

    def backward(self, output, target):
        n = len(target)
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # squared error through a linear output layer
        delta = 2.0 * (output - target) / n

        for i in range(len(self.weights) - 1, -1, -1):
            grads_w[i] = self.activations[i].T @ delta
            grads_b[i] = delta.sum(axis=0)

            if i > 0:
                delta = delta @ self.weights[i].T
                # ReLU derivative, unless that layer had no activation
                if self.activation and (i - 1) != self.middle - 1:
                    delta = delta * (self.zs[i - 1] > 0)

        return grads_w, grads_b

    def update(self, grads_w, grads_b, lr):
        self.step += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        c1 = 1 - b1 ** self.step
        c2 = 1 - b2 ** self.step

        for i in range(len(self.weights)):
            self.mw[i] = b1 * self.mw[i] + (1 - b1) * grads_w[i]
            self.vw[i] = b2 * self.vw[i] + (1 - b2) * grads_w[i] ** 2
            self.weights[i] -= lr * (self.mw[i] / c1) / (np.sqrt(self.vw[i] / c2) + eps)

            self.mb[i] = b1 * self.mb[i] + (1 - b1) * grads_b[i]
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * grads_b[i] ** 2
            self.biases[i] -= lr * (self.mb[i] / c1) / (np.sqrt(self.vb[i] / c2) + eps)

    def fit(self, X, X_val=None, epochs=150, batch_size=128, noise=0.0, seed=0, report_every=0):
        r = np.random.default_rng(seed)

        for e in range(epochs):
            # hold the step size, then shrink it twice near the end
            lr = self.learning_rate * (0.2 if e > 0.85 * epochs else 0.5 if e > 0.6 * epochs else 1.0)
            idx = r.permutation(len(X))

            for s in range(0, len(X), batch_size):
                target = X[idx[s:s + batch_size]]
                # the denoising variant corrupts the input only
                inputs = target if noise == 0 else np.clip(target + r.normal(0, noise, target.shape), 0, 1)
                grads_w, grads_b = self.backward(self.forward(inputs), target)
                self.update(grads_w, grads_b, lr)

            self.history['train_loss'].append(recon_loss(X, self.forward(X)))
            if X_val is not None:
                self.history['test_loss'].append(recon_loss(X_val, self.forward(X_val)))

            # --- recorded for the animation, changes nothing above ---
            self.frames.append({'Wd': self.weights[1].copy(),
                                'loss': self.history['train_loss'][-1]})

            if report_every and (e + 1) % report_every == 0:
                print(f'  epoch {e + 1:4d}  train J {self.history["train_loss"][-1]:.5f}')

    def encode(self, X):
        a = X
        for i in range(self.middle):
            a = self._act(a @ self.weights[i] + self.biases[i], i)
        return a

    def decode(self, Z):
        a = Z
        for i in range(self.middle, len(self.weights)):
            a = self._act(a @ self.weights[i] + self.biases[i], i)
        return a

    def parameter_count(self):
        return sum(w.size for w in self.weights) + sum(b.size for b in self.biases)


# ---------------------------------------------------------------------------
# PCA, written out the way the notebook writes it
# ---------------------------------------------------------------------------
pca_mean = X_train.mean(axis=0)
U, S, Vt = np.linalg.svd(X_train - pca_mean, full_matrices=False)
components = Vt.T

K = 2                                       # the bottleneck size being animated
Vk = components[:, :K]
pca_J = recon_loss(X_train, pca_mean + (X_train - pca_mean) @ Vk @ Vk.T)


def principal_angles(A, B):
    # columns of A and B span the two subspaces
    Q1, _ = np.linalg.qr(A)
    Q2, _ = np.linalg.qr(B)
    cosines = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return np.degrees(np.arccos(np.clip(cosines, -1, 1))), Q1


# ---------------------------------------------------------------------------
# Train - exactly the model the notebook trains at k = 2
# ---------------------------------------------------------------------------
ae = Autoencoder([64, K, 64], learning_rate=0.006, activation=False, seed=0)
Wd_init = ae.weights[1].copy()              # the He-initialised decoder, before any step
ae.fit(X_train, X_test, epochs=200, batch_size=128, seed=0)

# Per epoch: where the decoder subspace sits, and how far it is from the PCA subspace.
# The angles are measured in the full 64-dimensional space; only the drawing is 3-D.
P3 = components[:, :3]                      # the frame the picture is drawn in


def read_decoder(Wd):
    """Orthonormal basis of the decoder's row space, its angles to the PCA subspace,
    and the shadow of that basis in the first three principal components."""
    angles, Q1 = principal_angles(Wd.T, Vk)
    return angles, P3.T @ Q1                # 3 x 2 coordinates of the two basis vectors


states = []
for snap in [{'Wd': Wd_init, 'loss': recon_loss(X_train, X_train @ ae.weights[0] @ Wd_init)}] + ae.frames:
    angles, C3 = read_decoder(snap['Wd'])
    states.append({'loss': snap['loss'], 'angles': angles, 'C3': C3})

final_angles = states[-1]['angles']
print(f"final: J {states[-1]['loss']:.5f}   PCA J {pca_J:.5f}"
      f"   largest angle {final_angles.max():.4f}d   mean {final_angles.mean():.4f}d")


# ---------------------------------------------------------------------------
# Plane geometry. A 2-D subspace of R^64 casts a 2-D shadow in the three-component
# frame; its normal there is the cross product of the two shadowed basis vectors.
# The sign of the normal is kept continuous so the drawn patch does not flip.
# ---------------------------------------------------------------------------
def plane_quad(C3, radius, prev_normal=None):
    n = np.cross(C3[:, 0], C3[:, 1])
    n = n / np.linalg.norm(n)
    if prev_normal is not None and n @ prev_normal < 0:
        n = -n
    # two in-plane directions, built from the normal so they turn with it
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    corners = np.array([u + v, u - v, -u - v, -u + v]) * radius
    return corners, n


# scale the drawing to the spread of the data in the first three components
Z3 = (X_train - pca_mean) @ P3
LIM = 3.0
RAD = 2.6

# a small, fixed sample of digits so the 3-D panel stays readable and the GIF small
sample = np.random.default_rng(2).choice(len(Z3), 170, replace=False)
pts3 = Z3[sample]


# ---------------------------------------------------------------------------
# Frame plan: epochs sampled densely early, where the plane swings fastest
# ---------------------------------------------------------------------------
picks = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 23, 27, 31, 35, 40, 45, 50,
         56, 62, 68, 75, 82, 90, 98, 106, 115, 124, 134, 144, 155, 166, 178, 190, 200]
frames = [-1] * 4 + picks                    # a few frames of the data alone first
frames += [frames[-1]] * 6                   # hold the final state so the loop does not snap

# normals precomputed in order so the sign stays continuous along the run
normals = {}
prev = None
for i in picks:
    _, prev = plane_quad(states[i]['C3'], RAD, prev)
    normals[i] = prev


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=FIGSIZE, dpi=DPI)

ax = fig.add_axes([-0.02, 0.135, 0.66, 0.75], projection='3d')
ax.set_xlim(-LIM, LIM)
ax.set_ylim(-LIM, LIM)
ax.set_zlim(-LIM, LIM)
ax.set_box_aspect((1, 1, 0.72))
ax.view_init(elev=20, azim=-58)
ax.set_xlabel('PC1', fontsize=9, labelpad=-9)
ax.set_ylabel('PC2', fontsize=9, labelpad=-9)
ax.set_zlabel('PC3', fontsize=9, labelpad=-9)
ax.tick_params(labelleft=False, labelbottom=False, length=0)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.grid(False)
for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
    pane.pane.set_facecolor('white')
    pane.pane.set_edgecolor('#d8d8d8')

jax = fig.add_axes([0.735, 0.605, 0.235, 0.195])       # reconstruction error
jax.set_ylim(2.9, 15.0)
jax.set_xlim(-6, 206)
jax.tick_params(labelsize=7.5)
jax.set_xticks([0, 100, 200])
jax.set_yticks([5, 10, 15])
jax.set_xlabel('epoch', fontsize=7.5, labelpad=0)

aax = fig.add_axes([0.735, 0.205, 0.235, 0.195])       # largest principal angle
aax.set_ylim(-4, 94)
aax.set_xlim(-6, 206)
aax.tick_params(labelsize=7.5)
aax.set_xticks([0, 100, 200])
aax.set_yticks([0, 45, 90])
aax.set_yticklabels(['0', '45', '90'])
aax.set_xlabel('epoch', fontsize=7.5, labelpad=0)

# the encode / decode path, and what the bottleneck is doing
fig.text(0.035, 0.955,
         r'$\mathbf{x} \;\rightarrow\; \mathbf{h}=W_e\mathbf{x} \;\rightarrow\;'
         r' \hat{\mathbf{x}}=W_d\mathbf{h}$',
         fontsize=12, va='center')
fig.text(0.035, 0.898,
         r'$\dim(\mathbf{h})=2 \;<\; \dim(\mathbf{x})=64$'
         ' - the bottleneck forces the compression',
         fontsize=8.5, color=GREY, va='center')

fig.text(0.665, 0.878, 'reconstruction error', fontsize=8.5, color='#333333')
fig.text(0.665, 0.478, 'principal angles to PCA subspace',
         fontsize=8.5, color='#333333')

t_epoch = fig.text(0.972, 0.955, '', fontsize=9, color='#333333',
                   ha='right', va='center')
t_j = fig.text(0.665, 0.838, '', fontsize=8.5, color=COLORS[1])
t_ang = fig.text(0.665, 0.438, '', fontsize=8.5, color=COLORS[1])

# the subtlety, spelled out
fig.text(0.035, 0.118,
         'the autoencoder does NOT learn the principal components - it learns the same '
         'SUBSPACE,\nup to rotation and scaling inside it '
         r'($W_dW_e=(W_dA^{-1})(AW_e)$ for any invertible $A$),'
         '\nwhich is why the notebook compares principal ANGLES and not weight matrices.',
         fontsize=8, color='#444444', va='top', linespacing=1.35)

# artists in the 3-D panel
ax.scatter(pts3[:, 0], pts3[:, 1], pts3[:, 2], s=4, c=GREY, alpha=0.45,
           depthshade=False, linewidths=0)

# the PCA plane: span of the first two components, i.e. PC3 = 0. Fixed reference.
g = np.array([-RAD, RAD])
GX, GY = np.meshgrid(g, g)
ax.plot_surface(GX, GY, np.zeros_like(GX), color=COLORS[0], alpha=0.16,
                shade=False, zorder=1)
ax.plot(np.append(GX.ravel()[[0, 1, 3, 2]], GX.ravel()[0]),
        np.append(GY.ravel()[[0, 1, 3, 2]], GY.ravel()[0]),
        np.zeros(5), color=COLORS[0], lw=1.2, alpha=0.9)

learned = [None]                     # the moving plane, replaced each frame

fig.text(0.055, 0.825, 'PCA plane, span$(v_1, v_2)$', fontsize=8.5, color=COLORS[0])
fig.text(0.055, 0.775, 'decoder subspace, row space of $W_d$', fontsize=8.5, color=COLORS[1])


def draw(idx):
    e = frames[idx]

    if learned[0] is not None:
        learned[0].remove()
        learned[0] = None

    if e < 0:
        # the data alone, with the PCA plane it is about to be compared against
        t_epoch.set_text('5,620 digits, 64 pixels each')
        jax.set_prop_cycle(None)
        t_j.set_text('')
        t_ang.set_text('')
        for line in list(jax.lines) + list(aax.lines):
            line.remove()
        jax.axhline(pca_J, color=COLORS[0], lw=1.3, ls='--')
        jax.text(198, pca_J + 1.5, 'PCA optimum', fontsize=7.5, color=COLORS[0],
                 ha='right')
        aax.axhline(0, color=COLORS[0], lw=1.3, ls='--')
        return

    st = states[e]
    corners, _ = plane_quad(st['C3'], RAD, normals[e])
    # the decoder's subspace, drawn as a patch through the data mean
    quad = np.array([corners[0], corners[1], corners[2], corners[3]])
    learned[0] = ax.plot_surface(
        np.array([[quad[0, 0], quad[1, 0]], [quad[3, 0], quad[2, 0]]]),
        np.array([[quad[0, 1], quad[1, 1]], [quad[3, 1], quad[2, 1]]]),
        np.array([[quad[0, 2], quad[1, 2]], [quad[3, 2], quad[2, 2]]]),
        color=COLORS[1], alpha=0.34, shade=False, zorder=5)

    # curves up to this epoch
    es = np.arange(0, e + 1)
    losses = [states[i]['loss'] for i in range(e + 1)]
    angs = [states[i]['angles'].max() for i in range(e + 1)]

    for line in list(jax.lines)[1:]:
        line.remove()
    for line in list(aax.lines)[1:]:
        line.remove()
    jax.plot(es, losses, color=COLORS[1], lw=1.5)
    jax.plot([e], [losses[-1]], color=COLORS[1], marker='o', ms=4.5)
    aax.plot(es, angs, color=COLORS[1], lw=1.5)
    aax.plot([e], [angs[-1]], color=COLORS[1], marker='o', ms=4.5)

    t_epoch.set_text(f'[64, 2, 64] linear AE,  epoch {e} / 200')
    t_j.set_text(f'$J$ = {losses[-1]:.3f}       PCA optimum {pca_J:.3f}')
    t_ang.set_text(f'largest {angs[-1]:.2f}$\\degree$      mean '
                   f'{st["angles"].mean():.2f}$\\degree$')


anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / 12)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gifs',
                   'autoencoder_pca.gif')
save_gif(anim, os.path.normpath(out), fps=12)
print('frames:', len(frames))
