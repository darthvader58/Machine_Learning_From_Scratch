"""LDA -> QDA: the one assumption that bends the boundary.

LDA forces a single pooled covariance on every class. That is exactly why the
quadratic term x^T Sigma^-1 x is identical across classes, cancels in the argmax,
and leaves a straight boundary. QDA gives each class its own Sigma_c, the term
survives as -1/2 x^T (Sigma_c^-1 - Sigma_k^-1) x, and the boundary becomes a conic.

This animation walks the assumption continuously,

    Sigma_c(t) = (1 - t) * Sigma_pooled + t * Sigma_c,   t: 0 -> 1

and draws, in the same frame, the class covariance ellipses (the cause) and the
decision regions (the effect), with the size of the surviving quadratic term
printed as a number so the cancellation at t = 0 is visible rather than asserted.

Data, features, classes, colours and both model classes are taken from
classification/lda_qda.ipynb unchanged: UCI Wine, three cultivars, the two
features figure 3 of the notebook uses (flavanoids and colour intensity).

Regenerate with:
    /opt/anaconda3/envs/tf_mps/bin/python animations/mpl_scenes/lda_qda_boundary.py
"""
import os
import sys

import numpy as np
from matplotlib import animation
from ucimlrepo import fetch_ucirepo

# shared look for every animation in this repo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mpl_style import plt, COLORS, FIGSIZE, DPI, save_gif  # noqa: E402

SEED = 42
np.random.seed(SEED)

CLASS_COLOURS = ['#2a78d6', '#eb6834', '#1baf7a']   # notebook's cultivar colours
RIDGE = 1e-6


# ---------------------------------------------------------------- notebook code
# Everything in this block is copied verbatim from classification/lda_qda.ipynb.

def split_data(X, y, test_frac=0.3, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    n_test = int(round(test_frac * len(X)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def standardise(X_train, X_test):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    return (X_train - mu) / sd, (X_test - mu) / sd


def log_det(S):
    sign, logabsdet = np.linalg.slogdet(S)
    return logabsdet


def mahalanobis(X, mu, S):
    # squared Mahalanobis distance of every row of X from mu under covariance S
    diff = X - mu
    z = np.linalg.solve(S, diff.T).T
    return np.sum(diff * z, axis=1)


def class_covariance(Xc, mu):
    diff = Xc - mu
    return diff.T @ diff / max(len(Xc) - 1, 1)


class QDA:

    def __init__(self, ridge=RIDGE, diagonal=False):
        self.ridge = ridge
        self.diagonal = diagonal

    def fit(self, X, y):
        self.classes = np.unique(y)
        n, d = X.shape
        self.means = np.zeros((len(self.classes), d))
        self.covs = np.zeros((len(self.classes), d, d))
        self.priors = np.zeros(len(self.classes))

        for i, c in enumerate(self.classes):
            Xc = X[y == c]
            mu = Xc.mean(axis=0)
            S = class_covariance(Xc, mu)
            if self.diagonal:
                S = np.diag(np.diag(S))          # Gaussian Naive Bayes
            S = S + self.ridge * np.eye(d)       # guard the inversion
            self.means[i] = mu
            self.covs[i] = S
            self.priors[i] = len(Xc) / n

        self.log_dets = np.array([log_det(S) for S in self.covs])
        return self

    def decision_scores(self, X):
        scores = np.zeros((len(X), len(self.classes)))
        for i in range(len(self.classes)):
            m2 = mahalanobis(X, self.means[i], self.covs[i])
            scores[:, i] = -0.5 * self.log_dets[i] - 0.5 * m2 + np.log(self.priors[i])
        return scores

    def predict(self, X):
        return self.classes[np.argmax(self.decision_scores(X), axis=1)]


class LDA:

    def __init__(self, ridge=RIDGE, n_components=None):
        self.ridge = ridge
        self.n_components = n_components

    def fit(self, X, y):
        self.classes = np.unique(y)
        n, d = X.shape
        k = len(self.classes)

        self.means = np.zeros((k, d))
        self.priors = np.zeros(k)
        self.overall_mean = X.mean(axis=0)

        Sw = np.zeros((d, d))

        for i, c in enumerate(self.classes):
            Xc = X[y == c]
            mu = Xc.mean(axis=0)
            diff = Xc - mu
            Sw += diff.T @ diff                                   # within-class scatter
            self.means[i] = mu
            self.priors[i] = len(Xc) / n

        # pooled covariance: within-class scatter divided by N - C, plus ridge
        self.covariance = Sw / (n - k) + self.ridge * np.eye(d)

        # linear discriminant: w_c = Sigma^-1 mu_c, b_c = -0.5 mu_c^T Sigma^-1 mu_c + log pi_c
        self.weights = np.linalg.solve(self.covariance, self.means.T).T
        self.biases = -0.5 * np.sum(self.means * self.weights, axis=1) + np.log(self.priors)
        return self

    def decision_scores(self, X):
        return X @ self.weights.T + self.biases

    def predict(self, X):
        return self.classes[np.argmax(self.decision_scores(X), axis=1)]


def cov_ellipse(mu, S, n_std=2.0, points=160):
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, 0.0)
    t = np.linspace(0, 2 * np.pi, points)
    circle = np.column_stack([np.cos(t), np.sin(t)])
    pts = circle @ (vecs * np.sqrt(vals)).T * n_std
    return pts + mu


# ------------------------------------------------------------------- the data
wine = fetch_ucirepo(id=109)
feature_names = list(wine.data.features.columns)
X_all = wine.data.features.to_numpy(dtype=float)
y_raw = wine.data.targets.to_numpy().ravel().astype(int)
classes = np.unique(y_raw)
y_all = np.searchsorted(classes, y_raw)
class_labels = ['Cultivar %d' % c for c in classes]
C = len(classes)

X_train_raw, y_train, X_test_raw, y_test = split_data(X_all, y_all)
X_train, X_test = standardise(X_train_raw, X_test_raw)

# the same two features the notebook draws its decision regions on
f1 = feature_names.index('Flavanoids')
f2 = feature_names.index('Color_intensity')
pair_train = X_train[:, [f1, f2]]

lda2 = LDA().fit(pair_train, y_train)
qda2 = QDA().fit(pair_train, y_train)

POOLED = lda2.covariance          # the one covariance LDA imposes on every class
PER_CLASS = qda2.covs             # the covariance QDA lets each class keep


def model_at(t):
    """QDA with Sigma_c blended toward the pooled covariance.

    t = 0 gives every class the same Sigma, so every -1/2 x^T Sigma^-1 x term is
    identical and cancels in the argmax -- that is LDA exactly. t = 1 is QDA.

    The cancellation at t = 0 is exact, not approximate. POOLED and PER_CLASS both
    already carry the notebook's ridge eps*I (eps = 1e-6) from their own fits, so
    the blend carries it linearly too and Sigma_c(0) comes out bit-for-bit equal to
    the pooled matrix for every class. The reported Frobenius norm is therefore a
    true 0.00 at t = 0; any non-zero reading means t has already left zero.
    The classifier itself is the notebook's QDA, untouched; only the stored
    covariances are swapped.
    """
    m = QDA()
    m.classes = qda2.classes
    m.means = qda2.means
    m.priors = qda2.priors
    m.covs = np.array([(1.0 - t) * POOLED + t * S for S in PER_CLASS])
    m.log_dets = np.array([log_det(S) for S in m.covs])
    return m


def quadratic_term_size(m):
    """Largest ||Sigma_c^-1 - Sigma_k^-1||_F over class pairs.

    This is the coefficient matrix of the term that survives in the difference of
    two discriminants. Zero means the quadratic parts cancel and the boundary is
    a straight line; anything above zero means it is a conic.
    """
    invs = [np.linalg.inv(S) for S in m.covs]
    return max(np.linalg.norm(invs[i] - invs[j], 'fro')
               for i in range(len(invs)) for j in range(i + 1, len(invs)))


# ------------------------------------------------------------ the frame schedule
# smoothstep so the motion eases in and out instead of starting with a jolt
N_MOVE = 34
raw = np.linspace(0.0, 1.0, N_MOVE)
eased = raw * raw * (3.0 - 2.0 * raw)
T_VALUES = np.concatenate([np.zeros(8),        # hold on LDA so the line reads
                           eased,
                           np.ones(9)])        # hold on QDA so the loop does not snap

# fixed grid and limits, so the frames differ only where the boundary moves
PAD = 0.6
x_min, x_max = pair_train[:, 0].min() - PAD, pair_train[:, 0].max() + PAD
y_min, y_max = pair_train[:, 1].min() - PAD, pair_train[:, 1].max() + PAD
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 260),
                     np.linspace(y_min, y_max, 200))
grid = np.column_stack([xx.ravel(), yy.ravel()])

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
fig.subplots_adjust(left=0.10, right=0.98, top=0.80, bottom=0.16)


# headline and footnote are static, so they are drawn once rather than per frame
fig.suptitle('One shared $\\Sigma$ or one per class: $\\;'
         '\\Sigma_c(t)=(1-t)\\,\\bar\\Sigma+t\\,\\Sigma_c$', fontsize=12, y=0.975)
fig.text(0.5, 0.855,
         '$\\delta_c(\\mathbf{x})-\\delta_k(\\mathbf{x})='
         '-\\frac{1}{2}\\mathbf{x}^\\top(\\Sigma_c^{-1}-\\Sigma_k^{-1})\\mathbf{x}'
         '+\\mathbf{w}^\\top\\mathbf{x}+b$',
         ha='center', fontsize=10.5, color='#333333')
fig.text(0.5, 0.020,
         'cost of dropping the shared $\\Sigma$: 3 covariances to estimate, not 1 '
         '(273 vs 91 parameters on all 13 features)',
         ha='center', fontsize=8.5, color='#666666')


def draw(frame):
    t = T_VALUES[frame]
    m = model_at(t)
    zz = m.predict(grid).reshape(xx.shape)
    q = quadratic_term_size(m)

    ax.clear()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # decision regions, and the boundaries themselves in black
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=CLASS_COLOURS, alpha=0.20)
    ax.contour(xx, yy, zz, levels=[0.5, 1.5], colors='k', linewidths=1.6)

    # the cause, drawn beside the effect: each class's covariance at this t
    for c in range(C):
        msk = y_train == c
        ax.scatter(pair_train[msk, 0], pair_train[msk, 1], s=13, alpha=0.55,
                   color=CLASS_COLOURS[c], edgecolor='none', zorder=3)
        e = cov_ellipse(m.means[c], m.covs[c])
        ax.plot(e[:, 0], e[:, 1], color=CLASS_COLOURS[c], linewidth=2.0, zorder=4)
        ax.scatter([m.means[c][0]], [m.means[c][1]], marker='X', s=55,
                   color=CLASS_COLOURS[c], edgecolor='k', linewidth=0.6, zorder=5)

    ax.set_xlabel('Flavanoids (standardised)', fontsize=9)
    ax.set_ylabel('Color intensity (standardised)', fontsize=9)
    ax.tick_params(labelsize=8)

    # The caption has to match the frame it sits on, so it is read off q rather
    # than fixed: exactly zero while the covariances are still shared, a middle
    # state while the term is small and the bend is only just visible, and the
    # full conic statement once it dominates.
    if q == 0.0:
        verdict = 'quadratic terms identical $\\Rightarrow$ cancel $\\Rightarrow$ straight'
        box_col = CLASS_COLOURS[0]
    elif q < 1.0:
        verdict = 'quadratic term switching on $\\Rightarrow$ boundary starts to bend'
        box_col = COLORS[3]
    else:
        verdict = 'quadratic term survives $\\Rightarrow$ conic'
        box_col = CLASS_COLOURS[1]
    ax.text(0.025, 0.035,
            '$\\max_{c,k}\\;\\|\\Sigma_c^{-1}-\\Sigma_k^{-1}\\|_F = %.2f$\n%s' % (q, verdict),
            transform=ax.transAxes, fontsize=9, va='bottom', color='#222222', zorder=10,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor=box_col, linewidth=1.2))

    # which end of the interpolation we are at
    name = 'LDA' if t == 0.0 else ('QDA' if t == 1.0 else 'blend')
    ax.text(0.975, 0.965, '%s      t = %.3f' % (name, t), transform=ax.transAxes,
            fontsize=11, ha='right', va='top', color='#222222', zorder=10,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='#bbbbbb'))

    return []


def main():
    out = os.path.join(os.path.dirname(__file__), '..', 'gifs', 'lda_qda_boundary.gif')
    out = os.path.normpath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    anim = animation.FuncAnimation(fig, draw, frames=len(T_VALUES), interval=1000 / 12)
    save_gif(anim, out, fps=12)
    print('frames:', len(T_VALUES))


if __name__ == '__main__':
    main()
