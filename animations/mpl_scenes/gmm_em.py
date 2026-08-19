"""EM alternating its two steps on the Wine GMM, and why log L can only go up.

Replays `clustering/gmm.ipynb`. The `kmeans`, `logsumexp`, `log_gaussian` and `GMM`
code below is copied out of the notebook unchanged; the only addition is a `self.snaps`
list recorded alongside the existing `self.history`, holding the responsibilities and
the parameters either side of every M-step so the animation can replay them. The maths
is untouched.

Same data as the notebook: UCI Wine (id=109), 178 wines, all 13 features standardised,
C = 3, `covariance_type='full'`, seed 42, means and mixing weights initialised from
k-means exactly as `_initialise` does.

One deliberate change to the start, and only to the start. The notebook seeds each
covariance with the within-cluster scatter, which is already so close to the answer
that EM has nothing left to do - the log-likelihood moves by 0.28 in total and no
ellipse visibly shifts. Here every covariance instead starts as the identity, the unit
sphere, which is the shape k-means implicitly assumes and the limit the notebook
derives in its k-means section. That start is uninformative, so the fit is worth
watching, and it lands on the same local optimum: log L = -2089.5903 against the
notebook's -2089.5904, with means and covariances agreeing to 3e-6 and the mixing
weights identical to four decimals. The animation therefore converges to the fit the
notebook reports.

The picture is the Flavanoids / Color_intensity plane, the pair of columns the notebook
plots. Every Gaussian is still 13-dimensional; the ellipses are the 2x2 marginal
submatrices of the fitted 13x13 covariances, which is exactly what the notebook draws.

Each iteration is two visually separate beats:

  1. E-STEP - every point is recoloured by its responsibility vector, blending the
     three component colours in proportion to gamma_i0, gamma_i1, gamma_i2. A point
     half-owned by two components comes out a genuine mixture, not one colour or the
     other. Points still below 0.99 confidence are ringed. This soft ownership is the
     thing k-means cannot represent.
  2. M-STEP - the colours hold and each component refits to its weighted points: the
     1-sigma and 2-sigma ellipses slide from the unit circle onto the fitted shape and
     the mixing weights pi_k redraw as a stacked bar.

The panel on the right tracks log L per iteration. It rises at every single step and
never falls, which is the one guarantee EM provides.

Run from the repo root:
    /opt/anaconda3/envs/tf_mps/bin/python animations/mpl_scenes/gmm_em.py
"""
import os
import sys

import numpy as np

# animations/mpl_style.py holds the look shared by every GIF in this repo
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mpl_style import COLORS, GREY, FIGSIZE, DPI, save_gif  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from ucimlrepo import fetch_ucirepo  # noqa: E402

SEED = 42
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# The notebook's code, copied across verbatim.
# ---------------------------------------------------------------------------
def kmeans(X, k, seed=SEED, n_iter=300):
    rng = np.random.RandomState(seed)
    # k-means++ seeding
    centres = [X[rng.randint(len(X))]]
    for _ in range(k - 1):
        d2 = ((X[:, None, :] - np.array(centres)[None, :, :]) ** 2).sum(2).min(1)
        centres.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    centres = np.array(centres)

    labels = np.zeros(len(X), dtype=int)
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(2)
        labels = np.argmin(d2, axis=1)
        new = np.array([X[labels == c].mean(0) if np.any(labels == c) else centres[c]
                        for c in range(k)])
        if np.allclose(new, centres):
            break
        centres = new

    inertia = ((X - centres[labels]) ** 2).sum()
    return centres, labels, inertia


def logsumexp(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))


def log_gaussian(X, mean, cov):
    D = X.shape[1]
    L = np.linalg.cholesky(cov)
    diff = X - mean
    sol = np.linalg.solve(L, diff.T)              # L^-1 (x - mu), shape (D, N)
    maha = (sol ** 2).sum(0)                      # squared Mahalanobis distance
    logdet = 2.0 * np.sum(np.log(np.diag(L)))     # log|Sigma|
    return -0.5 * (D * np.log(2 * np.pi) + logdet + maha)


class GMM:
    # the notebook's signature minus `init`, which had only two settings and is fixed
    # to the k-means one here
    def __init__(self, n_components, covariance_type='full', max_iter=300,
                 tol=1e-7, reg=1e-6, seed=SEED):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg
        self.seed = seed

    # constrain a covariance to the chosen type and add the ridge
    def _shape(self, S):
        D = S.shape[0]
        if self.covariance_type == 'diagonal':
            S = np.diag(np.diag(S))
        elif self.covariance_type == 'spherical':
            S = np.eye(D) * np.mean(np.diag(S))
        return S + self.reg * np.eye(D)

    def _initialise(self, X):
        N, D = X.shape
        C = self.n_components
        centres, labels, _ = kmeans(X, C, seed=self.seed)

        # means from the centroids, weights from the cluster proportions
        self.means = centres.copy()
        counts = np.array([max(np.sum(labels == c), 1) for c in range(C)], dtype=float)
        self.weights = counts / counts.sum()

        # the one change to the notebook: every covariance starts as the unit sphere,
        # the shape k-means implicitly assumes, instead of the within-cluster scatter
        self.covariances = np.array([self._shape(np.eye(D)) for _ in range(C)])

    # log N(x_i | mu_c, Sigma_c) for every point and component, shape (N, C)
    def _log_gauss(self, X):
        return np.column_stack([log_gaussian(X, self.means[c], self.covariances[c])
                                for c in range(self.n_components)])

    # E-step: responsibilities and the total log-likelihood
    def _e_step(self, X):
        log_p = np.log(self.weights)[None, :] + self._log_gauss(X)
        log_norm = logsumexp(log_p, axis=1)        # per-point log-likelihood
        resp = np.exp(log_p - log_norm)
        return resp, float(log_norm.sum())

    # M-step: responsibility-weighted weights, means and covariances
    def _m_step(self, X, resp):
        N, D = X.shape
        Nc = resp.sum(axis=0) + 1e-12              # effective points per component
        self.weights = Nc / N
        self.means = (resp.T @ X) / Nc[:, None]
        for c in range(self.n_components):
            diff = X - self.means[c]
            S = (resp[:, c, None] * diff).T @ diff / Nc[c]
            self.covariances[c] = self._shape(S)

    def fit(self, X):
        self._initialise(X)
        self.history = []
        self.snaps = []                            # recorded for the animation only
        prev = -np.inf
        for _ in range(self.max_iter):
            resp, ll = self._e_step(X)
            self.history.append(ll)

            # --- recorded for the animation, changes nothing above ---
            before = (self.means.copy(), self.covariances.copy(), self.weights.copy())
            self._m_step(X, resp)
            self.snaps.append({
                'resp': resp,
                'll': ll,
                'before': before,
                'after': (self.means.copy(), self.covariances.copy(),
                          self.weights.copy()),
            })

            if abs(ll - prev) < self.tol * abs(ll):
                break
            prev = ll
        # one final E-step so the stored responsibilities match the final parameters
        self.resp, self.loglik = self._e_step(X)
        self.history.append(self.loglik)
        self.history = np.array(self.history)
        self.n_iter = len(self.history)

        # EM never decreases the log-likelihood
        steps = np.diff(self.history)
        assert steps.min() > -1e-6 * abs(self.loglik), 'log-likelihood decreased'
        return self


def covariance_ellipse(mean2, cov2, r=1.0, n=90):
    """Contour at Mahalanobis radius r, from the eigen-decomposition of the 2x2 block.

    The notebook draws its 95% contour at r^2 = -2 log(0.05) = 5.99; here r is passed
    directly so the 1-sigma and 2-sigma rings can both be drawn.
    """
    vals, vecs = np.linalg.eigh(cov2)             # ascending eigenvalues
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.column_stack([np.cos(t), np.sin(t)])
    pts = circle * (r * np.sqrt(np.maximum(vals, 0)))[None, :]
    return pts @ vecs.T + mean2                   # rotate and translate


# ---------------------------------------------------------------------------
# Data and fit - identical to the notebook apart from the covariance start
# ---------------------------------------------------------------------------
wine = fetch_ucirepo(id=109)
feature_names = list(wine.data.features.columns)
X_raw = wine.data.features.to_numpy(dtype=float)
X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)     # standardise every column
N, D = X.shape

C = 3
gmm = GMM(C, covariance_type='full').fit(X)
snaps = gmm.snaps
n_it = len(snaps)

FI = feature_names.index('Flavanoids')
FJ = feature_names.index('Color_intensity')
px, py = X[:, FI], X[:, FJ]

# the three component colours as RGB, so a responsibility vector can be turned into a
# colour by mixing them: blend_i = sum_k gamma_ik * rgb_k
RGB = np.array([[int(COLORS[c][i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
                for c in range(C)])


def marginal(cov, c):
    """The 2x2 covariance of the two plotted features, read off the 13x13 matrix."""
    return cov[c][np.ix_([FI, FJ], [FI, FJ])]


# ---------------------------------------------------------------------------
# Frame plan: every iteration is an E-STEP beat then an M-STEP slide. The first
# iteration moves from unit spheres to fitted ellipses so it gets the most frames;
# by the fourth the fit has stopped moving and one frame each is plenty.
# ---------------------------------------------------------------------------
frames = [('intro', 0, 0.0)] * 3
for p in range(n_it):
    if p == 0:
        n_e, n_m = 3, 6
    elif p == 1:
        n_e, n_m = 2, 3
    elif p == 2:
        n_e, n_m = 1, 2
    else:
        n_e, n_m = 1, 1
    frames += [('estep', p, 1.0)] * n_e
    frames += [('mstep', p, (s + 1) / n_m) for s in range(n_m)]
frames += [frames[-1]] * 6         # hold the final state so the loop does not snap


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=FIGSIZE, dpi=DPI)

ax = fig.add_axes([0.075, 0.195, 0.505, 0.555])       # the two plotted features
ax.set_xlim(-2.1, 3.5)
ax.set_ylim(-2.0, 3.8)
ax.set_xlabel('Flavanoids (standardised)', fontsize=9, labelpad=1)
ax.set_ylabel('Color_intensity', fontsize=9, labelpad=1)
ax.tick_params(labelsize=9)
ax.set_xticks([-2, 0, 2])
ax.set_yticks([-2, 0, 2])

lax = fig.add_axes([0.705, 0.470, 0.265, 0.280])      # log-likelihood per iteration
lax.set_xlim(0.4, n_it + 0.6)
lax.set_ylim(-3020, -2010)
lax.set_xlabel('EM iteration', fontsize=9, labelpad=1)
lax.tick_params(labelsize=9)
lax.set_xticks([1, 5, 10])
lax.set_yticks([-3000, -2500, -2089])
lax.set_yticklabels(['-3000', '-2500', '-2090'])

bax = fig.add_axes([0.705, 0.225, 0.265, 0.048])      # mixing weights, stacked
bax.set_xlim(0, 1)
bax.set_ylim(0, 1)
bax.set_xticks([])
bax.set_yticks([])
bax.set_facecolor('white')

# the two steps, written out. The active one is boxed and darkened each frame.
EQ_E = (r'$\bf{E}$:  $\gamma_{ik}=\dfrac{\pi_k\,\mathcal{N}'
        r'(x_i\mid\mu_k,\Sigma_k)}{\sum_j \pi_j\,\mathcal{N}(x_i\mid\mu_j,\Sigma_j)}$'
        r'    $\sum_k \gamma_{ik}=1$')
EQ_M = (r'$\bf{M}$:  $N_k=\sum_i \gamma_{ik}$,   $\pi_k=N_k/N$,   '
        r'$\mu_k=\frac{1}{N_k}\sum_i \gamma_{ik}x_i$,   '
        r'$\Sigma_k=\frac{1}{N_k}\sum_i \gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top$')
t_e = fig.text(0.055, 0.945, EQ_E, fontsize=9.5, va='center')
t_m = fig.text(0.055, 0.845, EQ_M, fontsize=8.5, va='center')
t_it = fig.text(0.975, 0.945, '', fontsize=9.5, ha='right', va='center')

fig.text(0.8375, 0.782, r'$\log L=\sum_i \log \sum_k \pi_k \mathcal{N}(x_i)$',
         fontsize=8.5, ha='center', va='center')
t_ll = fig.text(0.8375, 0.360, '', fontsize=9.5, ha='center')
fig.text(0.8375, 0.300, r'mixing weights $\pi_k$', fontsize=9, ha='center')
t_pi = [fig.text(0.0, 0.176, '', fontsize=8.5, ha='center', color=COLORS[c])
        for c in range(C)]
fig.text(0.8375, 0.126,
         "log L rises at every\nstep and never falls -\nthat is the guarantee\nEM provides.",
         fontsize=8.5, ha='center', va='top', color=GREY)

t_soft = fig.text(0.077, 0.075, '', fontsize=8.5, color='#333333')
fig.text(0.077, 0.026,
         'colour = the responsibility vector itself; a blend means genuine doubt',
         fontsize=8.5, color=GREY)

# artists: the wines, the two ellipse rings per component, the component means
pts = ax.scatter(px, py, s=24, c=np.full((N, 3), 0.55), linewidths=0.0, zorder=3)
# a ring drawn round the wines the model is still unsure about, on its own layer so it
# never muddies the blended fill colour underneath
ring_pts = ax.scatter([], [], s=78, facecolors='none', edgecolors='#151515',
                      linewidths=0.9, zorder=6)
rings = [[ax.plot([], [], color=COLORS[c], lw=lw, ls=ls, zorder=4 + (r == 0))[0]
          for r, (lw, ls) in enumerate([(1.8, '-'), (1.0, '--')])]
         for c in range(C)]
mus = ax.scatter(np.zeros(C), np.zeros(C), s=90, marker='X',
                 c=[COLORS[c] for c in range(C)], edgecolors='black',
                 linewidths=1.0, zorder=7)
ax.plot([], [], color=GREY, lw=1.8, ls='-', label=r'$1\sigma$')
ax.plot([], [], color=GREY, lw=1.0, ls='--', label=r'$2\sigma$')
ax.legend(loc='upper right', fontsize=8.5, handlelength=1.6, borderpad=0.3,
          labelspacing=0.25, framealpha=0.85)

lline, = lax.plot([], [], color=COLORS[4], lw=1.6, marker='o', ms=3.5, zorder=3)
ldot, = lax.plot([], [], color=COLORS[1], marker='o', ms=6.5, zorder=4)
bars = [bax.barh(0.5, 0.0, left=0.0, height=1.0, color=COLORS[c],
                 edgecolor='white', linewidth=1.0)[0] for c in range(C)]

HI_BOX = dict(facecolor='white', edgecolor=COLORS[0], boxstyle='round,pad=0.22')


def highlight(active):
    """Box and darken whichever of the two steps is running."""
    for text, name in ((t_e, 'estep'), (t_m, 'mstep')):
        if name == active:
            text.set_color('#222222')
            text.set_bbox(HI_BOX)
        else:
            text.set_color('#b0b0b0')
            text.set_bbox(None)


def draw_shapes(means, covs, weights):
    """Redraw the ellipses, the mean markers and the mixing-weight bar."""
    for c in range(C):
        S = marginal(covs, c)
        m2 = means[c][[FI, FJ]]
        for r, radius in enumerate((1.0, 2.0)):
            e = covariance_ellipse(m2, S, r=radius)
            rings[c][r].set_data(e[:, 0], e[:, 1])
    mus.set_offsets(np.c_[means[:, FI], means[:, FJ]])

    left = 0.0
    for c in range(C):
        bars[c].set_x(left)
        bars[c].set_width(weights[c])
        t_pi[c].set_position((0.705 + 0.265 * (left + weights[c] / 2), 0.176))
        t_pi[c].set_text('%.2f' % weights[c])
        left += weights[c]


def draw(idx):
    kind, p, t = frames[idx]
    means0, covs0, w0 = snaps[p]['before']

    if kind == 'intro':
        # the raw data with every component still a unit sphere
        pts.set_facecolor(np.full((N, 3), 0.55))
        ring_pts.set_offsets(np.empty((0, 2)))
        draw_shapes(means0, covs0, w0)
        lline.set_data([], [])
        ldot.set_data([], [])
        highlight(None)
        t_it.set_text('start: every $\\Sigma_k = I$, the k-means shape')
        t_ll.set_text('')
        t_soft.set_text('')
        return

    resp = snaps[p]['resp']

    # colour every wine by mixing the component colours in proportion to gamma_ik
    pts.set_facecolor(np.clip(resp @ RGB, 0, 1))
    unsure = resp.max(axis=1) < 0.99

    if kind == 'estep':
        # ring the wines the model is not yet sure about; the shapes hold still
        ring_pts.set_offsets(np.c_[px[unsure], py[unsure]])
        draw_shapes(means0, covs0, w0)
        highlight('estep')
        label = 'E-STEP'
    else:
        # the colours hold; each component refits to its responsibility-weighted points
        ring_pts.set_offsets(np.empty((0, 2)))
        means1, covs1, w1 = snaps[p]['after']
        ease = 0.5 - 0.5 * np.cos(np.pi * t)
        draw_shapes(means0 + ease * (means1 - means0),
                    covs0 + ease * (covs1 - covs0),
                    w0 + ease * (w1 - w0))
        highlight('mstep')
        label = 'M-STEP'

    # the log-likelihood readings taken so far, one per completed E-step
    ys = [snaps[q]['ll'] for q in range(p + 1)]
    lline.set_data(range(1, p + 2), ys)
    ldot.set_data([p + 1], [ys[-1]])

    t_it.set_text('iteration %d / %d   %s' % (p + 1, n_it, label))
    t_ll.set_text(r'$\log L$ = %.1f' % ys[-1])
    worst = resp[np.argmin(resp.max(axis=1))]
    t_soft.set_text('%d wines below 0.99 certainty   '
                    'least sure: $\\gamma$ = (%.2f, %.2f, %.2f)'
                    % (unsure.sum(), worst[0], worst[1], worst[2]))


anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / 9)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gifs',
                   'gmm_em.gif')
save_gif(anim, os.path.normpath(out), fps=9)
print('frames:', len(frames), ' EM iterations:', n_it,
      ' final log-likelihood: %.4f' % gmm.loglik)
