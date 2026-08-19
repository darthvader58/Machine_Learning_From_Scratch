"""Additive boosting from `classification/xgboost.ipynb`, one round at a time.

Boosting is the opposite of a random forest. A forest builds its trees in
parallel and averages them; a booster builds them in sequence, and every new
tree is fitted to what the trees so far got *wrong*.

The notebook's booster is the second-order (XGBoost) kind, so "what it got
wrong" is not a plain residual - it is the gradient and hessian of the log loss,

    g_i = p_i - y_i        h_i = p_i (1 - p_i)

and a leaf answers with the Newton step  w* = -G_j / (H_j + lambda)  rather than
with a mean. That is the quantity this animation draws, so the picture matches
the maths in the notebook exactly.

Top    : max heart rate against P(heart disease). Ticks are the training rows
         (top = presence, bottom = absence), grey dots are the observed rate in
         each of 10 equal-count regions, and the purple staircase is the current
         ensemble p_m(x) = sigmoid(F_m(x)).
Bottom : per region, the correction that region still wants, -G/(H + lambda),
         as bars; the newly fitted depth-3 tree h_m(x) drawn on top of them; and
         the shaded band eta*h_m(x), the fraction of it actually added. The bars
         shrink towards zero as the rounds accumulate - that is the ensemble
         running out of error to fit.

Everything is the notebook's: the same UCI Heart Disease rows, the same seed and
split, the same `Regression_Tree` / `XGBoost` classes copied verbatim, and the
same eta = 0.1, max_depth = 3, lambda = 1.0, gamma = 0.05. The only change is
that the model is fitted to one column (`thalach`) instead of all 13, so the
whole ensemble is a curve that fits in one plot.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpl_style import COLORS, GREY, FIGSIZE, DPI, save_gif, plt   # shared look
from matplotlib import animation
from matplotlib.patches import Patch
from ucimlrepo import fetch_ucirepo

np.random.seed(42)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'gifs', 'boosting_residuals.gif')


# ------------------------------------------------- the loss and its two derivatives
# Copied from the notebook. Differentiating the binary log loss with respect to
# the raw score F gives g = p - y and h = p(1 - p).
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def gradients(y, F):
    p = sigmoid(F)
    return p - y, p * (1 - p)


# ------------------------------------------------------------------- the tree
class Regression_Tree:
    """Copied verbatim from the notebook. Fitted to (g, h) rather than to
    labels; each leaf holds the Newton step -G/(H + lambda)."""

    def __init__(self, max_depth=3, lambda_=1.0, gamma=0.0, min_child_weight=1.0):
        self.max_depth = max_depth
        self.lambda_ = lambda_
        self.gamma = gamma
        self.min_child_weight = min_child_weight

    def leaf_weight(self, g, h):
        return -g.sum() / (h.sum() + self.lambda_)

    def similarity(self, g, h):
        return (g.sum() ** 2) / (h.sum() + self.lambda_)

    def fit(self, X, g, h):
        self.n_features = X.shape[1]
        self.gain_by_feature = np.zeros(self.n_features)
        self.n_candidates = 0
        self.n_negative_gain = 0
        self.n_gamma_rejected = 0
        self.n_child_weight_rejected = 0
        self.nodes_split = 0
        self.nodes_stopped_by_gamma = 0
        self.root = self.build(X, g, h, 0)
        return self

    def build(self, X, g, h, depth):
        node = {'leaf': True, 'weight': self.leaf_weight(g, h), 'n': len(g)}
        if depth >= self.max_depth or len(g) < 2:
            return node

        parent = self.similarity(g, h)
        best_gain = -np.inf
        best = None
        best_raw = 0.0

        for j in range(self.n_features):
            col = X[:, j]
            order = np.argsort(col, kind='mergesort')
            cs = col[order]
            gs = np.cumsum(g[order])
            hs = np.cumsum(h[order])
            G, H = gs[-1], hs[-1]

            # split positions only between distinct consecutive values
            cut = np.nonzero(cs[:-1] < cs[1:])[0]
            if len(cut) == 0:
                continue

            GL, HL = gs[cut], hs[cut]
            GR, HR = G - GL, H - HL

            # gain before the gamma penalty
            raw = 0.5 * (GL ** 2 / (HL + self.lambda_)
                         + GR ** 2 / (HR + self.lambda_) - parent)
            heavy = (HL >= self.min_child_weight) & (HR >= self.min_child_weight)

            self.n_candidates += len(cut)
            self.n_child_weight_rejected += int((~heavy).sum())
            self.n_negative_gain += int((heavy & (raw < 0)).sum())
            self.n_gamma_rejected += int((heavy & (raw >= 0) & (raw - self.gamma <= 0)).sum())

            gain = np.where(heavy, raw - self.gamma, -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > best_gain:
                best_gain = float(gain[k])
                best_raw = float(raw[k])
                best = (j, 0.5 * (cs[cut[k]] + cs[cut[k] + 1]))

        if best is None or best_gain <= 0:
            if best is not None and best_raw > 0:
                self.nodes_stopped_by_gamma += 1
            return node

        j, threshold = best
        left = X[:, j] <= threshold
        right = ~left
        self.nodes_split += 1
        self.gain_by_feature[j] += best_gain

        return {'leaf': False, 'feature': j, 'threshold': threshold,
                'gain': best_gain, 'n': len(g),
                'left': self.build(X[left], g[left], h[left], depth + 1),
                'right': self.build(X[right], g[right], h[right], depth + 1)}

    def predict(self, X):
        out = np.empty(len(X))
        for i in range(len(X)):
            node = self.root
            while not node['leaf']:
                if X[i, node['feature']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            out[i] = node['weight']
        return out


# ---------------------------------------------------------------- the booster
class XGBoost:
    """Copied verbatim from the notebook, minus the validation bookkeeping the
    animation does not use. The loop is the whole point: compute (g, h) against
    the current F, fit one tree to them, add eta times it to F."""

    def __init__(self, n_rounds=100, learning_rate=0.3, max_depth=3,
                 lambda_=1.0, gamma=0.0, min_child_weight=1.0):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda_ = lambda_
        self.gamma = gamma
        self.min_child_weight = min_child_weight

    def fit(self, X, y):
        self.trees = []

        # constant start: log-odds of the base rate
        base = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.F0 = np.log(base / (1 - base))

        F = np.full(len(y), self.F0)
        self.train_loss = []

        for t in range(self.n_rounds):
            g, h = gradients(y, F)

            tree = Regression_Tree(self.max_depth, self.lambda_,
                                   self.gamma, self.min_child_weight)
            tree.fit(X, g, h)
            self.trees.append(tree)

            F = F + self.learning_rate * tree.predict(X)
            self.train_loss.append(log_loss(y, sigmoid(F)))

        self.train_loss = np.array(self.train_loss)
        return self


# ------------------------------------------------------------------- the data
# UCI Heart Disease, loaded exactly as the notebook loads it: features and target
# joined before dropping the 6 incomplete rows so labels stay aligned, then the
# 0-4 severity grade binarised to presence / absence.
heart = fetch_ucirepo(id=45)
feature_names = list(heart.data.features.columns)

X_all = heart.data.features.values.astype(float)
num = heart.data.targets['num'].values.astype(float)

# the notebook joins features and target before `dropna`, so that dropping a row
# with a missing `ca` or `thal` drops its label too; the mask does the same job
complete = ~np.isnan(np.column_stack([X_all, num])).any(axis=1)
X = X_all[complete]
y = (num[complete] > 0).astype(int)
print('rows after dropping missing: %d of %d' % (len(X), len(X_all)))

# the notebook's split, from the seed set above
idx = np.random.permutation(len(X))
n_test = int(0.25 * len(X))
X_train, y_train = X[idx[n_test:]], y[idx[n_test:]]

# one column only, so the ensemble is a curve rather than a 13-dimensional surface
COL = feature_names.index('thalach')          # maximum heart rate achieved
X1 = X_train[:, [COL]]
x_flat = X1[:, 0]

ETA = 0.1
MAX_DEPTH = 3
LAMBDA = 1.0
GAMMA = 0.05
N_ROUNDS = 40

model = XGBoost(n_rounds=N_ROUNDS, learning_rate=ETA, max_depth=MAX_DEPTH,
                lambda_=LAMBDA, gamma=GAMMA, min_child_weight=1.0).fit(X1, y_train)

print('rows: %d   feature: %s   F0 = %.4f (base rate %.3f)'
      % (len(y_train), feature_names[COL], model.F0, sigmoid(model.F0)))
print('train log loss: %.4f at round 1 -> %.4f at round %d'
      % (model.train_loss[0], model.train_loss[-1], N_ROUNDS))


# ------------------------------------------------- replaying the rounds
# F_m on the training rows and on a plotting grid, accumulated the same way fit
# accumulated it: F <- F + eta * h_m(x). Storing them once keeps every frame a
# lookup instead of a refit.
X_LO, X_HI = x_flat.min() - 3, x_flat.max() + 3
grid = np.linspace(X_LO, X_HI, 360).reshape(-1, 1)

F_rows = [np.full(len(y_train), model.F0)]
F_grid = [np.full(len(grid), model.F0)]
tree_grid = []                                   # h_m(x) over the grid
for tree in model.trees:
    tree_grid.append(tree.predict(grid))
    F_rows.append(F_rows[-1] + ETA * tree.predict(X1))
    F_grid.append(F_grid[-1] + ETA * tree_grid[-1])

# 10 equal-count regions of x. Equal counts rather than equal width because the
# feature is bunched in the middle, and a region holding two rows would show a
# residual that is noise rather than error.
N_BINS = 10
edges = np.quantile(x_flat, np.linspace(0, 1, N_BINS + 1))
edges[0] -= 1e-6
edges[-1] += 1e-6
bin_of = np.clip(np.digitize(x_flat, edges) - 1, 0, N_BINS - 1)
centres = 0.5 * (edges[:-1] + edges[1:])
widths = 0.82 * (edges[1:] - edges[:-1])
observed = np.array([y_train[bin_of == k].mean() for k in range(N_BINS)])

# The bar for a region is that region's own leaf weight, -G/(H + lambda): the
# correction it would receive if the tree gave it a leaf of its own. It is the
# second-order version of "how much error is left here", and it is in the same
# units as the tree's leaves, so bars and staircase can share one axis.
region_w = np.zeros((N_ROUNDS, N_BINS))
for m in range(N_ROUNDS):
    g, h = gradients(y_train, F_rows[m])
    for k in range(N_BINS):
        sel = bin_of == k
        region_w[m, k] = -g[sel].sum() / (h[sel].sum() + LAMBDA)

print('largest region correction: %.3f at round 1 -> %.3f at round %d'
      % (np.abs(region_w[0]).max(), np.abs(region_w[-1]).max(), N_ROUNDS))


# ------------------------------------------------------------------ the frame
# every round while the corrections are large, then a coarser tail
rounds = list(range(1, 25)) + [26, 28, 30, 33, 36, 40]
HOLD = 6
frames = rounds + [rounds[-1]] * HOLD

fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
ax_fit = fig.add_axes([0.098, 0.545, 0.885, 0.345])
ax_res = fig.add_axes([0.098, 0.135, 0.885, 0.300])

# the update rule, with the notebook's learning rate substituted in
fig.text(0.5, 0.962,
         r'$F_m(x)=F_{m-1}(x)+\eta\,h_m(x),\qquad \eta=0.1,\qquad$'
         r'$w_j^{*}=-\dfrac{G_j}{H_j+\lambda}$',
         ha='center', va='center', fontsize=11.5, color='#3b3b3b')
fig.text(0.5, 0.030,
         'shrinkage: each tree is trusted for only a tenth of what it asks for '
         '- many small steps, not a few large ones',
         ha='center', va='center', fontsize=8, color=GREY)

# --- top: the data and the current ensemble
# rows drawn as ticks along the two labels rather than as 223 overlapping dots
ax_fit.plot(x_flat[y_train == 1], np.full((y_train == 1).sum(), 0.965), '|',
            color=COLORS[1], markersize=5, markeredgewidth=0.9, alpha=0.7,
            label='presence', zorder=2)
ax_fit.plot(x_flat[y_train == 0], np.full((y_train == 0).sum(), 0.035), '|',
            color=COLORS[0], markersize=5, markeredgewidth=0.9, alpha=0.7,
            label='absence', zorder=2)
ax_fit.plot(centres, observed, 'o', color=GREY, markersize=4.5,
            label='observed rate', zorder=3)
ax_fit.axhline(sigmoid(model.F0), color=GREY, linestyle='--', linewidth=0.9,
               alpha=0.8, zorder=1)
ax_fit.text(X_LO + 2, sigmoid(model.F0) + 0.035, r'$\sigma(F_0)$', fontsize=8.5,
            color=GREY, va='bottom')
(p_line,) = ax_fit.plot([], [], color=COLORS[4], linewidth=2.2,
                        label=r'$\sigma(F_m(x))$', zorder=4)
ax_fit.set_xlim(X_LO, X_HI)
ax_fit.set_ylim(-0.06, 1.06)
ax_fit.set_ylabel('P(disease)', fontsize=9)
ax_fit.set_xticklabels([])
ax_fit.tick_params(labelsize=8)
ax_fit.legend(loc='lower left', fontsize=7.5, ncol=2, framealpha=0.9,
              handletextpad=0.4, columnspacing=1.0, borderpad=0.3,
              labelspacing=0.25)
fit_text = ax_fit.text(0.985, 0.93, '', transform=ax_fit.transAxes, fontsize=9,
                       va='top', ha='right', color='#2b2b2b',
                       bbox=dict(boxstyle='round,pad=0.32', fc='white',
                                 ec=GREY, alpha=0.9))

# --- bottom: what is left over, and the tree fitted to it
bars = ax_res.bar(centres, np.zeros(N_BINS), width=widths, color=COLORS[0],
                  alpha=0.55, linewidth=0, zorder=2)
(step_eta,) = ax_res.plot([], [], color=COLORS[2], linewidth=1.0, alpha=0.9,
                          zorder=3)
eta_band = [ax_res.fill_between(grid[:, 0], 0, 0, color=COLORS[2], alpha=0.30,
                                linewidth=0, zorder=3)]
(step_h,) = ax_res.plot([], [], color=COLORS[2], linewidth=2.0, zorder=4,
                        label=r'new tree $h_m(x)$')
ax_res.axhline(0.0, color='#5a5a5a', linewidth=0.9, zorder=5)
ax_res.set_xlim(X_LO, X_HI)
ax_res.set_ylim(-1.35, 1.35)
ax_res.set_xlabel('thalach   (maximum heart rate achieved)', fontsize=9)
ax_res.set_ylabel(r'correction  $-G/(H{+}\lambda)$', fontsize=9)
ax_res.tick_params(labelsize=8)
res_text = ax_res.text(0.015, 0.05, '', transform=ax_res.transAxes, fontsize=8.5,
                       va='bottom', ha='left', color='#2b2b2b',
                       bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                 ec=GREY, alpha=0.9))
# proxies so the bars and the shaded band get named too; the bars switch colour
# with the sign of the correction, so the key for them is neutral grey
bar_key = Patch(facecolor=GREY, alpha=0.55, linewidth=0)
band_key = Patch(facecolor=COLORS[2], alpha=0.30, linewidth=0)
ax_res.legend([bar_key, step_h, band_key],
              ['error left, by region', r'new tree $h_m(x)$',
               r'$\eta\,h_m(x)$ added to $F$'],
              loc='upper right', fontsize=7.5, framealpha=0.9, borderpad=0.3,
              handletextpad=0.5, labelspacing=0.25)


def draw(m):
    """Draw the state entering boosting round m: the ensemble after m-1 trees,
    the error it has left, and the tree round m fits to that error."""
    prev = m - 1                     # trees already in the ensemble

    # top: the staircase the first (m-1) trees add up to
    p_line.set_data(grid[:, 0], sigmoid(F_grid[prev]))
    fit_text.set_text('round $m$ = %d\n%d trees in\nlog loss %.4f'
                      % (m, prev, model.train_loss[prev - 1] if prev else
                         log_loss(y_train, sigmoid(F_rows[0]))))

    # bottom: leftover correction per region, coloured by which way it pulls
    w = region_w[prev]
    for bar, v in zip(bars, w):
        bar.set_height(v)
        bar.set_color(COLORS[1] if v > 0 else COLORS[0])
        bar.set_alpha(0.55)

    # the tree fitted to exactly those gradients, and the tenth of it kept
    h = tree_grid[prev]
    step_h.set_data(grid[:, 0], h)
    step_eta.set_data(grid[:, 0], ETA * h)
    eta_band[0].remove()
    eta_band[0] = ax_res.fill_between(grid[:, 0], 0, ETA * h, color=COLORS[2],
                                      alpha=0.30, linewidth=0, zorder=3)

    res_text.set_text('largest correction still wanted: %.2f\n'
                      r'added this round: $\eta\,h_m$, up to %.2f'
                      % (np.abs(w).max(), ETA * np.abs(h).max()))
    return p_line, fit_text, step_h, step_eta, res_text


anim = animation.FuncAnimation(fig, draw, frames=frames, interval=125, blit=False)
save_gif(anim, OUT, fps=8)
print('frames:', len(frames))
