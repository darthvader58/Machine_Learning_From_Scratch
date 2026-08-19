"""Random forest: many high-variance trees, averaged into one low-variance vote.

Mirrors `classification/random_forest.ipynb`: same UCI Heart Disease data (id 45),
same `num > 0` binary target, same seeded 75/25 split, and the notebook's own
`gini`, `Decision_Tree` and `Random_Forest` classes copied in verbatim. The forest
is fitted with `random_state=42`, exactly as the notebook fits it.

The only change is the feature set. The notebook uses all 13 columns, which has no
picture; here the forest is fitted on two of them - `thalach` (maximum heart rate)
and `oldpeak` (ST depression) - so every tree's decision surface can be drawn. With
p = 2 the forest's own rule for the split subset, k = floor(sqrt(p)) = 1, means each
node picks ONE of the two features at random before searching thresholds. That is
the second source of randomness doing its job in plain sight: the root split of tree
m is on a different feature from the root split of tree m + 1.

Left panel: each tree's boundary as it arrives, accumulating in faint grey, with the
current tree bold. The training points are resized by how many times that tree's
bootstrap sample drew them, so the rows each tree actually saw are visible too.
Right panel: the running vote fraction over the trees so far, and the majority-vote
boundary it implies, smoothing out as the pile of jagged trees grows.
Bottom: test accuracy against the number of trees - it climbs, then flattens.

Run from the repo root:
    /opt/anaconda3/envs/tf_mps/bin/python animations/mpl_scenes/forest_variance.py
"""
import os
import sys

import numpy as np
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mpl_style import plt, COLORS, GREY, FIGSIZE, DPI, save_gif  # noqa: E402

from ucimlrepo import fetch_ucirepo  # noqa: E402

ABSENCE, PRESENCE = COLORS[0], COLORS[1]
AQUA, VIOLET = COLORS[2], COLORS[4]

N_TREES = 32          # one animation frame per tree
MAX_DEPTH = 8         # deep enough that a single tree is visibly jagged
GRID = 110            # resolution of the decision surface
VOTE_STEPS = 8        # the vote field is shaded in steps of 1/8, which keeps the GIF small
OUT = os.path.join(os.path.dirname(__file__), '..', 'gifs', 'forest_variance.gif')


# ---------------------------------------------------------------- the notebook's classes
def gini(labels):
    if len(labels) == 0:
        return 0.0
    p = np.bincount(labels, minlength=2) / len(labels)
    return 1.0 - np.sum(p ** 2)


class Decision_Tree:
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.random_state = random_state

    def fit(self, X, y):
        self.rng = np.random.default_rng(self.random_state)
        self.n_total_features = X.shape[1]
        self.n_train = len(y)
        self.importances = np.zeros(self.n_total_features)
        self.root = self.grow(X, y, 0)
        if self.importances.sum() > 0:
            self.importances = self.importances / self.importances.sum()
        return self

    def grow(self, X, y, depth):
        n_samples = len(y)
        counts = np.bincount(y, minlength=2)
        node = {'leaf': True, 'value': int(np.argmax(counts)), 'gini': gini(y),
                'n': n_samples, 'counts': counts, 'depth': depth}

        if (depth >= self.max_depth or n_samples < self.min_samples_split
                or counts.max() == n_samples):
            return node

        # a forest tree only looks at a random subset of features here
        if self.n_features is None:
            feature_ids = np.arange(self.n_total_features)
        else:
            feature_ids = self.rng.choice(self.n_total_features, size=self.n_features,
                                          replace=False)

        best = self.best_split(X, y, feature_ids)
        if best is None:
            return node

        feature, threshold, decrease = best
        left = X[:, feature] <= threshold

        # weight the impurity drop by how much of the training set reached this node
        self.importances[feature] += decrease * n_samples / self.n_train

        node['leaf'] = False
        node['feature'] = feature
        node['threshold'] = threshold
        node['decrease'] = decrease
        node['left'] = self.grow(X[left], y[left], depth + 1)
        node['right'] = self.grow(X[~left], y[~left], depth + 1)
        return node

    def best_split(self, X, y, feature_ids):
        n = len(y)
        parent = gini(y)
        best_gain = 0.0
        best = None

        for feature in feature_ids:
            column = X[:, feature]
            values = np.unique(column)
            if len(values) < 2:
                continue

            # midpoints between neighbouring observed values
            thresholds = (values[:-1] + values[1:]) / 2.0

            for threshold in thresholds:
                left = column <= threshold
                n_left = left.sum()
                n_right = n - n_left
                if n_left == 0 or n_right == 0:
                    continue

                gain = parent - (n_left / n) * gini(y[left]) - (n_right / n) * gini(y[~left])
                if gain > best_gain:
                    best_gain = gain
                    best = (feature, threshold, gain)

        return best

    def predict(self, X):
        return np.array([self.walk(x, self.root) for x in X])

    def walk(self, x, node):
        while not node['leaf']:
            if x[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['value']


class Random_Forest:
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2, n_features=None,
                 random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n = len(y)

        if self.n_features is None:
            k = max(1, int(np.sqrt(X.shape[1])))
        else:
            k = self.n_features

        self.trees = []
        self.bootstrap_rows = []

        for _ in range(self.n_trees):
            idx = rng.integers(0, n, size=n)          # bootstrap: n rows, with replacement
            tree = Decision_Tree(max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split,
                                 n_features=k,
                                 random_state=int(rng.integers(0, 2 ** 31 - 1)))
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
            self.bootstrap_rows.append(idx)

        self.n_features_used = k
        self.importances = np.mean([t.importances for t in self.trees], axis=0)
        self.oob_score = self.compute_oob(X, y)
        return self

    def tree_votes(self, X):
        return np.array([t.predict(X) for t in self.trees])

    def predict(self, X, n_trees=None):
        votes = self.tree_votes(X)
        if n_trees is not None:
            votes = votes[:n_trees]
        return (votes.mean(axis=0) > 0.5).astype(int)

    def compute_oob(self, X, y):
        n = len(y)
        votes = np.zeros((n, 2))

        for tree, idx in zip(self.trees, self.bootstrap_rows):
            oob = np.ones(n, dtype=bool)
            oob[idx] = False                          # rows this tree never saw
            if oob.sum() == 0:
                continue
            rows = np.where(oob)[0]
            votes[rows, tree.predict(X[rows])] += 1

        scored = votes.sum(axis=1) > 0
        predicted = np.argmax(votes[scored], axis=1)
        self.oob_rows = scored.sum()
        return (predicted == y[scored]).mean()


# ---------------------------------------------------------------- data, exactly as the notebook
heart = fetch_ucirepo(id=45)
all_names = list(heart.data.features.columns)

# join before dropping so features and target stay aligned
frame = heart.data.features.join(heart.data.targets).dropna()
X_all = frame[all_names].values.astype(float)
y = (frame['num'].values > 0).astype(int)             # 0 = absence, 1 = presence

# two features, so the boundary is drawable
f1, f2 = all_names.index('thalach'), all_names.index('oldpeak')
feature_names = ['thalach', 'oldpeak']
X = X_all[:, [f1, f2]]

rng = np.random.default_rng(42)
order = rng.permutation(len(y))
cut = int(0.75 * len(y))
train_idx, test_idx = order[:cut], order[cut:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# the reference: one unrestricted tree on the same two features
single = Decision_Tree(max_depth=MAX_DEPTH, random_state=1).fit(X_train, y_train)
single_acc = (single.predict(X_test) == y_test).mean()

forest = Random_Forest(n_trees=N_TREES, max_depth=MAX_DEPTH, random_state=42)
forest.fit(X_train, y_train)
print(f'{N_TREES} trees, {forest.n_features_used} of 2 features per split')
print('single tree test accuracy:', round(single_acc, 4))
print('forest test accuracy:     ', round((forest.predict(X_test) == y_test).mean(), 4))
print('forest OOB score:         ', round(forest.oob_score, 4))

# accuracy of the first m trees, the notebook's running-mean trick over the vote matrix
votes = forest.tree_votes(X_test)
running = np.cumsum(votes, axis=0) / np.arange(1, N_TREES + 1)[:, None]
acc_by_trees = ((running > 0.5).astype(int) == y_test).mean(axis=1)
for t in (1, 2, 4, 8, 16, N_TREES):
    print(f'{t:>3} trees: {acc_by_trees[t - 1]:.4f}')

# ---------------------------------------------------------------- every tree over the plane
pad_x = 0.04 * (X[:, 0].max() - X[:, 0].min())
pad_y = 0.04 * (X[:, 1].max() - X[:, 1].min())
gx = np.linspace(X[:, 0].min() - pad_x, X[:, 0].max() + pad_x, GRID)
gy = np.linspace(X[:, 1].min() - pad_y, X[:, 1].max() + pad_y, GRID)
mesh = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)

# each tree's own prediction everywhere, from the notebook's `predict`
surfaces = np.array([t.predict(mesh).reshape(GRID, GRID) for t in forest.trees],
                    dtype=np.float64)
# vote fraction of the first m trees; > 0.5 is the majority-vote class
vote_frac = np.cumsum(surfaces, axis=0) / np.arange(1, N_TREES + 1)[:, None, None]

# Shade that fraction in steps of 1/VOTE_STEPS. Rounding away from 0.5 rather than to
# the nearest step means a pixel the majority calls presence never shades as a tie, so
# the drawn boundary is still exactly where the vote flips.
side = np.sign(vote_frac - 0.5)
vote_shade = np.clip(0.5 + side * np.ceil(np.abs(vote_frac - 0.5) * VOTE_STEPS)
                     / VOTE_STEPS, 0.0, 1.0)
print('decision surfaces done')

# how many times each training row was drawn into each tree's bootstrap sample
draw_counts = np.array([np.bincount(idx, minlength=len(y_train))
                        for idx in forest.bootstrap_rows])


# ---------------------------------------------------------------- frame plan
# Each frame is the number of trees M currently in the vote. The first 16 arrive one
# per frame; after that they arrive in pairs, because by then a single extra tree
# changes almost nothing - which is the point the accuracy curve is making anyway.
# A couple of repeats at the start and a hold at the end keep the loop readable.
frames = [1, 1] + list(range(1, 17)) + list(range(18, N_TREES + 1, 2))
frames = frames + [frames[-1]] * 6


# ---------------------------------------------------------------- figure
# Taller than the shared FIGSIZE because the accuracy curve sits under the two
# panels; the width, which is what legibility and file size depend on, is unchanged.
fig = plt.figure(figsize=(FIGSIZE[0], FIGSIZE[1] + 0.95), dpi=DPI)

eq = fig.text(0.5, 0.960, r'$\hat{y} = \mathrm{mode}\{h_m(x)\}_{m=1}^{M}$',
              ha='center', va='center', fontsize=12)
head = fig.text(0.5, 0.895, '', ha='center', va='center', fontsize=9.5)

ax_tree = fig.add_axes([0.085, 0.400, 0.405, 0.425])
ax_ens = fig.add_axes([0.560, 0.400, 0.405, 0.425])
ax_acc = fig.add_axes([0.085, 0.090, 0.880, 0.195])

for ax, title in ((ax_tree, 'the trees, one at a time'),
                  (ax_ens, 'their majority vote')):
    ax.set_xlim(gx[0], gx[-1])
    ax.set_ylim(gy[0], gy[-1])
    ax.set_xlabel(feature_names[0], fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=10)
ax_tree.set_ylabel(feature_names[1], fontsize=9)
ax_ens.set_yticklabels([])

# --- left panel: the training set, then one boundary per tree piled on top
ax_tree.scatter(X_train[:, 0], X_train[:, 1], s=7, linewidths=0,
                color=[ABSENCE if c == 0 else PRESENCE for c in y_train],
                alpha=0.22, zorder=1)
# resized every frame by that tree's bootstrap draw counts
bag = ax_tree.scatter(X_train[:, 0], X_train[:, 1], s=np.zeros(len(y_train)),
                      linewidths=0, alpha=0.55, zorder=2,
                      color=[ABSENCE if c == 0 else PRESENCE for c in y_train])
root_note = ax_tree.text(0.03, 0.04, '', transform=ax_tree.transAxes, fontsize=9,
                         color=VIOLET, zorder=6,
                         bbox=dict(boxstyle='square,pad=0.22', facecolor='white',
                                   edgecolor='none', alpha=0.85))

# --- right panel: the vote fraction as a field, blue for absence, orange for presence
cmap = LinearSegmentedColormap.from_list('vote', [ABSENCE, '#ffffff', PRESENCE])
field = ax_ens.imshow(np.zeros((GRID, GRID)), origin='lower', cmap=cmap, vmin=0, vmax=1,
                      extent=(gx[0], gx[-1], gy[0], gy[-1]), aspect='auto',
                      interpolation='nearest', zorder=1)
ax_ens.scatter(X_train[:, 0], X_train[:, 1], s=7, linewidths=0,
               color=[ABSENCE if c == 0 else PRESENCE for c in y_train],
               alpha=0.35, zorder=2)
# the shading is the vote itself: solid colour where the trees agree, pale where
# they are close to evenly split, so the boundary is the pale seam between them
ax_ens.text(0.03, 0.04, 'shade = share of trees voting presence',
            transform=ax_ens.transAxes, fontsize=9, zorder=6,
            bbox=dict(boxstyle='square,pad=0.22', facecolor='white',
                      edgecolor='none', alpha=0.85))

# --- bottom panel: accuracy against forest size
ax_acc.axhline(single_acc, color=GREY, linestyle='--', linewidth=1.1)
ax_acc.text(N_TREES * 0.985, single_acc - 0.004, 'one tree', fontsize=9, color=GREY,
            ha='right', va='top')
curve, = ax_acc.plot([], [], color=AQUA, linewidth=1.8)
head_dot, = ax_acc.plot([], [], 'o', color=AQUA, markersize=5)
acc_note = ax_acc.text(0.0, 0.0, '', fontsize=9, color=AQUA, ha='left', va='bottom')
ax_acc.set_xlim(0.5, N_TREES + 0.5)
lo = min(acc_by_trees.min(), single_acc) - 0.03
hi = max(acc_by_trees.max(), single_acc) + 0.035
ax_acc.set_ylim(lo, hi)
ax_acc.set_xlabel('number of trees in the vote,  $M$', fontsize=9)
ax_acc.set_ylabel('test accuracy', fontsize=9)
ax_acc.tick_params(labelsize=8)

drawn = {}          # tree index -> its boundary, so a repeated frame does not redraw it


def update(i):
    M = frames[i]                      # trees in the vote at this frame
    newest = M - 1                     # index of the last tree to arrive

    # --- left: every tree so far, the newest one bold and the rest faded back
    for m in range(M):
        if m not in drawn:
            # no antialiasing: the boundaries are axis-aligned steps anyway, and hard
            # edges keep the GIF's colour palette small
            drawn[m] = ax_tree.contour(gx, gy, surfaces[m], levels=[0.5], colors=[VIOLET],
                                       linewidths=1.7, zorder=4, antialiased=False)
    for m, cs in drawn.items():
        if m == newest:
            continue
        # older trees stay on screen but fade back, so the pile of past disagreements
        # is visible without burying the tree that just arrived
        cs.set_alpha(max(0.11, 0.34 * 0.72 ** (newest - m)))
        cs.set_linewidth(0.6)
        cs.set_edgecolor(GREY)
    drawn[newest].set_alpha(0.95)
    drawn[newest].set_linewidth(1.7)
    drawn[newest].set_edgecolor(VIOLET)

    # the rows this tree was handed: area grows with how often the bootstrap drew them
    bag.set_sizes(9.0 * draw_counts[newest])

    root = forest.trees[newest].root
    root_note.set_text(f"tree {M} root split: {feature_names[root['feature']]} "
                       f"$\\leq$ {root['threshold']:.1f}")

    # --- right: the vote of the first M trees
    field.set_data(vote_shade[newest])

    # --- bottom: the accuracy curve up to here
    curve.set_data(np.arange(1, M + 1), acc_by_trees[:M])
    head_dot.set_data([M], [acc_by_trees[M - 1]])
    acc_note.set_text(f'{acc_by_trees[M - 1]:.3f}')
    acc_note.set_position((M - N_TREES * 0.015, acc_by_trees[M - 1] + 0.008))

    n_distinct = int((draw_counts[newest] > 0).sum())
    head.set_text(f'$M$ = {M} of {N_TREES} trees:  tree {M} saw {n_distinct} distinct '
                  f'rows of {len(y_train)},  {forest.n_features_used} of 2 features per split')
    return []


# 9 fps rather than the shared 12: every frame is a discrete event - one more tree
# joining the vote - so a slower step reads better and costs no extra bytes.
anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / 9,
                               blit=False)
save_gif(anim, os.path.abspath(OUT), fps=9)
print(f'frames: {len(frames)}')
