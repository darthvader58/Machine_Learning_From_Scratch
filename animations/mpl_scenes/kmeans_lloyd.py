"""Lloyd's algorithm alternating between its two steps, on Iris.

Replays `clustering/k-means.ipynb` exactly: the `K_Means` class is copied from the
notebook, seeded the same way (the first k rows of the data), and fitted on the same
four measurements with k=3, so it converges in 12 passes to the centroids the notebook
prints. The only change to the class is a `self.frames` list recorded next to the
existing `self.history` - it stores the assignment and the centroid positions on each
pass so the animation can replay them. It does not touch the maths.

The picture is the petal-length / petal-width plane, which is the pair of columns the
notebook plots. Distances are still measured across all four features, exactly as the
notebook measures them.

Each pass is drawn as two visually separate beats:

  1. ASSIGN - every point recolours to its nearest centroid, the star markers hold
     still, and points that switched cluster get a black ring.
  2. UPDATE - the colours hold still and the star markers slide onto the mean of the
     points that chose them, leaving a trail behind.

The side panel tracks the inertia J, which falls on every pass and never rises. That
is the reason the loop terminates.

Run from the repo root:
    /opt/anaconda3/envs/tf_mps/bin/python animations/mpl_scenes/kmeans_lloyd.py
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

np.random.seed(0)          # nothing here is random, but the GIF must be reproducible


# ---------------------------------------------------------------------------
# The notebook's class, copied across. The only addition is the block marked
# "recorded for the animation" - it saves a snapshot of each pass.
# ---------------------------------------------------------------------------
class K_Means:
    def __init__(self, k=3, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        # start with the first k points as the centroids
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]

        self.history = []
        self.frames = []

        for i in range(self.max_iter):
            # empty the buckets before every pass
            self.classifications = {}
            for j in range(self.k):
                self.classifications[j] = []

            # put each point with the closest centroid
            labels = []
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid])
                             for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
                labels.append(classification)

            self.history.append(self.inertia())

            prev_centroids = dict(self.centroids)

            # move each centroid onto the average of its own points
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            # --- recorded for the animation, changes nothing above ---
            self.frames.append({
                'labels': np.array(labels),
                'before': np.array([prev_centroids[c] for c in range(self.k)]),
                'after': np.array([self.centroids[c] for c in range(self.k)]),
                'inertia': self.history[-1],
            })

            # if nothing moved by more than tol, we are finished
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum(np.abs(current_centroid - original_centroid)) > self.tol:
                    optimized = False

            if optimized:
                break

        self.n_iter = i + 1

    def inertia(self):
        # total squared distance from every point to its own centroid
        total = 0
        for classification in self.classifications:
            for featureset in self.classifications[classification]:
                total += np.linalg.norm(featureset - self.centroids[classification]) ** 2
        return total


# ---------------------------------------------------------------------------
# Data and fit - identical to the notebook
# ---------------------------------------------------------------------------
iris = fetch_ucirepo(id=53)
X = iris.data.features.values
feature_names = list(iris.data.features.columns)

K = 3
clf = K_Means(k=K)
clf.fit(X)
passes = clf.frames
n_pass = clf.n_iter

PL, PW = 2, 3                      # the two columns the notebook plots
px, py = X[:, PL], X[:, PW]


# ---------------------------------------------------------------------------
# Frame plan: every pass gets an ASSIGN beat then an UPDATE slide. Early passes
# move a long way so they get more frames; the last few barely move and get two.
# ---------------------------------------------------------------------------
frames = [('intro', 0, 0.0)] * 4
for p in range(n_pass):
    if p < 3:
        n_assign, n_update = 2, 3
    elif p < 7:
        n_assign, n_update = 2, 2
    else:
        n_assign, n_update = 2, 1
    frames += [('assign', p, 1.0)] * n_assign
    frames += [('update', p, (s + 1) / n_update) for s in range(n_update)]
frames += [frames[-1]] * 6         # hold the final state so the loop does not snap


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=FIGSIZE, dpi=DPI)

ax = fig.add_axes([0.075, 0.16, 0.55, 0.66])          # the petal plane
ax.set_xlim(0.6, 7.4)
ax.set_ylim(-0.18, 2.78)
ax.set_xlabel(feature_names[PL], fontsize=9)
ax.set_ylabel(feature_names[PW], fontsize=9)
ax.tick_params(labelsize=9)

jax = fig.add_axes([0.725, 0.47, 0.245, 0.33])        # the inertia panel
jax.set_yscale('log')
jax.set_xlim(0.4, n_pass + 0.6)
jax.set_ylim(55, 3200)
jax.set_xlabel('pass', fontsize=9, labelpad=1)
jax.tick_params(labelsize=9)
jax.set_yticks([100, 1000])
jax.set_yticklabels(['100', '1000'])
jax.set_xticks([1, 6, 12])

# the two steps, written out. The active one is highlighted each frame.
EQ_ASSIGN = r'$\bf{1.\ ASSIGN}$   $c_i = \mathrm{arg\,min}_j\ \|x_i - \mu_j\|^2$'
EQ_UPDATE = r'$\bf{2.\ UPDATE}$   $\mu_j = \frac{1}{|C_j|}\sum_{x \in C_j} x$'
t_assign = fig.text(0.055, 0.945, EQ_ASSIGN, fontsize=10, va='center')
t_update = fig.text(0.055, 0.868, EQ_UPDATE, fontsize=10, va='center')
t_pass = fig.text(0.975, 0.945, '', fontsize=10, ha='right', va='center')

fig.text(0.80, 0.845, r'inertia  $J=\sum_j\sum_{x \in C_j}\|x-\mu_j\|^2$',
         fontsize=9, ha='center', va='center')
t_j = fig.text(0.66, 0.355, '', fontsize=10)
t_counts = [fig.text(0.66, 0.283 - 0.055 * j, '', fontsize=9, color=COLORS[j])
            for j in range(K)]
fig.text(0.66, 0.095, 'J can only fall, so\nthe loop must stop.',
         fontsize=9, color=GREY, va='top')
fig.text(0.075, 0.045,
         'distances use all 4 features; the petal plane is shown',
         fontsize=9, color=GREY)

# artists
pts = ax.scatter(px, py, s=22, c=[GREY] * len(px), linewidths=0.0,
                 edgecolors='k', zorder=2)
trails = [ax.plot([], [], color=COLORS[j], lw=1.0, alpha=0.55, zorder=3)[0]
          for j in range(K)]
cents = ax.scatter(np.zeros(K), np.zeros(K), s=300, marker='*',
                   c=[COLORS[j] for j in range(K)],
                   edgecolors='k', linewidths=1.2, zorder=6)
jline, = jax.plot([], [], color=COLORS[4], lw=1.6, marker='o', ms=3.5, zorder=3)
jdot, = jax.plot([], [], color=COLORS[1], marker='o', ms=6.5, zorder=4)

HI_BOX = dict(facecolor='white', edgecolor=COLORS[0], boxstyle='round,pad=0.25')


def highlight(active):
    """Box and darken whichever of the two steps is running."""
    for text, name in ((t_assign, 'assign'), (t_update, 'update')):
        if name == active:
            text.set_color('#222222')
            text.set_bbox(HI_BOX)
        else:
            text.set_color('#aaaaaa')
            text.set_bbox(None)


def draw(idx):
    kind, p, t = frames[idx]

    if kind == 'intro':
        # the raw data with the three seed centroids sitting on the first 3 rows
        pts.set_facecolor([GREY] * len(px))
        pts.set_linewidth(np.zeros(len(px)))
        cent = passes[0]['before']
        for j in range(K):
            trails[j].set_data([], [])
        jline.set_data([], [])
        jdot.set_data([], [])
        highlight(None)
        t_pass.set_text('seeded on the first 3 rows')
        t_j.set_text('')
        for j in range(K):
            t_counts[j].set_text('')
        cents.set_offsets(np.c_[cent[:, PL], cent[:, PW]])
        return

    labels = passes[p]['labels']
    prev = passes[p - 1]['labels'] if p > 0 else np.full(len(px), -1)

    # colour every point by the centroid it chose
    pts.set_facecolor([COLORS[c] for c in labels])

    if kind == 'assign':
        # ring the points that switched cluster on this pass
        pts.set_linewidth(np.where(labels != prev, 1.0, 0.0))
        cent = passes[p]['before']
        highlight('assign')
        step = 'ASSIGN'
    else:
        # the colours hold; the centroids slide to the mean of their own points
        pts.set_linewidth(np.zeros(len(px)))
        ease = 0.5 - 0.5 * np.cos(np.pi * t)
        cent = passes[p]['before'] + ease * (passes[p]['after'] - passes[p]['before'])
        highlight('update')
        step = 'UPDATE'

    cents.set_offsets(np.c_[cent[:, PL], cent[:, PW]])

    # trail: where each centroid has been, up to where it is now
    for j in range(K):
        path = [passes[q]['before'][j] for q in range(p + 1)] + [cent[j]]
        path = np.array(path)
        trails[j].set_data(path[:, PL], path[:, PW])

    # inertia so far - one reading per completed assignment
    ys = [passes[q]['inertia'] for q in range(p + 1)]
    jline.set_data(range(1, p + 2), ys)
    jdot.set_data([p + 1], [ys[-1]])

    t_pass.set_text(f'pass {p + 1} / {n_pass}   {step}')
    t_j.set_text(f'$J$ = {ys[-1]:.1f}')
    counts = np.bincount(labels, minlength=K)
    for j in range(K):
        t_counts[j].set_text(f'cluster {j}:  {counts[j]} flowers')


anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / 12)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gifs',
                   'kmeans_lloyd.gif')
save_gif(anim, os.path.normpath(out), fps=12)
print('frames:', len(frames), ' passes:', n_pass)
