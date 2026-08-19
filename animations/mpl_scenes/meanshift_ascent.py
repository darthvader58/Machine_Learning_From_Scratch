"""Mean Shift as gradient ascent on a kernel density estimate.

The notebook `clustering/mean_shift.ipynb` runs on 800 standardised skin-segmentation
pixels (UCI id 229) with bandwidth h = 0.45.  This animation uses that same sample and
that same bandwidth, restricted to the R-G plane the notebook plots, so the density
surface the centroids climb can actually be drawn.

What is on screen:
  * the filled contours are the Gaussian KDE built from all 800 points,
        f(x) = (1/n) sum_i exp(-||x - x_i||^2 / 2h^2)
  * every walker is a centroid started on a data point, moved each frame by the real
    mean shift vector m(c) - c, which points along grad f / f
  * the trails are where each centroid has been, so the ascent paths stay visible
  * at the end the surviving centroids are counted - the cluster count is discovered,
    never supplied.

Run from the repo root:
    /opt/anaconda3/envs/tf_mps/bin/python animations/mpl_scenes/meanshift_ascent.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mpl_style import plt, animation, COLORS, FIGSIZE, DPI, save_gif  # noqa: E402

from ucimlrepo import fetch_ucirepo  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), '..', 'gifs', 'meanshift_ascent.gif')
BANDWIDTH = 0.45          # the notebook's h
N_WALKERS = 150           # how many of the 800 centroids we actually draw
FPS = 10


# ---------------------------------------------------------------- the algorithm
class Mean_Shift:
    """The notebook's class, with the centroid positions recorded each pass.

    Only `self.history` is new; the update itself is untouched.
    """

    def __init__(self, bandwidth=0.45, max_iter=100):
        self.bandwidth = bandwidth
        self.max_iter = max_iter

    def fit(self, data):
        # every point starts as its own centroid
        centroids = np.array(data, dtype=float)
        self.history = [centroids.copy()]

        for i in range(self.max_iter):
            new_centroids = np.zeros_like(centroids)

            for j in range(len(centroids)):
                # nearby points get a big weight, distant ones almost none
                distances = np.linalg.norm(data - centroids[j], axis=1)
                weights = np.exp(-(distances ** 2) / (2 * self.bandwidth ** 2))

                # move to the weighted average, which is a step towards the denser side
                new_centroids[j] = (data * weights[:, None]).sum(axis=0) / weights.sum()

            shift = np.abs(new_centroids - centroids).max()
            centroids = new_centroids
            self.history.append(centroids.copy())

            if shift < 1e-4:
                break

        # centroids that climbed to the same peak become one cluster
        merged = []
        for c in centroids:
            if all(np.linalg.norm(c - m) > self.bandwidth / 2 for m in merged):
                merged.append(c)

        self.centroids = {i: merged[i] for i in range(len(merged))}
        self.n_iter = i + 1


# ---------------------------------------------------------------- the data
skin = fetch_ucirepo(id=229)
X_all = skin.data.features.values.astype(float)

# the notebook's sample: same seed, same size
rng = np.random.default_rng(0)
sample = rng.choice(len(X_all), 800, replace=False)
X = X_all[sample]
X = (X - X.mean(axis=0)) / X.std(axis=0)

# the R-G plane, which is the pair the notebook plots
data = X[:, [2, 1]]

ms = Mean_Shift(bandwidth=BANDWIDTH)
ms.fit(data)
history = np.array(ms.history)          # (passes+1, 800, 2)
n_modes = len(ms.centroids)
modes = np.array([ms.centroids[c] for c in sorted(ms.centroids)])
print(f'{n_modes} modes found in {ms.n_iter} passes')

# draw only a subset of the walkers so the frame does not turn into mush;
# the density surface below still comes from all 800 points
pick = np.random.default_rng(1).choice(len(data), N_WALKERS, replace=False)
paths = history[:, pick, :]

# ---------------------------------------------------------------- the density surface
pad = 0.55
xs = np.linspace(data[:, 0].min() - pad, data[:, 0].max() + pad, 220)
ys = np.linspace(data[:, 1].min() - pad, data[:, 1].max() + pad, 220)
GX, GY = np.meshgrid(xs, ys)
grid = np.stack([GX.ravel(), GY.ravel()], axis=1)

# f(x) = mean_i exp(-||x - x_i||^2 / 2h^2), evaluated in blocks to keep memory sane
dens = np.zeros(len(grid))
for lo in range(0, len(grid), 4000):
    chunk = grid[lo:lo + 4000]
    d2 = ((chunk[:, None, :] - data[None, :, :]) ** 2).sum(axis=2)
    dens[lo:lo + 4000] = np.exp(-d2 / (2 * BANDWIDTH ** 2)).mean(axis=1)
dens = dens.reshape(GX.shape)

# ---------------------------------------------------------------- the figure
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
fig.subplots_adjust(left=0.09, right=0.98, top=0.74, bottom=0.10)

# filled contours: few levels keeps the GIF palette small and the bands readable
ax.contourf(GX, GY, dens, levels=11, cmap='Blues', zorder=0)
ax.contour(GX, GY, dens, levels=11, colors='white', linewidths=0.4, alpha=0.5, zorder=1)

ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(ys[0], ys[-1])
ax.set_xlabel('R (standardised)', fontsize=9)
ax.set_ylabel('G (standardised)', fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(False)

# header: title, then the governing equation, then what the data is.  The y values
# are chosen so the tall \dfrac never collides with the line above or below it.
fig.suptitle('Mean Shift climbs the kernel density estimate', fontsize=11.5, y=0.985)
fig.text(0.5, 0.872,
         r'$m(x)=\dfrac{\sum_i K(x-x_i)\,x_i}{\sum_i K(x-x_i)}-x\;\;\propto\;\;'
         r'\dfrac{\nabla \hat f(x)}{\hat f(x)}$',
         ha='center', va='center', fontsize=9.5)
fig.text(0.5, 0.775,
         f'800 skin pixels, Gaussian KDE, bandwidth $h={BANDWIDTH}$  -  no $k$ is given',
         ha='center', va='center', fontsize=8.5, color='#555555')

# the trails: one line per walker, drawn faint so the paths accumulate
trails = [ax.plot([], [], lw=0.8, color=COLORS[1], alpha=0.35, zorder=2)[0]
          for _ in range(N_WALKERS)]
walkers = ax.scatter(paths[0, :, 0], paths[0, :, 1], s=13, color=COLORS[1],
                     edgecolors='white', linewidths=0.3, zorder=4)
mode_marks = ax.scatter([], [], s=170, marker='x', color='k', linewidths=2.2, zorder=5)

caption = ax.text(0.985, 0.03, '', transform=ax.transAxes, ha='right', va='bottom',
                  fontsize=9.5, color='#222222',
                  bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#999999', alpha=0.85))

# ---------------------------------------------------------------- the timeline
HOLD_IN = 5                       # sit on the raw data first
n_steps = len(paths) - 1
HOLD_OUT = 8                      # sit on the answer so the loop does not snap
frames = HOLD_IN + n_steps + HOLD_OUT


def draw(f):
    if f < HOLD_IN:
        t = 0
    elif f < HOLD_IN + n_steps:
        t = f - HOLD_IN + 1
    else:
        t = n_steps

    walkers.set_offsets(paths[t])

    # every walker's path so far, so the ascent stays on screen at the end
    for w, line in enumerate(trails):
        line.set_data(paths[:t + 1, w, 0], paths[:t + 1, w, 1])

    if f < HOLD_IN:
        caption.set_text('800 centroids, one per point')
        mode_marks.set_offsets(np.empty((0, 2)))
    elif t < n_steps:
        caption.set_text(f'pass {t}   -   each step is $m(c)-c$')
        mode_marks.set_offsets(np.empty((0, 2)))
    else:
        # merge: keep a centroid only if it is further than h/2 from every one kept
        caption.set_text(f'converged - {n_modes} distinct modes survive the merge')
        mode_marks.set_offsets(modes)

    return trails + [walkers, mode_marks, caption]


anim = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 // FPS, blit=False)
save_gif(anim, os.path.normpath(OUT), fps=FPS)
print('frames:', frames)
