"""KNN: the query ball grows until it holds exactly k neighbours, then they vote.

Mirrors `classification/k_nearest_neighbors.ipynb`: same Breast Cancer Wisconsin
data (UCI id 15), same seeded 80/20 split, same two plotted measurements
(Uniformity_of_cell_size, Bare_nuclei), same Euclidean distance, and the
notebook's own `k_nearest_neighbors` function copied in verbatim.

The measurements are whole numbers from 1 to 10, so hundreds of training samples
land on identical coordinates. The notebook adds a seeded jitter when it plots
them; this animation uses that same jittered set as the data it runs on, so the
circle drawn on screen and the distances the algorithm sorts are the same
numbers. With the notebook's jitter (seed 1, sigma 0.12) the k = 5 vote comes out
3 malignant to 2 benign, exactly the split the notebook reports.

Run from the repo root:
    /opt/anaconda3/envs/tf_mps/bin/python animations/mpl_scenes/knn_vote.py
"""
import os
import sys
from collections import Counter

import numpy as np
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib import animation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mpl_style import plt, COLORS, GREY, FIGSIZE, DPI, save_gif  # noqa: E402

from ucimlrepo import fetch_ucirepo  # noqa: E402

BENIGN, MALIGNANT = COLORS[0], COLORS[1]
K_SMALL, K_LARGE = 5, 25
OUT = os.path.join(os.path.dirname(__file__), '..', 'gifs', 'knn_vote.gif')


# ---------------------------------------------------------------- the notebook's function
def k_nearest_neighbors(data, predict, k=5):
    distances = []

    # distance from the new point to every training sample
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    # the k closest ones vote
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result, confidence


# ---------------------------------------------------------------- data, exactly as the notebook
cancer = fetch_ucirepo(id=15)
features = cancer.data.features
targets = cancer.data.targets

# 16 rows have a missing Bare_nuclei value
keep = ~features.isna().any(axis=1)
X = features[keep].values.astype(float)
y = targets[keep].values.ravel()
feature_names = list(features.columns)

rng = np.random.default_rng(0)
order = rng.permutation(len(X))
split = int(0.8 * len(X))

train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
for i in order[:split]:
    train_set[y[i]].append(X[i])
for i in order[split:]:
    test_set[y[i]].append(X[i])

f1, f2 = 1, 5   # uniformity of cell size, bare nuclei

# the notebook's jitter, reused here as the coordinates the algorithm actually sees
jitter = np.random.default_rng(1)
two_d = {}
for group in (2, 4):
    pts = np.array([[p[f1], p[f2]] for p in train_set[group]])
    two_d[group] = pts + jitter.normal(0, 0.12, pts.shape)

# the notebook's chosen test sample: a malignant one sitting between the groups
point = np.array([test_set[4][3][f1], test_set[4][3][f2]])

# every distance, sorted - this ordering is the capture order of the growing ball
ranked = []
for group in two_d:
    for p in two_d[group]:
        ranked.append((np.linalg.norm(p - point), group, p))
ranked.sort(key=lambda d: d[0])

radii = np.array([d[0] for d in ranked])
groups = np.array([d[1] for d in ranked])
coords = np.array([d[2] for d in ranked])

# the animation's numbers come from the notebook's function, not from a second implementation
vote_small, conf_small = k_nearest_neighbors(two_d, point, k=K_SMALL)
vote_large, conf_large = k_nearest_neighbors(two_d, point, k=K_LARGE)
assert vote_small == Counter(groups[:K_SMALL]).most_common(1)[0][0]
assert vote_large == Counter(groups[:K_LARGE]).most_common(1)[0][0]
print(f'k={K_SMALL}: vote {vote_small}, confidence {conf_small:.2f}, r={radii[K_SMALL - 1]:.2f}')
print(f'k={K_LARGE}: vote {vote_large}, confidence {conf_large:.2f}, r={radii[K_LARGE - 1]:.2f}')

# the same vote run over a grid, so the two k values can be compared as whole boundaries
grid = np.linspace(0, 11, 45)
zones = {}
for k in (K_SMALL, K_LARGE):
    zone = np.zeros((len(grid), len(grid)))
    for i, a in enumerate(grid):
        for j, b in enumerate(grid):
            v, _ = k_nearest_neighbors(two_d, [a, b], k=k)
            zone[j, i] = v
    zones[k] = zone
    print(f'decision field for k={k} done')


# ---------------------------------------------------------------- frame plan
# Each frame is a dict the update function reads. Radius drives everything: the
# number of captured neighbours is just how many sorted distances fall inside it.
def growth(k, n_frames, start=0.55):
    """Radii that grow from `start` out to just past the k-th neighbour."""
    return list(np.linspace(start, radii[k - 1] + 0.03, n_frames))


frames = []


def add(n, **kw):
    for _ in range(n):
        frames.append(dict(kw))


LABEL = {2: 'benign', 4: 'malignant'}

# --- k = 5: set up, grow, freeze on the k-th capture, declare
add(4, mode='vote', k=K_SMALL, r=0.0, show_circle=False)
for r in growth(K_SMALL, 20):
    frames.append(dict(mode='vote', k=K_SMALL, r=r, show_circle=True))
add(6, mode='vote', k=K_SMALL, r=radii[K_SMALL - 1] + 0.03, show_circle=True, frozen=True)
add(8, mode='vote', k=K_SMALL, r=radii[K_SMALL - 1] + 0.03, show_circle=True,
    frozen=True, declare=True)

# --- k = 25: same query, same ball, more of it
add(3, mode='vote', k=K_LARGE, r=0.0, show_circle=False)
for r in growth(K_LARGE, 16):
    frames.append(dict(mode='vote', k=K_LARGE, r=r, show_circle=True))
add(5, mode='vote', k=K_LARGE, r=radii[K_LARGE - 1] + 0.03, show_circle=True, frozen=True)
add(8, mode='vote', k=K_LARGE, r=radii[K_LARGE - 1] + 0.03, show_circle=True,
    frozen=True, declare=True)

# --- the payoff: the whole plane decided at each k
add(12, mode='field')
add(6, mode='field')            # brief hold so the loop does not snap


# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=FIGSIZE, dpi=DPI)

# square box (0.481 * 6.4in == 0.70 * 4.4in) so the query ball draws as a circle
ax = fig.add_axes([0.090, 0.10, 0.481, 0.70])
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0.3, 10.9)
ax.set_ylim(0.3, 10.9)
ax.set_xlabel(feature_names[f1], fontsize=9)
ax.set_ylabel(feature_names[f2], fontsize=9)
ax.tick_params(labelsize=8)

# the training set, drawn once and never redrawn
for group, color in ((2, BENIGN), (4, MALIGNANT)):
    ax.scatter(two_d[group][:, 0], two_d[group][:, 1], color=color, s=12, alpha=0.40,
               linewidths=0, zorder=1)

spokes = LineCollection([], colors='k', linewidths=0.8, alpha=0.55, zorder=3)
ax.add_collection(spokes)
caught = ax.scatter([], [], s=70, facecolors='none', linewidths=1.4, zorder=4)
ball = Circle(point, 0.0, fill=False, edgecolor='k', linewidth=1.6,
              linestyle='--', zorder=5, visible=False)
ax.add_patch(ball)
ax.scatter([point[0]], [point[1]], color='k', marker='*', s=230, zorder=6)
ax.annotate('$x_q$', point, xytext=(9, 6), textcoords='offset points',
            fontsize=10, zorder=6)

# vote tally: one bar per class, filled in as the ball swallows neighbours
ax_vote = fig.add_axes([0.640, 0.135, 0.285, 0.175])
bars = ax_vote.barh([1, 0], [0, 0], color=[MALIGNANT, BENIGN], height=0.62)
ax_vote.set_yticks([0, 1])
ax_vote.set_yticklabels(['benign', 'malig.'], fontsize=8)
ax_vote.set_xticks([])
ax_vote.set_xlim(0, K_SMALL)
ax_vote.set_ylim(-0.55, 1.55)
ax_vote.tick_params(length=0)
bar_labels = [ax_vote.text(0, yy, '', va='center', ha='left', fontsize=8.5)
              for yy in (1, 0)]

eq = fig.text(0.5, 0.945, '', ha='center', va='center', fontsize=11)
head = fig.text(0.090, 0.850, '', ha='left', va='center', fontsize=10)
info = fig.text(0.635, 0.740, '', ha='left', va='top', fontsize=8.5,
                family='monospace', linespacing=1.55)
verdict = fig.text(0.635, 0.365, '', ha='left', va='bottom', fontsize=9)

# --- the two decision fields, built now and revealed at the end
field_axes = []
for n, (kk, left) in enumerate([(K_SMALL, 0.065), (K_LARGE, 0.525)]):
    axf = fig.add_axes([left, 0.095, 0.4125, 0.60])
    axf.set_aspect('equal', adjustable='box')
    axf.contourf(grid, grid, zones[kk], levels=[1, 3, 5],
                 colors=[BENIGN, MALIGNANT], alpha=0.20)
    for group, color in ((2, BENIGN), (4, MALIGNANT)):
        axf.scatter(two_d[group][:, 0], two_d[group][:, 1], color=color, s=5,
                    alpha=0.45, linewidths=0)
    axf.scatter([point[0]], [point[1]], color='k', marker='*', s=110)
    axf.set_xlim(0.3, 10.9)
    axf.set_ylim(0.3, 10.9)
    axf.set_xticks([])
    axf.set_yticks([])
    axf.set_title(f'k = {kk}   ({"jagged" if n == 0 else "smoothed"})', fontsize=10)
    axf.set_visible(False)
    field_axes.append(axf)

vote_only = [ax, ax_vote]


def update(i):
    f = frames[i]

    if f['mode'] == 'field':
        for a in vote_only:
            a.set_visible(False)
        for a in field_axes:
            a.set_visible(True)
        eq.set_text('$\\hat{y}(x)$ everywhere on the plane')
        head.set_text('small $k$ follows every stray point; large $k$ averages them away')
        info.set_text('')
        verdict.set_text('')
        return []

    for a in field_axes:
        a.set_visible(False)
    for a in vote_only:
        a.set_visible(True)

    k, r = f['k'], f['r']
    n = int(np.searchsorted(radii, r))          # neighbours inside the ball
    n = min(n, k)

    ball.set_radius(r)
    ball.set_visible(f.get('show_circle', False))
    ball.set_linestyle('-' if f.get('frozen') else '--')
    ball.set_edgecolor('k' if f.get('frozen') else GREY)

    spokes.set_segments([[point, coords[j]] for j in range(n)])
    if n:
        caught.set_offsets(coords[:n])
        caught.set_edgecolors([MALIGNANT if g == 4 else BENIGN for g in groups[:n]])
    else:
        caught.set_offsets(np.empty((0, 2)))

    n_mal = int((groups[:n] == 4).sum())
    n_ben = n - n_mal
    ax_vote.set_xlim(0, k)
    for bar, w in zip(bars, (n_mal, n_ben)):
        bar.set_width(w)
    for lab, w in zip(bar_labels, (n_mal, n_ben)):
        lab.set_text(str(w) if w else '')
        lab.set_position((w + k * 0.03, lab.get_position()[1]))

    if f.get('declare'):
        eq.set_text('$\\hat{y} = \\arg\\max_c \\sum_{i \\in N_k(x_q)} '
                    '\\mathbb{1}[y_i = c]$')
    else:
        eq.set_text('$d(x_q, x_i) = \\|x_q - x_i\\|_2$')

    if not f.get('show_circle'):
        head.set_text(f'one query point, $k = {k}$: grow $r$ until '
                      f'{k} neighbours are inside')
    elif f.get('frozen'):
        head.set_text(f'$k = {k}$ captured at $r = {radii[k - 1]:.2f}$')
    else:
        head.set_text(f'$r = {r:.2f}$   captured {n} of {k}')

    # running list of capture distances; too many to list once k gets large
    if k <= 6:
        lines = [f'd{j + 1} = {radii[j]:.2f}  {LABEL[groups[j]][0].upper()}'
                 for j in range(n)]
        info.set_text('\n'.join(lines))
    else:
        if n:
            info.set_text(f'nearest {n} of {k}\nd{n} = {radii[n - 1]:.2f}')
        else:
            info.set_text(f'nearest 0 of {k}')

    if f.get('declare'):
        conf = max(n_mal, n_ben) / k
        winner = 4 if n_mal > n_ben else 2
        verdict.set_text(f'$\\hat{{y}}$ = {LABEL[winner]}\n'
                         f'{max(n_mal, n_ben)}/{k} = {conf:.2f} confidence')
        verdict.set_color(MALIGNANT if winner == 4 else BENIGN)
    else:
        verdict.set_text('')

    return []


anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / 12,
                               blit=False)
save_gif(anim, os.path.abspath(OUT), fps=12)
print(f'frames: {len(frames)}')
