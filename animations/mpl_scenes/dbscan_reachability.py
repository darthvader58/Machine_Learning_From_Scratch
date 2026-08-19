"""DBSCAN: density reachability, and how a cluster grows out of one core point.

The notebook `clustering/dbscan.ipynb` runs on the UCI Wholesale customers file
(id 292): 440 clients, six spending categories, `np.log1p` then standardised.  It
fixes `min_samples = 12` (the 2d rule of thumb for six dimensions) and picks eps off
the k-distance knee, and it draws its results in the plane of the two leading
eigenvectors of the covariance of `X` - the `project_2d` helper in cell 27.

This animation runs the notebook's own `DBSCAN` class in that drawn plane rather than
in all six dimensions.  That is the one deliberate change, and it is what makes the
picture honest: the eps-circle on screen really is the neighbourhood being counted, so
the number next to it is the number of dots inside it.  A six-dimensional ball
projected onto a plane would look like a circle holding far more points than it
counts.

Parameters follow the notebook's own procedure applied in that plane:
  * `min_samples = 12`, exactly the notebook's value
  * the k-distance knee in the plane lands at eps = 0.85, which reproduces the
    notebook's knee answer almost exactly (one cluster, 29 clients refused against the
    notebook's 27) - and one cluster shows nothing growing
  * so, as the notebook itself does at `EPS_TIGHT`, this uses the tighter value at the
    top of the window where the dense cores are still separate: eps = 0.55.  That
    gives 2 clusters, 321 core, 57 border and 62 noise - the notebook's headline
    shape of answer, two dense cores and a large unassigned set.

What is on screen:
  * an eps-circle visits three points in turn and the count inside it is compared with
    minPts, giving the verdict core, border or noise
  * then the real breadth-first expansion is replayed from the class's own queue: an
    unclaimed core point seeds a cluster, its eps-neighbours join, and each new core
    point pushes its own neighbours on, so the cluster creeps along the dense region
    and stops where the density runs out
  * the final frame is the finished clustering, with everything density-reachable from
    nothing drawn as black crosses.

Run from the repo root:
    /opt/anaconda3/envs/tf_mps/bin/python animations/mpl_scenes/dbscan_reachability.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mpl_style import plt, animation, COLORS, FIGSIZE, DPI, save_gif  # noqa: E402

from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from ucimlrepo import fetch_ucirepo  # noqa: E402

np.random.seed(42)          # the notebook's seed; nothing here is random anyway

OUT = os.path.join(os.path.dirname(__file__), '..', 'gifs', 'dbscan_reachability.gif')
MIN_SAMPLES = 12            # the notebook's min_samples
EPS = 0.55                  # the notebook's tighter operating point, in the drawn plane
FPS = 10

GREY_PT = '#b6b6b6'
INK = '#222222'
CORE_C = COLORS[2]          # green verdict
BORDER_C = COLORS[3]        # amber verdict
NOISE_C = '#111111'


# ---------------------------------------------------------------- the notebook's code
def pairwise_distances(a):
    sq = np.sum(a ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (a @ a.T)
    np.maximum(d2, 0, out=d2)
    return np.sqrt(d2)


class DBSCAN:
    """The notebook's class verbatim, with one list appended to.

    `self.trace` records every seed and every point the queue pulls in, in the order
    the loop does it.  The clustering itself is untouched.
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, data):
        n = len(data)
        d = pairwise_distances(data)

        # eps-neighbourhood of every point, self included
        neighbours = [np.flatnonzero(d[i] <= self.eps) for i in range(n)]
        counts = np.array([len(v) for v in neighbours])
        self.core_points = counts >= self.min_samples

        labels = np.full(n, -1)
        cluster_id = 0
        self.trace = []
        self.counts = counts
        self.neighbours = neighbours

        for i in range(n):
            # only an unclaimed core point can seed a cluster
            if not self.core_points[i] or labels[i] != -1:
                continue

            labels[i] = cluster_id
            queue = list(neighbours[i])
            head = 0
            self.trace.append(('seed', i, cluster_id, True, len(queue)))

            # breadth-first expansion over everything density-reachable
            while head < len(queue):
                j = queue[head]
                head += 1
                if labels[j] == -1:
                    labels[j] = cluster_id
                    # only core points extend the chain further
                    if self.core_points[j]:
                        queue.extend(neighbours[j])
                    self.trace.append(('join', int(j), cluster_id,
                                       bool(self.core_points[j]), len(queue) - head))

            self.trace.append(('done', i, cluster_id, True, 0))
            cluster_id += 1

        self.labels = labels
        self.n_clusters = cluster_id
        self.border_points = (labels >= 0) & ~self.core_points
        self.noise_points = labels == -1
        return self


# ---------------------------------------------------------------- the data
wholesale = fetch_ucirepo(id=292)
frame = wholesale.data.original
spend_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
X_raw = frame[spend_cols].values.astype(float)

# the notebook's transform: log1p to kill the right skew, then standardise so one eps
# can work across all six categories
X_log = np.log1p(X_raw)
X = (X_log - X_log.mean(axis=0)) / X_log.std(axis=0)


def project_2d(a):
    """The notebook's projection: the two directions of largest variance."""
    cov = np.cov(a, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    return vecs[:, order[:2]], vals[order] / vals.sum()


axes_2d, var_ratio = project_2d(X)
P = X @ axes_2d
VAR = 100 * var_ratio[:2].sum()

db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(P)
counts = db.counts
print('eps = %.2f  min_samples = %d  ->  %d clusters, %d core, %d border, %d noise'
      % (EPS, MIN_SAMPLES, db.n_clusters, db.core_points.sum(),
         db.border_points.sum(), db.noise_points.sum()))

# ---------------------------------------------------------------- the three sample points
# one of each verdict, picked to sit somewhere uncrowded so the circle stays readable
DEMO_CORE = 27      # 30 neighbours, well inside the dense mass
DEMO_BORDER = 40    # 10 neighbours, on the rim - it joins a cluster but never extends one
DEMO_NOISE = 47     # 1 neighbour, itself, out in the empty corner
assert db.core_points[DEMO_CORE] and db.border_points[DEMO_BORDER] and db.noise_points[DEMO_NOISE]


# ---------------------------------------------------------------- the figure
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
LEFT, RIGHT, BOTTOM, TOP = 0.085, 0.585, 0.085, 0.775
fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP)

# the axes box has to be set to the data's own aspect, otherwise an eps-circle would be
# drawn as an ellipse and the whole picture would lie about distance
box_ratio = ((RIGHT - LEFT) * FIGSIZE[0]) / ((TOP - BOTTOM) * FIGSIZE[1])
cx, cy = P[:, 0].mean(), P[:, 1].mean()
half_y = max(np.abs(P[:, 1] - cy).max(), np.abs(P[:, 0] - cx).max() / box_ratio) + 0.45
half_x = half_y * box_ratio
ax.set_xlim(cx - half_x, cx + half_x)
ax.set_ylim(cy - half_y, cy + half_y)
ax.set_aspect('equal')
ax.set_xlabel('first direction of the standardised log spending', fontsize=8)
ax.set_ylabel('second direction', fontsize=8)
ax.tick_params(labelsize=7)

fig.suptitle('DBSCAN: a cluster grows by density reachability', fontsize=11.5, y=0.982)
eq = fig.text(0.5, 0.905, '', ha='center', va='center', fontsize=11.5, color=INK)
# two short lines rather than one long one, so nothing runs off a 640px frame
fig.text(0.5, 0.828,
         '440 wholesale clients, log-standardised spending, drawn in its two\n'
         'leading directions (%.0f%% of the variance)  -  $\\varepsilon = %.2f$,  '
         'minPts $= %d$' % (VAR, EPS, MIN_SAMPLES),
         ha='center', va='center', fontsize=7.6, color='#555555', linespacing=1.5)

# one scatter for every client; colours, sizes and edges are rewritten each frame
pts = ax.scatter(P[:, 0], P[:, 1], s=np.full(len(P), 12.0), zorder=3,
                 facecolors=np.tile(to_rgba(GREY_PT), (len(P), 1)),
                 edgecolors=np.tile(to_rgba(GREY_PT), (len(P), 1)), linewidths=0.0)
# noise only becomes noise once the run is over, so it gets its own marker at the end
noise_marks = ax.scatter([], [], marker='x', s=26, color=NOISE_C, linewidths=1.1, zorder=4)

ring = Circle((0, 0), EPS, facecolor=to_rgba(COLORS[0], 0.13), edgecolor=INK,
              lw=1.6, zorder=5, visible=False)
ax.add_patch(ring)
centre = ax.scatter([], [], s=34, color=INK, zorder=7)

caption = ax.text(0.985, 0.025, '', transform=ax.transAxes, ha='right', va='bottom',
                  fontsize=8.5, color=INK,
                  bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#999999', alpha=0.9))

# the right-hand column: the verdict, then a line saying why
verdict = fig.text(0.79, 0.63, '', ha='center', va='center', fontsize=15, color=INK)
because = fig.text(0.79, 0.53, '', ha='center', va='top', fontsize=8.5, color='#444444',
                   linespacing=1.6, wrap=False)

handles = [
    Line2D([], [], ls='', marker='o', ms=5, mfc=COLORS[0], mec='white', label='cluster 0'),
    Line2D([], [], ls='', marker='o', ms=5, mfc=COLORS[1], mec='white', label='cluster 1'),
    Line2D([], [], ls='', marker='o', ms=5, mfc='white', mec=INK, mew=1.1, label='border point'),
    Line2D([], [], ls='', marker='x', ms=5, color=NOISE_C, label='noise (no cluster)'),
    Line2D([], [], ls='', marker='o', ms=4, mfc=GREY_PT, mec=GREY_PT, label='not yet reached'),
]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.79, 0.09),
           fontsize=7.5, frameon=True, handletextpad=0.5, borderpad=0.6, labelspacing=0.55)


# ---------------------------------------------------------------- the storyboard
def eq_text(k):
    """The core test with the live count dropped in."""
    rel = r'\geq' if k >= MIN_SAMPLES else '<'
    return r'$|N_\varepsilon(p)| = %d \;%s\; \mathrm{minPts} = %d$' % (k, rel, MIN_SAMPLES)


EQ_PLAIN = r'$|N_\varepsilon(p)| \;\geq\; \mathrm{minPts} = %d$' % MIN_SAMPLES

storyboard = []          # one dict per frame
blank = np.full(len(P), -1)


def add(labels, focus=None, halo=(), halo_fill=None, eq_s=EQ_PLAIN, cap='',
        verd=('', INK), why='', final=False, repeat=1):
    for _ in range(repeat):
        storyboard.append(dict(labels=labels, focus=focus, halo=np.asarray(halo, dtype=int),
                               halo_fill=halo_fill, eq=eq_s, cap=cap, verd=verd, why=why,
                               final=final))


# --- opening: 440 clients, nothing decided
add(blank, cap='440 clients, nothing labelled yet', repeat=4,
    why='DBSCAN asks one question of\nevery point: how many others\nlie within $\\varepsilon$ of it?')

# --- the three verdicts, one point at a time
demos = [
    (DEMO_CORE, 'CORE', CORE_C,
     'at least minPts inside, so it\ncan seed a cluster and pass\nthe chain along'),
    (DEMO_BORDER, 'BORDER', BORDER_C,
     'under minPts, but inside a core\npoint\'s circle - it joins that\ncluster and extends it no further'),
    (DEMO_NOISE, 'NOISE', NOISE_C,
     'under minPts and inside nobody\nelse\'s circle, so no chain ever\nreaches it'),
]
for idx, name, col, why in demos:
    nb = db.neighbours[idx]
    k = int(counts[idx])
    add(blank, focus=idx, eq_s=eq_text(k), repeat=2,
        cap='counting $N_\\varepsilon(p)$ ...', why='')
    add(blank, focus=idx, halo=nb, halo_fill=col, eq_s=eq_text(k), repeat=3,
        cap='%d points within $\\varepsilon$' % k, verd=(name, col), why=why)

# --- the expansion, replayed from the class's own queue
labels_now = np.full(len(P), -1)
sizes_by_cluster = [int((db.labels == c).sum()) for c in range(db.n_clusters)]
JOINS_PER_FRAME = {0: 15, 1: 7}       # tuned so each cluster gets a readable number of frames

focus = None
chunk, joined = [], 0
for kind, idx, cid, is_core, waiting in db.trace:
    if kind == 'seed':
        labels_now = labels_now.copy()
        labels_now[idx] = cid
        focus = idx
        add(labels_now, focus=idx, halo=[idx], eq_s=eq_text(int(counts[idx])), repeat=2,
            cap='cluster %d seeded - %d neighbours queued' % (cid, waiting),
            verd=('CORE', CORE_C),
            why='an unclaimed core point starts\na new cluster and pushes its\n'
                '$\\varepsilon$-neighbours onto the queue')
        chunk, joined = [], 0
        continue

    if kind == 'done':
        if chunk:                      # flush whatever is left in the part-filled frame
            labels_now = labels_now.copy()
            labels_now[chunk] = cid
            add(labels_now, focus=focus, halo=chunk, eq_s=eq_text(int(counts[focus])),
                cap='queue: %d waiting' % waiting, verd=('CLUSTER %d' % cid, COLORS[cid]),
                why='every new core point pushes its\nown neighbours on, so the chain\nkeeps going')
            chunk, joined = [], 0
        add(labels_now, focus=None, repeat=3,
            cap='queue empty - cluster %d closed at %d clients'
                % (cid, sizes_by_cluster[cid]),
            verd=('STOP', INK),
            why='nothing left is density-reachable:\nthe chain died on border points\nand the region ran out')
        continue

    # a point the queue pulled in
    chunk.append(idx)
    joined += 1
    if is_core:
        focus = idx
    if joined >= JOINS_PER_FRAME[cid]:
        labels_now = labels_now.copy()
        labels_now[chunk] = cid
        add(labels_now, focus=focus, halo=chunk, eq_s=eq_text(int(counts[focus])),
            cap='queue: %d waiting' % waiting, verd=('CLUSTER %d' % cid, COLORS[cid]),
            why='every point popped off the queue\njoins; core ones push their own\nneighbours on behind them')
        chunk, joined = [], 0

# --- the finished clustering
add(db.labels, eq_s=EQ_PLAIN, final=True, repeat=8,
    cap='%d clusters, %d clients refused' % (db.n_clusters, int(db.noise_points.sum())),
    verd=('DONE', INK),
    why='%d core, %d border, %d noise\n%.0f%% of the file was left\nunassigned' % (
        db.core_points.sum(), db.border_points.sum(), db.noise_points.sum(),
        100 * db.noise_points.mean()))

N_FRAMES = len(storyboard)
print('frames:', N_FRAMES)


# ---------------------------------------------------------------- drawing
GREY_RGBA = to_rgba(GREY_PT)
CLUSTER_RGBA = [to_rgba(COLORS[c]) for c in range(db.n_clusters)]
WHITE = to_rgba('white')
INK_RGBA = to_rgba(INK)


def draw(f):
    s = storyboard[f]
    labels = s['labels']

    face = np.tile(GREY_RGBA, (len(P), 1))
    edge = np.tile(GREY_RGBA, (len(P), 1))
    size = np.full(len(P), 12.0)
    lw = np.zeros(len(P))

    for c in range(db.n_clusters):
        sel = labels == c
        if not sel.any():
            continue
        # core points are filled, border points are hollow: the fill is what carries the
        # chain on, the ring is where it stops
        core_sel = sel & db.core_points
        bord_sel = sel & ~db.core_points
        face[core_sel] = CLUSTER_RGBA[c]
        edge[core_sel] = WHITE
        size[core_sel] = 16.0
        lw[core_sel] = 0.3
        face[bord_sel] = WHITE
        edge[bord_sel] = CLUSTER_RGBA[c]
        size[bord_sel] = 18.0
        lw[bord_sel] = 1.1

    # under the circle during the three verdicts, the neighbours are filled in with the
    # verdict colour; during the expansion the points just pulled in get a dark ring
    if len(s['halo']):
        h = s['halo']
        if s['halo_fill'] is not None:
            face[h] = to_rgba(s['halo_fill'])
            edge[h] = WHITE
            size[h] = 24.0
            lw[h] = 0.4
        else:
            edge[h] = INK_RGBA
            size[h] = np.maximum(size[h], 18.0)
            lw[h] = 0.9

    if s['final']:
        # noise is only knowable once the whole run is over
        n_sel = db.noise_points
        size[n_sel] = 0.0
        noise_marks.set_offsets(P[n_sel])
    else:
        noise_marks.set_offsets(np.empty((0, 2)))

    pts.set_facecolor(face)
    pts.set_edgecolor(edge)
    pts.set_sizes(size)
    pts.set_linewidth(lw)

    if s['focus'] is None:
        ring.set_visible(False)
        centre.set_offsets(np.empty((0, 2)))
    else:
        ring.set_visible(True)
        ring.set_center(P[s['focus']])
        centre.set_offsets(P[s['focus']][None, :])

    eq.set_text(s['eq'])
    caption.set_text(s['cap'])
    verdict.set_text(s['verd'][0])
    verdict.set_color(s['verd'][1])
    because.set_text(s['why'])
    return [pts, noise_marks, ring, centre, eq, caption, verdict, because]


anim = animation.FuncAnimation(fig, draw, frames=N_FRAMES, interval=1000 // FPS, blit=False)
save_gif(anim, os.path.normpath(OUT), fps=FPS)
