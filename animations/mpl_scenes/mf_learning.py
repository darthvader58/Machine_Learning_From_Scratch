"""Matrix factorisation learning on MovieLens 1M.

Three things happen at once and the animation shows them together, because the
relationship between them is the lesson:

  left    the sparse rating matrix filling in as P Q^T becomes able to predict
          the cells it was never shown
  middle  the item factors organising, so films that share a genre drift
          together without any genre label ever being supplied
  right   train and test RMSE separating - the moment the model stops learning
          the data and starts memorising it

Run from the repo root:
    python animations/mpl_scenes/mf_learning.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mpl_style import plt, animation, COLORS   # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), '..', '..')
OUT = os.path.join(REPO, 'animations', 'gifs', 'mf_learning.gif')

K = 16
LR = 0.008
REG = 0.06
EPOCHS = 30
SEED = 0


def load():
    rows = []
    with open(os.path.join(REPO, 'data/movielens/ml-1m/ratings.dat'), encoding='latin-1') as f:
        for line in f:
            a, b, c, _ = line.rstrip('\n').split('::')
            rows.append((int(a), int(b), float(c)))
    rows = np.array(rows)
    ru, ri = rows[:, 0].astype(np.int64), rows[:, 1].astype(np.int64)
    umap = {v: k for k, v in enumerate(np.unique(ru))}
    imap = {v: k for k, v in enumerate(np.unique(ri))}
    u = np.array([umap[x] for x in ru])
    i = np.array([imap[x] for x in ri])

    genre_names = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                   'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                   'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    genres = np.zeros((len(imap), len(genre_names)), dtype=int)
    with open(os.path.join(REPO, 'data/movielens/ml-1m/movies.dat'), encoding='latin-1') as f:
        for line in f:
            mid, _t, gs = line.rstrip('\n').split('::')
            mid = int(mid)
            if mid in imap:
                for g in gs.split('|'):
                    if g in genre_names:
                        genres[imap[mid], genre_names.index(g)] = 1
    return u, i, rows[:, 2], len(umap), len(imap), genres, genre_names


def main():
    u, i, r, n_users, n_items, genres, gnames = load()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(r))
    cut = int(0.8 * len(r))
    tr, te = order[:cut], order[cut:]
    ur, ir, rr = u[tr], i[tr], r[tr]
    ue, ie, re_ = u[te], i[te], r[te]
    mu = rr.mean()

    def rmse(p, t):
        return float(np.sqrt(np.mean((p - t) ** 2)))

    g = np.random.default_rng(SEED)
    P = g.normal(0, 0.1, (n_users, K))
    Q = g.normal(0, 0.1, (n_items, K))
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)

    # a small dense-ish block to watch fill in: the most active users and items
    per_u = np.bincount(ur, minlength=n_users)
    per_i = np.bincount(ir, minlength=n_items)
    bu_idx = np.argsort(-per_u)[:60]
    bi_idx = np.argsort(-per_i)[:60]
    observed = np.full((60, 60), np.nan)
    lookup = {}
    for a, b, v in zip(ur, ir, rr):
        lookup[(a, b)] = v
    for a, uu in enumerate(bu_idx):
        for b, ii in enumerate(bi_idx):
            if (uu, ii) in lookup:
                observed[a, b] = lookup[(uu, ii)]

    popular = np.flatnonzero(per_i >= 200)
    show_g = ['Children\'s', 'Horror', 'Documentary', 'Action']
    gidx = [gnames.index(x) for x in show_g]

    frames = []
    sweep = np.arange(len(tr))
    for ep in range(EPOCHS):
        g.shuffle(sweep)
        for n in sweep:
            a, b, v = ur[n], ir[n], rr[n]
            e = v - (mu + bu[a] + bi[b] + P[a] @ Q[b])
            bu[a] += LR * (e - REG * bu[a])
            bi[b] += LR * (e - REG * bi[b])
            pa = P[a].copy()
            P[a] += LR * (e * Q[b] - REG * pa)
            Q[b] += LR * (e * pa - REG * Q[b])

        tr_e = rmse(np.clip(mu + bu[ur] + bi[ir] + np.sum(P[ur] * Q[ir], 1), 1, 5), rr)
        te_e = rmse(np.clip(mu + bu[ue] + bi[ie] + np.sum(P[ue] * Q[ie], 1), 1, 5), re_)

        recon = np.clip(mu + bu[bu_idx][:, None] + bi[bi_idx][None, :]
                        + P[bu_idx] @ Q[bi_idx].T, 1, 5)

        # 2D view of the item factors, via PCA written out
        Qp = Q[popular]
        X = Qp - Qp.mean(0)
        cov = (X.T @ X) / (len(X) - 1)
        ev, evec = np.linalg.eigh(cov)
        Z = X @ evec[:, np.argsort(-ev)[:2]]
        frames.append((ep + 1, recon.copy(), Z.copy(), tr_e, te_e))
        print(f'  epoch {ep + 1:2d}  train {tr_e:.4f}  test {te_e:.4f}', flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.3), dpi=100)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.80, bottom=0.13, wspace=0.30)

    ax[0].imshow(observed, cmap='viridis', vmin=1, vmax=5)
    ax[0].set_title('what it was shown\n60 busiest users x 60 busiest films', fontsize=9)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    im1 = ax[1].imshow(frames[0][1], cmap='viridis', vmin=1, vmax=5)
    ax[1].set_xticks([]); ax[1].set_yticks([])

    sc = []
    ax[2].set_xlabel('factor component 1', fontsize=8)
    ax[2].set_ylabel('component 2', fontsize=8)

    fig2_ax = ax[2]

    def draw(fi):
        ep, recon, Z, tr_e, te_e = frames[fi]
        im1.set_data(recon)
        ax[1].set_title(r'what $\mu + b_u + b_i + p_u^\top q_i$ predicts'
                        '\n(every cell, including the empty ones)', fontsize=9)
        fig2_ax.clear()
        fig2_ax.scatter(Z[:, 0], Z[:, 1], s=5, color='0.8', zorder=1)
        for c, (nm, gi) in enumerate(zip(show_g, gidx)):
            sel = genres[popular][:, gi] == 1
            fig2_ax.scatter(Z[sel, 0], Z[sel, 1], s=13, alpha=0.85,
                            color=COLORS[c], label=nm, zorder=2)
        fig2_ax.set_title('item factors, coloured by genre afterwards\n'
                          'no genre was ever shown to the model', fontsize=9)
        fig2_ax.set_xlabel('factor component 1', fontsize=8)
        fig2_ax.set_ylabel('component 2', fontsize=8)
        fig2_ax.tick_params(labelsize=7)
        fig2_ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
        gap = te_e - tr_e
        fig.suptitle(f'epoch {ep} of {EPOCHS}      train RMSE {tr_e:.4f}      '
                     f'test RMSE {te_e:.4f}      gap {gap:+.4f}',
                     fontsize=11)
        return [im1]

    seq = list(range(len(frames))) + [len(frames) - 1] * 6
    ani = animation.FuncAnimation(fig, draw, frames=seq, interval=140, blit=False)

    # Write mp4 first, then let ffmpeg build one optimal palette for the whole
    # clip. Pillow's per-frame palette costs about 4x on a heatmap this noisy.
    import subprocess
    import tempfile
    mp4 = os.path.join(tempfile.gettempdir(), 'mf_learning.mp4')
    ani.save(mp4, writer='ffmpeg', fps=7, dpi=100)
    subprocess.run(['bash', os.path.join(REPO, 'animations', 'gif_from_mp4.sh'),
                    mp4, OUT, '820', '7'], check=True)
    print('size: %.2f MB' % (os.path.getsize(OUT) / 1e6))


if __name__ == '__main__':
    main()
