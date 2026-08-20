"""Why L1 gives exact zeros and L2 does not - the constraint-region picture.

The notebook makes this argument in prose: the RSS level sets are ellipses centred on the
unconstrained least squares solution, the L2 budget region is a disc and the L1 budget region
is a diamond with its corners ON the axes, and the constrained solution is where the smallest
ellipse that still reaches the region touches it. A smooth disc is touched at a generic point
with every coordinate non-zero; a diamond is touched at a corner, and a corner has a coordinate
equal to zero exactly.

Everything drawn here is computed, not drawn by hand:

  * the two panels are real two-column subproblems of the same UCI Communities and Crime data
    the notebook fits, standardised on the notebook's own 60% training split,
  * beta_hat is the least squares solution of that subproblem,
  * the contact point on the diamond is the fit of the notebook's `Lasso_Regression` class,
    copied verbatim, with lambda found by bisection so that ||beta||_1 lands exactly on the
    budget t,
  * the contact point on the disc is the notebook's `Ridge_Regression` closed form, with lambda
    bisected the same way so that ||beta||_2 lands on the same t,
  * the ellipse that touches is the exact RSS level set through the solution it touches, so the
    tangency in the picture is the tangency in the algebra.

Run with no arguments from the repo root to rebuild the GIF:

    /opt/anaconda3/envs/tf_mps/bin/python animations/manim_scenes/lasso_l1_geometry.py
"""
import glob
import os
import stat
import subprocess
import sys
import tempfile

import numpy as np
from manim import *

NAME = 'lasso_l1_geometry'
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The repo's shared palette, on the light background the matplotlib scenes use.
INK = '#33333a'
BLUE = '#2a78d6'      # the RSS ellipses and beta_hat
ORANGE = '#eb6834'    # everything L1
AQUA = '#1baf7a'      # everything L2
GOLD = '#eda100'      # contact points
PURPLE = '#4a3aa7'    # elastic net
PANEL = '#e8e8ea'
GRID = '#ffffff'

config.background_color = '#fbfbfc'

# The two column pairs drawn, chosen from the 4950 available pairs because their least squares
# solutions sit in different quadrants and zero out different coordinates.
PAIRS = [('pctWPubAsst', 'PctIlleg'), ('pctWInvInc', 'PctNotHSGrad')]
BUDGET_FRACTION = 0.55        # t as a fraction of ||beta_hat||_1
VIEW = 1.1                    # the beta-plane drawn runs from -VIEW to +VIEW on both axes


# ----------------------------------------------------------------------------------------
# the notebook's own code, copied verbatim
# ----------------------------------------------------------------------------------------
def soft_threshold(z, gamma):
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)


class Lasso_Regression:
    def __init__(self, lam=1.0, max_iter=500, tol=1e-6):
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.beta = None
        self.intercept = 0.0
        self.history = []
        self.sweeps = 0

    def fit(self, X, y, beta_init=None):
        n, d = X.shape
        col_sq = (X ** 2).sum(axis=0)

        beta = np.zeros(d) if beta_init is None else beta_init.copy()
        intercept = y.mean() - X.mean(axis=0) @ beta
        resid = y - X @ beta - intercept

        self.history = []
        for sweep in range(self.max_iter):
            # unpenalised intercept: its optimum is the residual mean
            shift = resid.mean()
            intercept += shift
            resid -= shift

            max_change = 0.0
            for j in range(d):
                old = beta[j]
                rho = X[:, j] @ resid + col_sq[j] * old
                new = soft_threshold(rho, self.lam / 2.0) / col_sq[j]
                if new != old:
                    resid -= (new - old) * X[:, j]
                    beta[j] = new
                    change = abs(new - old)
                    if change > max_change:
                        max_change = change

            self.history.append(np.sum(resid ** 2) + self.lam * np.abs(beta).sum())
            if max_change < self.tol:
                break

        self.beta = beta
        self.intercept = intercept
        self.sweeps = sweep + 1
        return self


class Ridge_Regression:
    def __init__(self, lam=1.0):
        self.lam = lam
        self.beta = None
        self.intercept = 0.0

    def fit(self, X, y):
        n, d = X.shape
        Xb = np.hstack([np.ones((n, 1)), X])

        # identity with a zero in the intercept slot, so the intercept is not penalised
        P = np.eye(d + 1)
        P[0, 0] = 0.0

        beta_full = np.linalg.solve(Xb.T @ Xb + self.lam * P, Xb.T @ y)
        self.intercept = beta_full[0]
        self.beta = beta_full[1:]
        return self


# ----------------------------------------------------------------------------------------
# data: the notebook's cleaning, split and scaling, then the two-column subproblems
# ----------------------------------------------------------------------------------------
def load_training_matrix():
    """Reproduce the notebook's cleaned, split and standardised training set."""
    from ucimlrepo import fetch_ucirepo

    crime = fetch_ucirepo(id=183)
    frame = crime.data.features.replace('?', np.nan)
    frame = frame.drop(columns=['state', 'county', 'community', 'communityname', 'fold'])
    frame = frame.astype(float)

    # the LEMAS survey columns are missing for most communities; drop them whole
    missing_frac = frame.isna().mean()
    frame = frame.drop(columns=list(missing_frac[missing_frac > 0.2].index))

    names = list(frame.columns)
    keep = ~(frame.isna().any(axis=1).values | crime.data.targets.isna().any(axis=1).values)
    X_all = frame.values[keep].astype(float)
    y_all = crime.data.targets['ViolentCrimesPerPop'].values[keep].astype(float)

    # the notebook's split: RandomState(0), first 60% of the shuffled rows are the training set
    rng = np.random.RandomState(0)
    order = rng.permutation(len(X_all))
    train_idx = order[:int(0.6 * len(X_all))]

    X_mean = X_all[train_idx].mean(axis=0)
    X_std = X_all[train_idx].std(axis=0)
    X_std[X_std == 0] = 1.0
    X_train = (X_all[train_idx] - X_mean) / X_std

    # the target is centred as in the notebook, then divided by its own standard deviation so
    # the coefficients of a two-column fit come out around 0.5 rather than 0.1 and the plot
    # axes carry readable numbers. Scaling y multiplies every beta by one constant, so it
    # changes nothing about the geometry.
    y_train = y_all[train_idx] - y_all[train_idx].mean()
    y_train = y_train / y_train.std()
    return X_train, y_train, names


def lasso_at_budget(X, y, t):
    """Smallest RSS with ||beta||_1 <= t, by bisecting lambda on the notebook's lasso."""
    lo, hi = 0.0, 2.0 * np.abs(X.T @ y).max()      # the notebook's lambda_max zeroes everything
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.abs(Lasso_Regression(lam=mid).fit(X, y).beta).sum() > t:
            lo = mid
        else:
            hi = mid
    return Lasso_Regression(lam=0.5 * (lo + hi)).fit(X, y).beta


def ridge_at_budget(X, y, t):
    """Smallest RSS with ||beta||_2 <= t, by bisecting lambda on the ridge closed form."""
    lo, hi = 0.0, 1e8
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.linalg.norm(Ridge_Regression(lam=mid).fit(X, y).beta) > t:
            lo = mid
        else:
            hi = mid
    return Ridge_Regression(lam=0.5 * (lo + hi)).fit(X, y).beta


def subproblem(X_train, y_train, names, pair):
    """Everything the picture needs for one two-column subproblem."""
    cols = [names.index(pair[0]), names.index(pair[1])]
    X = X_train[:, cols]
    G = X.T @ X                                   # RSS(beta) = const + (beta-bhat)' G (beta-bhat)
    bhat = np.linalg.solve(G, X.T @ y_train)      # unconstrained least squares
    t = BUDGET_FRACTION * np.abs(bhat).sum()      # the budget, as a fraction of the OLS L1 norm

    b_lasso = lasso_at_budget(X, y_train, t)
    b_ridge = ridge_at_budget(X, y_train, t)

    def level(b):                                 # RSS above the minimum at beta
        d = b - bhat
        return float(d @ G @ d)

    return dict(pair=pair, G=G, bhat=bhat, t=t,
                b_lasso=b_lasso, b_ridge=b_ridge,
                c_lasso=level(b_lasso), c_ridge=level(b_ridge))


def ellipse_points(bhat, G, c, n=180):
    """Points on the RSS level set (beta - bhat)' G (beta - bhat) = c."""
    w, V = np.linalg.eigh(G)
    th = np.linspace(0.0, 2.0 * np.pi, n)
    unit = np.stack([np.cos(th), np.sin(th)])
    return (bhat[:, None] + V @ (np.sqrt(c / w)[:, None] * unit)).T


# ----------------------------------------------------------------------------------------
# the scene
# ----------------------------------------------------------------------------------------
class LassoL1Geometry(Scene):

    def construct(self):
        Tex.set_default(color=INK)
        MathTex.set_default(color=INK)
        Text.set_default(color=INK)

        X_train, y_train, names = load_training_matrix()
        self.probs = [subproblem(X_train, y_train, names, p) for p in PAIRS]
        for p in self.probs:
            print('%-22s %-22s bhat=(%+.3f,%+.3f) t=%.3f lasso=(%+.3f,%+.3f) ridge=(%+.3f,%+.3f)'
                  % (p['pair'][0], p['pair'][1], p['bhat'][0], p['bhat'][1], p['t'],
                     p['b_lasso'][0], p['b_lasso'][1], p['b_ridge'][0], p['b_ridge'][1]))

        self.build_frame()
        self.act_contours()
        self.act_regions()
        self.act_touch(self.probs[0], first=True)
        self.act_second_beta_hat()
        self.act_touch(self.probs[1], first=False)
        self.act_why()
        self.act_algebra()

    # -- scaffolding ---------------------------------------------------------------------
    def build_frame(self):
        """Headline objective, two beta-planes, one caption line at the bottom."""
        self.objective = MathTex(r'\min_\beta\ \lVert y - X\beta\rVert_2^2',
                                 r'\quad\text{subject to}\quad', font_size=34).move_to(UP * 3.35)
        self.objective[0].set_color(BLUE)

        self.axL = self.make_axes(LEFT * 3.45)
        self.axR = self.make_axes(RIGHT * 3.45)

        self.labL = MathTex(r'\lVert\beta\rVert_2 \le t', font_size=36, color=AQUA)
        self.labL.move_to(self.axL[0].get_top() + UP * 0.62)
        self.labR = MathTex(r'\lVert\beta\rVert_1 \le t', font_size=36, color=ORANGE)
        self.labR.move_to(self.axR[0].get_top() + UP * 0.62)

        # an invisible placeholder so the first say() has something to fade out
        self.caption = Dot(radius=0.001, fill_opacity=0.0).move_to(DOWN * 3.42)

        self.play(FadeIn(self.objective), Create(self.axL), Create(self.axR), run_time=0.45)
        self.add(self.caption)

    def make_axes(self, shift):
        """A ggplot-style grey panel with a beta_1 / beta_2 coordinate frame on it."""
        ax = Axes(x_range=[-VIEW, VIEW, 0.5], y_range=[-VIEW, VIEW, 0.5],
                  x_length=4.25, y_length=4.25,
                  axis_config=dict(color=INK, stroke_width=2.2, include_ticks=True,
                                   tick_size=0.05, include_tip=False))
        panel = Rectangle(width=4.25, height=4.25, fill_color=PANEL, fill_opacity=1.0,
                          stroke_width=0)
        grid = VGroup(*[Line(ax.c2p(-VIEW, v), ax.c2p(VIEW, v), color=GRID, stroke_width=1.6)
                        for v in (-0.5, 0.0, 0.5)],
                      *[Line(ax.c2p(v, -VIEW), ax.c2p(v, VIEW), color=GRID, stroke_width=1.6)
                        for v in (-0.5, 0.0, 0.5)])
        xl = MathTex(r'\beta_1', font_size=28).next_to(ax.c2p(VIEW, 0), DR, buff=0.08)
        yl = MathTex(r'\beta_2', font_size=28).next_to(ax.c2p(0, VIEW), UR, buff=0.05)
        group = VGroup(panel, grid, ax, xl, yl).shift(shift)
        return group

    def plane(self, group):
        """The Axes inside a panel group, for coordinate conversion."""
        return group[2]

    def say(self, tex, run_time=0.3):
        new = Tex(tex, font_size=30).move_to(DOWN * 3.42)
        self.play(FadeOut(self.caption, shift=DOWN * 0.12),
                  FadeIn(new, shift=DOWN * 0.12), run_time=run_time)
        self.caption = new

    def dot_at(self, group, b, color, radius=0.055):
        return Dot(self.plane(group).c2p(b[0], b[1]), radius=radius, color=color, z_index=6)

    def ellipse(self, group, prob, c):
        pts = ellipse_points(prob['bhat'], prob['G'], max(c, 1e-9))
        curve = VMobject(stroke_color=BLUE, stroke_width=3.0, z_index=4)
        curve.set_points_as_corners([self.plane(group).c2p(x, y) for x, y in pts])
        return curve

    # -- act 1: the level sets are ellipses centred on beta_hat ---------------------------
    def act_contours(self):
        p = self.probs[0]
        self.play(FadeIn(self.labL), FadeIn(self.labR), run_time=0.3)

        self.bh_dots = VGroup(*[self.dot_at(g, p['bhat'], BLUE) for g in (self.axL, self.axR)])
        self.bh_lab = MathTex(r'\hat\beta', font_size=32, color=BLUE)
        self.bh_lab.next_to(self.bh_dots[1], UR, buff=0.06)

        self.say(r'least squares lands at $\hat\beta$ --- the squared error grows outward '
                 r'from it in ellipses')
        self.play(FadeIn(self.bh_dots), FadeIn(self.bh_lab), run_time=0.3)

        self.faint = VGroup()
        for frac in (0.18, 0.45, 0.85):
            for g in (self.axL, self.axR):
                e = self.ellipse(g, p, frac * p['c_lasso'])
                e.set_stroke(color=BLUE, width=1.6, opacity=0.35)
                self.faint.add(e)
        self.play(Create(self.faint), run_time=0.4)
        self.wait(0.25)

    # -- act 2: the two constraint regions ------------------------------------------------
    def act_regions(self):
        p = self.probs[0]
        self.say(r'the budget $t$ confines $\beta$ to a region: a disc for L2, '
                 r'a diamond for L1')
        self.play(FadeOut(self.faint), run_time=0.3)

        self.disc = self.make_disc(p['t'])
        self.diamond = self.make_diamond(p['t'])
        self.corner_lab = MathTex(r'(t,\,0)', font_size=26, color=ORANGE)
        self.corner_lab.next_to(self.plane(self.axR).c2p(p['t'], 0), DR, buff=0.06)

        self.play(DrawBorderThenFill(self.disc), DrawBorderThenFill(self.diamond), run_time=0.35)
        self.play(FadeIn(self.corner_lab), run_time=0.3)
        self.wait(0.25)

    def make_disc(self, t):
        ax = self.plane(self.axL)
        r = np.linalg.norm(ax.c2p(t, 0) - ax.c2p(0, 0))
        return Circle(radius=r, color=AQUA, fill_color=AQUA, fill_opacity=0.16,
                      stroke_width=3).move_to(ax.c2p(0, 0)).set_z_index(2)

    def make_diamond(self, t):
        ax = self.plane(self.axR)
        return Polygon(ax.c2p(t, 0), ax.c2p(0, t), ax.c2p(-t, 0), ax.c2p(0, -t),
                       color=ORANGE, fill_color=ORANGE, fill_opacity=0.16,
                       stroke_width=3).set_z_index(2)

    # -- act 3: grow the ellipse until it first touches each region ------------------------
    def act_touch(self, p, first):
        if first:
            self.say(r'grow the ellipse out of $\hat\beta$ until it first reaches each region')

        c = ValueTracker(0.02 * p['c_ridge'])
        growL = always_redraw(lambda: self.ellipse(self.axL, p, c.get_value()))
        growR = always_redraw(lambda: self.ellipse(self.axR, p, c.get_value()))
        self.add(growL, growR)

        # first contact on the smooth disc
        self.play(c.animate.set_value(p['c_ridge']), run_time=0.65, rate_func=rate_functions.ease_in_out_sine)
        self.remove(growL)
        frozen = self.ellipse(self.axL, p, p['c_ridge'])
        hitL = self.dot_at(self.axL, p['b_ridge'], GOLD, radius=0.07)
        self.add(frozen, hitL)
        self.play(Flash(hitL, color=GOLD, line_length=0.13, flash_radius=0.2), run_time=0.3)

        readL = MathTex(r'\hat\beta_{\text{ridge}} = (%.3f,\ %.3f)' % tuple(p['b_ridge']),
                        font_size=26, color=AQUA)
        readL.move_to(self.axL[0].get_bottom() + DOWN * 0.35)
        self.play(FadeIn(readL), run_time=0.3)
        if first:
            self.say(r'the disc is smooth, so it is touched at a generic point --- '
                     r'both coordinates non-zero')
            self.wait(0.25)

        # the diamond sits inside the disc, so the ellipse has to grow further to reach it
        self.play(c.animate.set_value(p['c_lasso']), run_time=0.55, rate_func=rate_functions.ease_in_out_sine)
        self.remove(growR)
        frozenR = self.ellipse(self.axR, p, p['c_lasso'])
        hitR = self.dot_at(self.axR, p['b_lasso'], GOLD, radius=0.07)
        self.add(frozenR, hitR)
        self.play(Flash(hitR, color=GOLD, line_length=0.16, flash_radius=0.26), run_time=0.3)

        zero_first = abs(p['b_lasso'][0]) < 1e-12
        parts = [r'\hat\beta_{\text{lasso}} = (', r'%.3f' % p['b_lasso'][0], r',\ ',
                 r'%.3f' % p['b_lasso'][1], r')']
        readR = MathTex(*parts, font_size=26, color=ORANGE)
        readR[1 if zero_first else 3].set_color(GOLD)
        readR.move_to(self.axR[0].get_bottom() + DOWN * 0.35)
        self.play(FadeIn(readR), run_time=0.3)

        which = r'\beta_1 = 0' if zero_first else r'\beta_2 = 0'
        self.say(r'the diamond is touched \emph{at a corner} --- $%s$ exactly, not merely small'
                 % which)
        self.play(Indicate(readR[1 if zero_first else 3], color=GOLD, scale_factor=1.5),
                  run_time=0.3)
        self.wait(0.9 if first else 1.1)

        self.contact = VGroup(frozen, frozenR, hitL, hitR, readL, readR)

    # -- act 4: a different beta_hat, a different pair of columns --------------------------
    def act_second_beta_hat(self):
        p = self.probs[1]
        self.say(r'a different pair of columns puts $\hat\beta$ somewhere else entirely')

        new_disc = self.make_disc(p['t'])
        new_diamond = self.make_diamond(p['t'])
        new_dots = VGroup(*[self.dot_at(g, p['bhat'], BLUE) for g in (self.axL, self.axR)])
        new_lab = MathTex(r'\hat\beta', font_size=32, color=BLUE)
        new_lab.next_to(new_dots[1], UL, buff=0.06)
        new_corner = MathTex(r'(0,\,t)', font_size=26, color=ORANGE)
        new_corner.next_to(self.plane(self.axR).c2p(0, p['t']), UR, buff=0.06)

        self.play(FadeOut(self.contact), run_time=0.3)
        self.play(Transform(self.disc, new_disc), Transform(self.diamond, new_diamond),
                  Transform(self.bh_dots, new_dots), Transform(self.bh_lab, new_lab),
                  Transform(self.corner_lab, new_corner), run_time=0.38)

    # -- act 5: why the corner is the common case, not a coincidence ------------------------
    def act_why(self):
        p = self.probs[1]
        ax = self.plane(self.axR)
        corner = ax.c2p(p['b_lasso'][0], 0)
        hit = self.plane(self.axL).c2p(*p['b_ridge'])
        origin = self.plane(self.axL).c2p(0, 0)

        # on the disc there is exactly one outward normal at the contact point
        n = (hit - origin) / np.linalg.norm(hit - origin)
        one = Arrow(hit, hit + n * 0.85, buff=0, color=AQUA, stroke_width=4,
                    max_tip_length_to_length_ratio=0.28, z_index=7)

        # at a corner of the diamond the outward normals fill a quarter turn
        wedge = AnnularSector(inner_radius=0.0, outer_radius=0.95, arc_center=corner,
                              start_angle=PI * 0.75, angle=PI * 0.5, color=ORANGE,
                              fill_opacity=0.22, stroke_width=0, z_index=3)
        fan = VGroup(*[Arrow(corner, corner + 0.9 * np.array([np.cos(a), np.sin(a), 0.0]),
                             buff=0, color=ORANGE, stroke_width=3.5,
                             max_tip_length_to_length_ratio=0.3, z_index=7)
                       for a in np.linspace(PI * 0.75, PI * 1.25, 5)])

        self.say(r'one contour orientation fits a smooth point; a \emph{cone} of them fits '
                 r'a corner')
        self.play(GrowArrow(one), run_time=0.3)
        self.play(FadeIn(wedge), *[GrowArrow(a) for a in fan], run_time=0.35)
        self.wait(0.35)
        self.why = VGroup(one, wedge, fan)

    # -- act 6: the same fact in the algebra, and elastic net ------------------------------
    def act_algebra(self):
        p = self.probs[1]
        self.say(r'the corner is the kink in $|\beta_j|$: coordinate descent hits it exactly')

        # clear the left panel and put the soft-thresholding operator there instead
        self.play(FadeOut(self.why), FadeOut(self.contact), FadeOut(self.disc),
                  FadeOut(self.bh_dots), FadeOut(self.bh_lab), FadeOut(self.labL),
                  FadeOut(self.axL), FadeOut(self.objective), run_time=0.3)

        st_ax = Axes(x_range=[-3, 3, 1], y_range=[-2.2, 2.2, 1], x_length=4.25, y_length=3.1,
                     axis_config=dict(color=INK, stroke_width=2.2, include_ticks=True,
                                      tick_size=0.05, include_tip=False))
        st_panel = Rectangle(width=4.25, height=3.1, fill_color=PANEL, fill_opacity=1.0,
                             stroke_width=0)
        dead = Rectangle(width=4.25 * (2.0 / 6.0), height=3.1, fill_color=GOLD,
                         fill_opacity=0.18, stroke_width=0).move_to(st_ax.c2p(0, 0))
        ident = DashedVMobject(Line(st_ax.c2p(-2.2, -2.2), st_ax.c2p(2.2, 2.2),
                                    color=INK, stroke_width=2, stroke_opacity=0.5),
                               num_dashes=22)
        # the notebook's soft_threshold, plotted rather than described
        z = np.linspace(-3, 3, 241)
        s = soft_threshold(z, 1.0)
        curve = VMobject(stroke_color=ORANGE, stroke_width=4)
        curve.set_points_as_corners([st_ax.c2p(a, b) for a, b in zip(z, s)])
        st_formula = MathTex(r'S(z,\gamma)=\mathrm{sign}(z)\max(|z|-\gamma,\,0)', font_size=30)
        dead_lab = Tex(r'exactly $0$ on $[-\gamma,\gamma]$', font_size=26, color=GOLD)

        st = VGroup(st_panel, dead, ident, curve, st_ax).move_to(self.axL[0].get_center())
        st_formula.move_to(st[0].get_top() + UP * 0.45)
        dead_lab.move_to(st[0].get_bottom() + DOWN * 0.35)
        self.play(FadeIn(st_panel), Create(st_ax), run_time=0.3)
        self.play(FadeIn(dead), FadeIn(ident), Create(curve), Write(st_formula),
                  FadeIn(dead_lab), run_time=0.45)
        self.wait(0.35)

        # elastic net: the same corners, edges bowed outward by the L2 part
        self.say(r'on the full fit lasso set $42$ of $100$ coefficients to exactly '
                 r'zero; ridge set $0$')
        enet = self.make_enet(p['t'], alpha=0.5)
        enet_lab = MathTex(r'\alpha\lVert\beta\rVert_1+\tfrac{1}{2}(1-\alpha)'
                           r'\lVert\beta\rVert_2^2 \le t,\quad \alpha=\tfrac{1}{2}',
                           font_size=24, color=PURPLE)
        enet_lab.move_to(self.axR[0].get_bottom() + DOWN * 0.38)
        self.play(Create(enet), FadeIn(enet_lab), run_time=0.4)
        self.wait(0.63)

    def make_enet(self, t, alpha):
        """Level set of alpha||b||_1 + (1-alpha)/2 ||b||_2^2 through the diamond's corners."""
        ax = self.plane(self.axR)
        c = alpha * t + 0.5 * (1 - alpha) * t ** 2      # the value at the vertex (t, 0)
        th = np.linspace(0, 2 * np.pi, 241)
        s = np.abs(np.cos(th)) + np.abs(np.sin(th))     # the L1 norm of the unit direction
        # solve (1-alpha)/2 r^2 + alpha*s*r - c = 0 for the positive root
        r = (-alpha * s + np.sqrt((alpha * s) ** 2 + 2 * (1 - alpha) * c)) / (1 - alpha)
        pts = [ax.c2p(r[i] * np.cos(th[i]), r[i] * np.sin(th[i])) for i in range(len(th))]
        curve = VMobject(stroke_color=PURPLE, stroke_width=3.5, z_index=5)
        curve.set_points_as_corners(pts)
        return curve


# ----------------------------------------------------------------------------------------
# render: manim to mp4, then the repo's two-pass palette encoder to a GIF
# ----------------------------------------------------------------------------------------
def working_ffmpeg(env, build):
    """Return an env whose PATH reaches an ffmpeg that actually starts.

    Homebrew's ffmpeg 7.1_4 is linked against libx265.215, which a later `brew upgrade
    x265` moved aside, so the binary dies on load until ffmpeg is reinstalled. dyld can
    still find the old library through DYLD_FALLBACK_LIBRARY_PATH, but macOS strips
    DYLD_* variables whenever a system shell starts, and the encoder is a shell script.
    So the variable has to be set by something the shell itself runs: a one-line shim
    named ffmpeg, put first on PATH. If ffmpeg already works this does nothing.
    """
    real = None
    for d in env.get('PATH', '').split(os.pathsep):
        cand = os.path.join(d, 'ffmpeg')
        if os.access(cand, os.X_OK):
            real = cand
            break
    if real is None:
        return env
    if subprocess.run([real, '-version'], env=env, capture_output=True).returncode == 0:
        return env

    fallback = glob.glob('/opt/homebrew/Cellar/*/*/lib') + ['/usr/local/lib', '/usr/lib']
    shim_dir = os.path.join(build, 'shim')
    os.makedirs(shim_dir, exist_ok=True)
    shim = os.path.join(shim_dir, 'ffmpeg')
    with open(shim, 'w') as fh:
        fh.write('#!/bin/sh\nDYLD_FALLBACK_LIBRARY_PATH="%s"\nexport '
                 'DYLD_FALLBACK_LIBRARY_PATH\nexec %s "$@"\n'
                 % (os.pathsep.join(fallback), real))
    os.chmod(shim, os.stat(shim).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    env['PATH'] = shim_dir + os.pathsep + env['PATH']
    return env


def render():
    env = os.environ.copy()
    # latex lives in a user tree and dvisvgm comes from homebrew; manim needs both
    env['PATH'] = '/Library/TeX/texbin:/opt/homebrew/bin:' + env.get('PATH', '')

    out_gif = os.path.join(REPO, 'animations', 'gifs', NAME + '.gif')
    with tempfile.TemporaryDirectory() as build:
        subprocess.run([sys.executable, '-m', 'manim', '-qm', '--format=mp4',
                        '--disable_caching', '--media_dir', build, '-o', NAME,
                        os.path.abspath(__file__), 'LassoL1Geometry'],
                       check=True, cwd=REPO, env=env)
        mp4 = os.path.join(build, 'videos', NAME, '720p30', NAME + '.mp4')
        subprocess.run(['bash', os.path.join(REPO, 'animations', 'gif_from_mp4.sh'),
                        mp4, out_gif, '640', '15'],
                       check=True, env=working_ffmpeg(env, build))

    mb = os.path.getsize(out_gif) / 1e6
    print('wrote %s  %.2f MB' % (out_gif, mb))
    if mb > 1.5:
        raise SystemExit('OVER BUDGET: %.2f MB > 1.5 MB cap' % mb)
    try:
        from PIL import Image
        print('frames: %d' % Image.open(out_gif).n_frames)
    except Exception:
        pass


if __name__ == '__main__':
    render()
