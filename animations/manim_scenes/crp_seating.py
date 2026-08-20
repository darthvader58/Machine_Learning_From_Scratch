"""
crp_seating.py - the Chinese Restaurant Process, for clustering/dirichlet_process_mixture.ipynb

The notebook fits a Dirichlet process mixture to the UCI Wine data by collapsed Gibbs
sampling and is never told how many clusters there are. The reason it does not need to be
told is the CRP: a prior over partitions that keeps a brand-new table on the menu at every
single step, so the number of clusters is a sampled quantity rather than a setting.

This scene seats 14 customers one at a time under the notebook's own `crp_sample` rule,
with the seating probabilities drawn live as bars, then re-runs the same 15 customers at
two values of alpha, then connects the seating rule to the Gibbs update the notebook
implements and to the posterior over K it actually reports.

Render:  /opt/anaconda3/envs/tf_mps/bin/python animations/manim_scenes/crp_seating.py
"""

import glob
import os
import subprocess
import sys

import numpy as np
from manim import *

# ----------------------------------------------------------------------------------------
# paths and the look
# ----------------------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
ANIM_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(ANIM_DIR)

# white ground so the scene sits next to the notebook's ggplot figures, notebook palette
BG = "#ffffff"
INK = "#2b2b2b"
MUTED = "#5a6b7a"
PALE = "#c2c8ce"
CLUSTER_COLOURS = ['#2a78d6', '#eb6834', '#1baf7a',
                   '#8b4fc0', '#c9a227', '#3fb8c4',
                   '#d1467f', '#5a6b7a', '#7fbf3f',
                   '#a0522d', '#4b4b8f', '#e39ab8']
# alpha and the "new table" branch are ALWAYS this colour - in prose, bars and equations
ALPHA_COL = "#d1467f"
# ...so no table is ever allowed to be that colour
PANEL_COLOURS = [c for c in CLUSTER_COLOURS if c != ALPHA_COL]

config.background_color = BG

SEED = 7          # the notebook's seed
ALPHA = 1.0       # the notebook's alpha for the fitted model
N_CUST = 14       # customers seated in the main restaurant


# ----------------------------------------------------------------------------------------
# the CRP itself, taken from the notebook
# ----------------------------------------------------------------------------------------

def crp_sample(n, alpha, rng):
    """Verbatim from clustering/dirichlet_process_mixture.ipynb - seats n customers."""
    counts = []
    growth = np.empty(n, dtype=int)
    for i in range(n):
        w = np.array(counts + [alpha], dtype=float)
        k = rng.choice(len(w), p=w / w.sum())
        if k == len(counts):
            counts.append(0)
        counts[k] += 1
        growth[i] = len(counts)
    return np.array(counts), growth


def crp_trace(n, alpha, seed):
    """
    The same rule as crp_sample, drawing from the same Generator in the same order, but
    recording which table each customer picked. That record is what the animation replays,
    so the dynamics on screen are sampled rather than scripted.
    """
    rng = np.random.default_rng(seed)
    counts, picks = [], []
    for i in range(n):
        w = np.array(counts + [alpha], dtype=float)
        k = int(rng.choice(len(w), p=w / w.sum()))
        picks.append(k)
        if k == len(counts):
            counts.append(0)
        counts[k] += 1
    return counts, picks


# the two must agree exactly, which is the check that crp_trace really is crp_sample
assert list(crp_trace(N_CUST, ALPHA, SEED)[0]) == \
       list(crp_sample(N_CUST, ALPHA, np.random.default_rng(SEED))[0]), \
       'crp_trace has drifted from the notebook rule'

SEATING_COUNTS, SEATING_PICKS = crp_trace(N_CUST, ALPHA, SEED)          # -> [8, 2, 1, 3]
SMALL_A, BIG_A, N_COMPARE = 0.5, 10.0, 15                               # notebook demo alphas
SMALL_COUNTS, SMALL_PICKS = crp_trace(N_COMPARE, SMALL_A, SEED)         # -> 2 tables
BIG_COUNTS, BIG_PICKS = crp_trace(N_COMPARE, BIG_A, SEED)               # -> 7 tables

# numbers the notebook's fitted model reports (UCI Wine, N=178, 4 PCs, alpha=1, 500 sweeps)
K_VALUES = [5, 6, 7, 8, 9, 10]
K_POST = [0.040, 0.157, 0.426, 0.271, 0.094, 0.011]
K_MODE = 7


# ----------------------------------------------------------------------------------------
# small mobject helpers
# ----------------------------------------------------------------------------------------

def txt(s, size=28, colour=INK, weight=NORMAL):
    return Text(s, font_size=size, color=colour, weight=weight)


def mtex(*s, size=34, colour=INK):
    return MathTex(*s, font_size=size, color=colour)


def fit_width(mob, w):
    """Manim will happily typeset past the frame edge; clamp anything that does."""
    if mob.width > w:
        mob.scale_to_fit_width(w)
    return mob


# geometry of the main restaurant
COL_X = [-5.30, -2.90, -0.50, 1.90]     # the four tables this particular seating opens
NEW_X = 4.90                            # the "new table" slot, always on screen
TABLE_Y = 0.55
TABLE_R = 0.42
RING_R = 0.74
BAR_BASE = -1.50
BAR_W = 0.46
BAR_MAX = 1.05
PROB_Y = -1.82
CAP_Y = -2.45
ARRIVE = np.array([-6.30, 2.20, 0.0])


def seat_pos(centre, idx, slots=8):
    """Chairs evenly around a table, first chair at the top."""
    ang = PI / 2 - idx * TAU / slots
    return centre + RING_R * np.array([np.cos(ang), np.sin(ang), 0.0])


def make_bar(x, h, colour):
    h = max(h, 0.012)
    r = Rectangle(width=BAR_W, height=h, stroke_width=0,
                  fill_color=colour, fill_opacity=0.88)
    r.move_to(np.array([x, BAR_BASE + 0.5 * h, 0.0]))
    return r


def grid_slots(cx, k, cols=4, dx=1.33, rows=(0.70, -0.90)):
    """Table positions for a comparison panel: one centred row, or two if there are many."""
    if k <= cols:
        return [np.array([cx + (i - 0.5 * (k - 1)) * dx, 0.0, 0.0]) for i in range(k)]
    top, bot = cols, k - cols
    out = [np.array([cx + (i - 0.5 * (top - 1)) * dx, rows[0], 0.0]) for i in range(top)]
    out += [np.array([cx + (i - 0.5 * (bot - 1)) * dx, rows[1], 0.0]) for i in range(bot)]
    return out


class CRPSeating(Scene):

    # ------------------------------------------------------------------------------------
    def construct(self):
        self.camera.background_color = BG
        self.beat_title()
        self.beat_seating()
        self.beat_rich_get_richer()
        self.beat_open_door()
        self.beat_alpha()
        self.beat_gibbs()
        self.beat_posterior()

    # ------------------------------------------------------------------------------------
    def beat_title(self):
        title = txt('The Chinese Restaurant Process', 46, INK, BOLD)
        sub = txt('a prior over partitions that never fixes the number of clusters',
                  28, MUTED)
        title.move_to([0, 0.55, 0])
        sub.next_to(title, DOWN, buff=0.38)
        self.play(FadeIn(title, shift=0.3 * UP), run_time=0.4)
        self.play(FadeIn(sub), run_time=0.3)
        self.wait(0.25)
        self.play(FadeOut(title), FadeOut(sub), run_time=0.3)

    # ------------------------------------------------------------------------------------
    def beat_seating(self):
        """Seat 14 customers by the CRP rule, with the seating probabilities drawn live."""

        # the governing rule, kept on screen the whole way through
        eq = mtex(r"P(z_n = k \mid z_{1:n-1}) = \frac{n_k}{n-1+\alpha}",
                  r"\qquad",
                  r"P(\text{new table}) = \frac{\alpha}{n-1+\alpha}", size=34)
        eq[2].set_color(ALPHA_COL)
        fit_width(eq, 12.6).move_to([0, 3.05, 0])

        head_occ = txt('occupied tables', 24, MUTED).move_to([-1.70, 1.66, 0])
        head_new = txt('new table', 24, ALPHA_COL).move_to([NEW_X, 1.66, 0])
        divider = DashedLine([3.42, -2.10, 0], [3.42, 1.75, 0], color=PALE,
                             stroke_width=2, dash_length=0.12)

        # the new-table slot: dashed, empty, and permanently on the menu
        new_circle = DashedVMobject(Circle(radius=TABLE_R, color=ALPHA_COL, stroke_width=4),
                                    num_dashes=24)
        new_circle.move_to([NEW_X, TABLE_Y, 0])
        new_inner = mtex(r"\alpha", size=34, colour=ALPHA_COL).move_to([NEW_X, TABLE_Y, 0])
        new_bar = make_bar(NEW_X, 0.012, ALPHA_COL)
        new_prob = txt('0.00', 24, ALPHA_COL).move_to([NEW_X, PROB_Y, 0])
        base = Line([-6.10, BAR_BASE, 0], [6.10, BAR_BASE, 0], color=PALE, stroke_width=2)

        self.play(Write(eq), run_time=0.5)
        self.play(FadeIn(head_occ), FadeIn(head_new), Create(divider), Create(base),
                  Create(new_circle), FadeIn(new_inner), FadeIn(new_bar), FadeIn(new_prob),
                  run_time=0.35)

        counter = txt('customer 1 walks in', 28, INK).move_to([-3.90, 2.20, 0])
        caption = txt('the first customer has nowhere to sit but a new table',
                      27, MUTED).move_to([0, CAP_Y, 0])
        self.play(FadeIn(counter), FadeIn(caption), run_time=0.3)

        # captions keyed to the customer about to arrive, describing the state on screen
        captions_at = {
            2: 'one seated: 1/2 to join him, 1/2 to open a table',
            4: 'three singletons, and a new table just as likely as any of them',
            6: 'the door is still open - table 4 opens next, at probability 0.17',
            10: 'table 3 has sat at one customer since the start. the tail is real',
            12: 'and every arrival makes the big table more attractive still',
        }

        tables = []            # one dict per occupied table
        for i, pick in enumerate(SEATING_PICKS):
            counts_before = [len(t['seats']) for t in tables]
            n_prev = sum(counts_before)
            denom = n_prev + ALPHA
            probs = [c / denom for c in counts_before] + [ALPHA / denom]

            # --- the customer arrives, and every probability is recomputed in front of them
            dot = Dot(ARRIVE, radius=0.115, color=MUTED)
            anims = [FadeIn(dot, scale=0.4),
                     Transform(counter, txt('customer %d walks in' % (i + 1), 28, INK)
                               .move_to([-3.90, 2.20, 0]))]
            for t, p in zip(tables, probs[:-1]):
                anims.append(Transform(t['bar'], make_bar(t['x'], BAR_MAX * p, t['colour'])))
                anims.append(Transform(t['prob'], txt('%.2f' % p, 24, t['colour'])
                                       .move_to([t['x'], PROB_Y, 0])))
            anims.append(Transform(new_bar, make_bar(NEW_X, BAR_MAX * probs[-1], ALPHA_COL)))
            anims.append(Transform(new_prob, txt('%.2f' % probs[-1], 24, ALPHA_COL)
                                   .move_to([NEW_X, PROB_Y, 0])))
            if i + 1 in captions_at:
                anims.append(Transform(caption, txt(captions_at[i + 1], 27, MUTED)
                                       .move_to([0, CAP_Y, 0])))
            self.play(*anims, run_time=0.33 if i < 3 else 0.34)

            # --- the draw lands on an occupied table, or on the alpha branch
            if pick == len(tables):
                colour = CLUSTER_COLOURS[pick]
                x = COL_X[pick]
                circ = Circle(radius=TABLE_R, color=colour, stroke_width=4).move_to([x, TABLE_Y, 0])
                cnt = txt('0', 30, colour).move_to([x, TABLE_Y, 0])
                bar = make_bar(x, 0.012, colour)
                prob = txt('0.00', 24, colour).move_to([x, PROB_Y, 0])
                tables.append(dict(circle=circ, count=cnt, bar=bar, prob=prob,
                                   x=x, colour=colour, seats=[]))
                self.play(Indicate(new_bar, scale_factor=1.18, color=ALPHA_COL),
                          Create(circ), FadeIn(cnt), FadeIn(bar), FadeIn(prob),
                          run_time=0.3)
            else:
                self.play(Indicate(tables[pick]['bar'], scale_factor=1.18,
                                   color=tables[pick]['colour']),
                          run_time=0.3 if i < 4 else 0.24)

            # the customer takes a chair, and that table's count goes up
            t = tables[pick]
            target = seat_pos(np.array([t['x'], TABLE_Y, 0.0]), len(t['seats']))
            t['seats'].append(dot)
            self.play(dot.animate.move_to(target).set_color(t['colour']),
                      Transform(t['count'], txt(str(len(t['seats'])), 30, t['colour'])
                                .move_to([t['x'], TABLE_Y, 0])),
                      run_time=0.3 if i < 3 else 0.26)

        self.wait(0.25)
        self.frame_kit = dict(eq=eq, head_occ=head_occ, head_new=head_new, divider=divider,
                              base=base, new_circle=new_circle, new_inner=new_inner,
                              new_bar=new_bar, new_prob=new_prob, counter=counter,
                              caption=caption)
        self.tables = tables

    # ------------------------------------------------------------------------------------
    def beat_rich_get_richer(self):
        """Name the first property: the pull is linear in occupancy."""
        tables, kit = self.tables, self.frame_kit
        sizes = [len(t['seats']) for t in tables]

        line = txt('table sizes  %s  -  a few large ones and a tail of small ones'
                   % ', '.join(str(s) for s in sizes), 28, INK).move_to([0, CAP_Y, 0])
        strap = VGroup(txt('RICH GET RICHER', 32, CLUSTER_COLOURS[0], BOLD),
                       txt('-  the pull of a table is linear in', 26, MUTED),
                       mtex(r"n_k", size=30, colour=CLUSTER_COLOURS[0])
                       ).arrange(RIGHT, buff=0.24).move_to([0, -3.15, 0])

        self.play(FadeOut(kit['counter']), Transform(kit['caption'], line),
                  FadeOut(kit['new_bar']), FadeOut(kit['new_prob']), FadeOut(kit['base']),
                  *[FadeOut(t['bar']) for t in tables],
                  *[FadeOut(t['prob']) for t in tables], run_time=0.3)
        self.play(FadeIn(strap),
                  Indicate(tables[0]['circle'], scale_factor=1.14, color=CLUSTER_COLOURS[0]),
                  run_time=0.35)
        self.wait(0.39)

        leaving = VGroup(strap, kit['eq'], kit['head_occ'], kit['head_new'],
                         kit['divider'], kit['new_circle'], kit['new_inner'], kit['caption'],
                         *[m for t in tables for m in [t['circle'], t['count']] + t['seats']])
        self.play(FadeOut(leaving), run_time=0.3)

    # ------------------------------------------------------------------------------------
    def beat_open_door(self):
        """Name the second property: the alpha branch shrinks but never closes."""
        title = txt('the door is never closed', 38, ALPHA_COL, BOLD).move_to([0, 3.25, 0])
        formula = mtex(r"P(\text{new table}) = \frac{\alpha}{n-1+\alpha}",
                       size=34, colour=ALPHA_COL).move_to([-3.20, 2.40, 0])

        # a plain hand-built pair of axes: customers seated on x, P(new table) on y
        x0, x1, y0, y1 = -6.20, -0.30, -2.30, 1.80
        ns = np.arange(1, 179)
        ps = ALPHA / (ns - 1.0 + ALPHA)          # 1, 1/2, 1/3, ... for alpha = 1

        def to_pt(n, p):
            return np.array([x0 + (x1 - x0) * (n - 1) / 177.0,
                             y0 + (y1 - y0) * p, 0.0])

        xaxis = Line([x0, y0, 0], [x1 + 0.25, y0, 0], color=MUTED, stroke_width=2.5)
        yaxis = Line([x0, y0, 0], [x0, y1 + 0.22, 0], color=MUTED, stroke_width=2.5)
        curve = VMobject(color=ALPHA_COL, stroke_width=5)
        curve.set_points_as_corners([to_pt(n, p) for n, p in zip(ns, ps)])
        xlab = txt('customers seated  n', 22, MUTED).move_to([-3.20, y0 - 0.42, 0])
        ticks = VGroup(txt('1', 22, MUTED).move_to([x0 - 0.30, y1, 0]),
                       txt('0.5', 22, MUTED).move_to([x0 - 0.42, 0.5 * (y0 + y1), 0]),
                       txt('0', 22, MUTED).move_to([x0 - 0.30, y0, 0]))

        marks, labs = VGroup(), VGroup()
        for n, note in [(2, 'n=2:  0.50'), (15, 'n=15:  0.07'), (178, 'n=178:  0.006')]:
            p = ALPHA / (n - 1.0 + ALPHA)
            d = Dot(to_pt(n, p), radius=0.075, color=ALPHA_COL)
            l = txt(note, 21, ALPHA_COL).next_to(d, UR, buff=0.14)
            if n == 178:
                l.next_to(d, UP, buff=0.20).shift(LEFT * 0.80)
            marks.add(d)
            labs.add(l)

        never = txt('it shrinks - but it never reaches zero', 26, INK).move_to([-3.20, -3.30, 0])

        grow = mtex(r"E[K] \approx \alpha \log\!\left(1 + \tfrac{N}{\alpha}\right)",
                    size=32, colour=CLUSTER_COLOURS[0])
        fit_width(grow, 5.0).move_to([3.70, 1.45, 0])
        grow_note = txt('so the number of clusters is\n'
                        'unbounded a priori, yet it\n'
                        'grows only like log N:\n'
                        'doubling the data adds a\n'
                        'constant, not a factor',
                        25, MUTED).move_to([3.70, -0.90, 0])

        self.play(FadeIn(title), Write(formula), run_time=0.45)
        self.play(Create(xaxis), Create(yaxis), FadeIn(xlab), FadeIn(ticks), run_time=0.3)
        self.play(Create(curve), run_time=0.45)
        self.play(FadeIn(marks), FadeIn(labs), FadeIn(never), run_time=0.3)
        self.wait(0.25)
        self.play(Write(grow), run_time=0.4)
        self.play(FadeIn(grow_note), run_time=0.3)
        self.wait(0.45)
        self.play(FadeOut(VGroup(title, formula, xaxis, yaxis, curve, xlab, ticks,
                                 marks, labs, never, grow, grow_note)), run_time=0.3)

    # ------------------------------------------------------------------------------------
    def beat_alpha(self):
        """Same 15 customers, same rule, two values of alpha."""
        head = VGroup(txt('same process, same 15 customers - only ', 30, INK),
                      mtex(r"\alpha", size=36, colour=ALPHA_COL),
                      txt(' changes', 30, INK)).arrange(RIGHT, buff=0.12).move_to([0, 3.15, 0])
        rule = DashedLine([0, -2.60, 0], [0, 2.45, 0], color=PALE, stroke_width=2)
        self.play(FadeIn(head), Create(rule), run_time=0.3)

        panels = []
        for cx, a, picks, counts, off in [(-3.40, SMALL_A, SMALL_PICKS, SMALL_COUNTS, 0),
                                          (3.40, BIG_A, BIG_PICKS, BIG_COUNTS, 3)]:
            panels.append(dict(picks=picks, off=off, tables=[],
                               lab=mtex(r"\alpha = %.1f" % a, size=40,
                                        colour=ALPHA_COL).move_to([cx, 2.15, 0]),
                               slots=grid_slots(cx, len(counts))))
        self.play(*[FadeIn(p['lab']) for p in panels], run_time=0.3)

        # seat both restaurants in lockstep, so the divergence is the thing you watch
        for i in range(N_COMPARE):
            anims = []
            for p in panels:
                pick = p['picks'][i]
                if pick == len(p['tables']):
                    colour = PANEL_COLOURS[(p['off'] + pick) % len(PANEL_COLOURS)]
                    c = Circle(radius=0.225, color=colour, stroke_width=4,
                               fill_color=colour, fill_opacity=0.14).move_to(p['slots'][pick])
                    n = txt('1', 26, colour).move_to(p['slots'][pick])
                    p['tables'].append(dict(circ=c, num=n, colour=colour, count=1))
                    anims += [GrowFromCenter(c), FadeIn(n)]
                else:
                    t = p['tables'][pick]
                    t['count'] += 1
                    anims.append(t['circ'].animate.scale_to_fit_width(
                        2 * (0.18 + 0.045 * t['count'])))
                    anims.append(Transform(t['num'], txt(str(t['count']), 26, t['colour'])
                                           .move_to(t['circ'].get_center())))
            self.play(*anims, run_time=0.3)

        r1 = txt('%d tables' % len(SMALL_COUNTS), 32, CLUSTER_COLOURS[0],
                 BOLD).move_to([-3.40, -2.30, 0])
        r2 = txt('%d tables' % len(BIG_COUNTS), 32, PANEL_COLOURS[3],
                 BOLD).move_to([3.40, -2.30, 0])
        note = txt('alpha is a concentration, not a cluster count: it sets how eagerly '
                   'a new table opens', 26, MUTED).move_to([0, -3.20, 0])
        fit_width(note, 12.8)
        self.play(FadeIn(r1), FadeIn(r2), FadeIn(note), run_time=0.3)
        self.wait(0.45)

        self.play(FadeOut(VGroup(head, rule, r1, r2, note,
                                 *[p['lab'] for p in panels],
                                 *[m for p in panels for t in p['tables']
                                   for m in (t['circ'], t['num'])])), run_time=0.3)

    # ------------------------------------------------------------------------------------
    def beat_gibbs(self):
        """Seating is assignment: this rule, times a likelihood, is the notebook's sampler."""
        head = txt("each table carries that cluster's parameters, so seating IS assignment",
                   30, INK).move_to([0, 3.30, 0])
        fit_width(head, 12.8)

        e1 = mtex(r"P(z_i = c \mid \mathbf{z}_{-i}, \mathbf{X}) \;\propto\;",
                  r"n_{-i,c}", r"\;\cdot\;",
                  r"t_{\nu_n-D+1}\!\left(\mathbf{x}_i \mid \boldsymbol\mu_n, \Sigma_n\right)",
                  size=34)
        e2 = mtex(r"P(z_i = \text{new} \mid \mathbf{z}_{-i}, \mathbf{X}) \;\propto\;",
                  r"\alpha", r"\;\cdot\;",
                  r"t_{\nu_0-D+1}\!\left(\mathbf{x}_i \mid \boldsymbol\mu_0, \Sigma_0\right)",
                  size=34)
        e1[1].set_color(CLUSTER_COLOURS[0])
        e2[1].set_color(ALPHA_COL)
        e1[3].set_color(CLUSTER_COLOURS[2])
        e2[3].set_color(CLUSTER_COLOURS[2])
        eqs = VGroup(e1, e2).arrange(DOWN, buff=0.34, aligned_edge=LEFT).move_to([0, 2.20, 0])
        fit_width(eqs, 12.2)

        leg1 = VGroup(mtex(r"n_{-i,c}", size=30, colour=CLUSTER_COLOURS[0]),
                      txt('and', 24, MUTED),
                      mtex(r"\alpha", size=30, colour=ALPHA_COL),
                      txt(':  the CRP seating rule you just watched', 25, MUTED),
                      ).arrange(RIGHT, buff=0.16)
        leg2 = VGroup(mtex(r"t_{\nu}(\cdot)", size=30, colour=CLUSTER_COLOURS[2]),
                      txt(':  the posterior predictive of that cluster', 25, MUTED),
                      ).arrange(RIGHT, buff=0.16)
        legend = VGroup(leg1, leg2).arrange(DOWN, buff=0.24, aligned_edge=LEFT)
        legend.move_to([0, 1.05, 0])

        self.play(FadeIn(head), run_time=0.3)
        self.play(Write(e1), run_time=0.5)
        self.play(Write(e2), run_time=0.45)
        self.play(FadeIn(legend), run_time=0.3)
        self.wait(0.32)

        # a miniature restaurant, to show a cluster dying and a cluster being born
        mini_y = -1.30
        minis = []
        for x, k, ci in [(-4.60, 4, 0), (-1.55, 3, 1), (1.50, 1, 2)]:
            colour = CLUSTER_COLOURS[ci]
            minis.append(dict(
                circ=Circle(radius=TABLE_R, color=colour, stroke_width=4).move_to([x, mini_y, 0]),
                dots=VGroup(*[Dot(seat_pos(np.array([x, mini_y, 0.0]), j), radius=0.10,
                                  color=colour) for j in range(k)]),
                colour=colour, x=x))
        cap = txt('every sweep, the sampler removes one customer and re-seats them',
                  27, MUTED).move_to([0, -3.25, 0])
        self.play(*[Create(m['circ']) for m in minis], *[FadeIn(m['dots']) for m in minis],
                  FadeIn(cap), run_time=0.35)

        # remove the singleton: its cluster empties, so it is deleted
        lone = minis[2]['dots'][0]
        self.play(lone.animate.move_to([1.50, mini_y + 1.20, 0]).set_color(MUTED), run_time=0.3)
        gone = txt('cluster deleted', 23, MUTED).move_to([1.50, mini_y - 0.80, 0])
        self.play(FadeOut(minis[2]['circ'], scale=0.4), FadeIn(gone), run_time=0.3)
        # it re-seats where the n_{-i,c} term is largest
        self.play(lone.animate.move_to(seat_pos(np.array([-4.60, mini_y, 0.0]), 4))
                  .set_color(CLUSTER_COLOURS[0]),
                  Indicate(minis[0]['circ'], scale_factor=1.14, color=CLUSTER_COLOURS[0]),
                  FadeOut(gone), run_time=0.3)

        # and on another sweep the alpha branch wins, so a brand-new cluster appears
        born = minis[1]['dots'][2]
        newc = Circle(radius=TABLE_R, color=CLUSTER_COLOURS[4],
                      stroke_width=4).move_to([4.55, mini_y, 0])
        cap2 = txt('clusters vanish when they empty, and appear when the likelihood pays',
                   27, MUTED).move_to([0, -3.25, 0])
        fit_width(cap2, 12.8)
        self.play(born.animate.move_to(seat_pos(np.array([4.55, mini_y, 0.0]), 0))
                  .set_color(CLUSTER_COLOURS[4]),
                  GrowFromCenter(newc), Transform(cap, cap2), run_time=0.38)
        self.wait(0.45)

        self.play(FadeOut(VGroup(head, e1, e2, legend, cap, newc,
                                 minis[0]['circ'], minis[1]['circ'],
                                 *[m['dots'] for m in minis])), run_time=0.3)

    # ------------------------------------------------------------------------------------
    def beat_posterior(self):
        """What the notebook actually reports: a posterior over K, not a chosen K."""
        head = txt('K is never supplied. It is read off the posterior.', 36, INK, BOLD)
        head.move_to([0, 3.15, 0])

        base_y, cx, w, gap, scale = -1.85, -3.20, 0.70, 1.08, 5.0
        xs = [cx + (i - 2.5) * gap for i in range(len(K_VALUES))]
        bars, klabs = VGroup(), VGroup()
        for x, k, p in zip(xs, K_VALUES, K_POST):
            colour = CLUSTER_COLOURS[0] if k == K_MODE else PALE
            h = max(scale * p, 0.02)
            bars.add(Rectangle(width=w, height=h, stroke_width=0, fill_color=colour,
                               fill_opacity=0.92).move_to([x, base_y + 0.5 * h, 0]))
            klabs.add(txt(str(k), 26, MUTED).move_to([x, base_y - 0.32, 0]))
        axis = Line([xs[0] - 0.55, base_y, 0], [xs[-1] + 0.55, base_y, 0],
                    color=MUTED, stroke_width=2.5)
        xtitle = txt('occupied clusters K', 24, MUTED).move_to([cx, base_y - 0.88, 0])
        modelab = txt('0.426', 25, CLUSTER_COLOURS[0], BOLD)
        modelab.move_to([xs[2], base_y + scale * K_POST[2] + 0.30, 0])

        facts = VGroup(
            txt('UCI Wine  -  N = 178, 4 PCs', 27, INK),
            mtex(r"\alpha = 1.0,\;\; 500\ \text{sweeps},\;\; 150\ \text{burn-in}",
                 size=28, colour=MUTED),
            txt('posterior mode      K = 7', 28, CLUSTER_COLOURS[0], BOLD),
            txt('posterior mean       7.26', 27, INK),
            txt('95% credible set   [5, 9]', 27, INK),
            txt('ARI vs cultivars    0.619', 27, CLUSTER_COLOURS[2]),
        ).arrange(DOWN, buff=0.30, aligned_edge=LEFT).move_to([3.60, 0.30, 0])

        self.play(FadeIn(head), run_time=0.3)
        self.play(Create(axis), FadeIn(klabs), FadeIn(xtitle), run_time=0.3)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.14),
                  run_time=0.55)
        self.play(FadeIn(modelab), run_time=0.3)
        self.play(LaggedStart(*[FadeIn(f, shift=0.15 * RIGHT) for f in facts],
                              lag_ratio=0.18), run_time=0.7)
        self.wait(0.28)

        closing = txt('the cluster count is a property of the sample, not a setting of the model',
                      28, MUTED).move_to([0, -3.42, 0])
        fit_width(closing, 12.8)
        self.play(FadeIn(closing), run_time=0.3)
        self.wait(0.63)          # a long hold so the loop does not snap


# ----------------------------------------------------------------------------------------
# render itself: manim -> mp4 -> two-pass palette gif
# ----------------------------------------------------------------------------------------

if __name__ == '__main__':
    name = 'crp_seating'
    media = os.path.join(ANIM_DIR, '_manim_media')
    out_gif = os.path.join(ANIM_DIR, 'gifs', '%s.gif' % name)
    os.makedirs(os.path.join(ANIM_DIR, 'gifs'), exist_ok=True)

    env = dict(os.environ)
    # latex lives in a user tree and dvisvgm comes from homebrew; manim needs both on PATH
    env['PATH'] = '/Library/TeX/texbin:/opt/homebrew/bin:' + env.get('PATH', '')
    # the homebrew ffmpeg on this machine is linked against an x265 that brew has since
    # bumped, so it refuses to start until the older Cellar lib is back on the search path
    x265 = [d for d in glob.glob('/opt/homebrew/Cellar/x265/*/lib') if os.path.isdir(d)]
    if x265:
        env['DYLD_FALLBACK_LIBRARY_PATH'] = ':'.join(
            x265 + [env.get('DYLD_FALLBACK_LIBRARY_PATH', '')]).strip(':')

    subprocess.run([sys.executable, '-m', 'manim', '-qm', '--format=mp4',
                    '--media_dir', media, '-o', name,
                    os.path.abspath(__file__), 'CRPSeating'],
                   check=True, env=env, cwd=ANIM_DIR)

    mp4 = os.path.join(media, 'videos', name, '720p30', '%s.mp4' % name)
    subprocess.run(['bash', os.path.join(ANIM_DIR, 'gif_from_mp4.sh'),
                    mp4, out_gif, '640', '15'], check=True, env=env)

    mb = os.path.getsize(out_gif) / 1e6
    print('gif: %s  (%.2f MB)' % (out_gif, mb))
    if mb > 1.5:
        raise SystemExit('GIF is %.2f MB, over the 1.5 MB cap' % mb)
