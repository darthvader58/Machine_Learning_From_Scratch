"""Scaled dot-product attention, and why the 1/sqrt(d_k) is not cosmetic.

Target notebook: neural_networks/transformer.ipynb

The notebook's encoder is d_model = 32 split over n_heads = 4, so every head runs
attention at d_k = 8 over T = 60 positions. Those are the shapes used here.

Every matrix in this animation is real: Q and K are drawn once from N(0, 1) - the
assumption the variance argument is stated under - and the scores, the softmax and
the context vectors are computed with the notebook's own `softmax`, copied verbatim
below. Nothing is tweened by hand.

The argument, in three beats:

  1. the pipeline as matrices, with the shape at every step
  2. q.k = sum of d_k independent unit-variance products, so Var = d_k and sd = sqrt(d_k);
     the score histogram widens as sqrt(d_k), and dividing by sqrt(d_k) puts it back
     on a unit curve for every d_k
  3. the same scores softmaxed with and without the divisor. Without it the row
     collapses onto one position as d_k grows and the Jacobian trace 1 - sum(a^2)
     goes to zero, which is the gradient going to zero. With it, both are flat in d_k.

Run with no arguments from the repo root to regenerate animations/gifs/attention_scaling.gif.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from manim import *

# ---------------------------------------------------------------- constants

SEED = 0

# the notebook's real hyperparameters (Transformer(d_model=32, n_heads=4), T_LEN=60)
T_LEN = 60
D_MODEL = 32
N_HEADS = 4
D_K = D_MODEL // N_HEADS          # = 8, the width of one head
DK_SWEEP = [8, 32, 128]           # 8 is this notebook's head; 128 is a full-size one
DK_MAX = max(DK_SWEEP)

QUERY_ROW = 0                     # the query position whose attention row we plot

# notebook palette
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
GOLD_ = "#eda100"
PURPLE = "#9a86e8"
FAINT = "#9aa0a6"

# ------------------------------------------------------- the notebook's code


def softmax(z, axis=-1):
    # copied verbatim from transformer.ipynb: shift by the max for stability,
    # then exponentiate and normalise along `axis`
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ------------------------------------------------------------- the real data
# One draw of Q and K with independent standard-normal entries, wide enough for the
# largest d_k in the sweep. Narrower d_k use a prefix of the same columns, so moving
# along the sweep adds dimensions to the same vectors rather than resampling them.
_rng = np.random.RandomState(SEED)
Q_FULL = _rng.normal(size=(T_LEN, DK_MAX))
K_FULL = _rng.normal(size=(T_LEN, DK_MAX))
V_FULL = _rng.normal(size=(T_LEN, DK_MAX))


def scores_at(dk):
    """The T x T matrix of raw dot products q_i . k_j using the first dk dimensions."""
    return Q_FULL[:, :dk] @ K_FULL[:, :dk].T


# the head this notebook actually runs
S_HEAD = scores_at(D_K)
S_HEAD_SCALED = S_HEAD / np.sqrt(D_K)
A_HEAD = softmax(S_HEAD_SCALED, axis=-1)
V_HEAD = V_FULL[:, :D_K]
OUT_HEAD = A_HEAD @ V_HEAD

# score spread as d_k grows, measured rather than asserted
EMP_SD = {dk: scores_at(dk).std() for dk in DK_SWEEP}

# the attention row for one query, with and without the divisor, per d_k
ROW_RAW = {}
ROW_SCALED = {}
for _dk in DK_SWEEP:
    _s = scores_at(_dk)[QUERY_ROW]
    ROW_RAW[_dk] = softmax(_s)
    ROW_SCALED[_dk] = softmax(_s / np.sqrt(_dk))


def jac_trace(a):
    """trace(diag(a) - a a^T) = 1 - sum a_i^2: the total gradient the softmax passes back."""
    return 1.0 - float((a ** 2).sum())


# -------------------------------------------------------------- image helpers


def _hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float64)


_BG = np.array([14.0, 14.0, 16.0])          # near-black, so zero blends into the frame
_NEG = _hex_rgb(BLUE)
_POS = _hex_rgb(ORANGE)
_HOT = _hex_rgb(GOLD_)


def diverging(mat, clip):
    """Signed values -> blue/orange on a dark ground, on a FIXED scale.

    The scale is fixed on purpose: it is what makes the divided-down score matrix
    visibly paler than the raw one instead of being renormalised back to the same
    picture.
    """
    t = np.clip(mat / clip, -1.0, 1.0)
    t = np.round(t * 4) / 4                 # 9 levels, not a continuous ramp
    t = t[..., None]
    pos = _BG + (_POS - _BG) * np.clip(t, 0, 1)
    neg = _BG + (_NEG - _BG) * np.clip(-t, 0, 1)
    return np.uint8(np.clip(pos + neg - _BG, 0, 255))


def sequential(mat, vmax):
    """Non-negative values (attention weights) -> dark to gold."""
    t = np.round(np.clip(mat / vmax, 0.0, 1.0) * 5) / 5
    t = t[..., None]
    return np.uint8(np.clip(_BG + (_HOT - _BG) * t, 0, 255))


def block(rgb, height, label, shape, label_color=WHITE, fs=26):
    """One matrix: the pixels themselves, a thin border, a name and a shape."""
    img = ImageMobject(rgb)
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    img.height = height              # width follows from the pixel aspect ratio
    border = SurroundingRectangle(img, buff=0, color=FAINT, stroke_width=1.2)
    name = MathTex(label, font_size=fs, color=label_color).next_to(border, UP, buff=0.14)
    shp = Text(shape, font_size=19, color=FAINT).next_to(border, DOWN, buff=0.14)
    return Group(img, border, name, shp)


def bars(weights, width, height, color):
    """A row of softmax weights as bar heights on a 0..1 axis."""
    n = len(weights)
    bw = width / n
    grp = VGroup()
    for i, w in enumerate(weights):
        h = max(float(w) * height, 1e-3)
        r = Rectangle(width=bw * 0.78, height=h, stroke_width=0,
                      fill_color=color, fill_opacity=1.0)
        r.move_to(np.array([-width / 2 + (i + 0.5) * bw, h / 2, 0]))
        grp.add(r)
    return grp


def density(samples, lo, hi, nbins=41):
    """Empirical density of real samples as (x, y) pairs - a histogram, not a fitted curve."""
    counts, edges = np.histogram(samples, bins=nbins, range=(lo, hi), density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, counts


def build_equation():
    """The attention equation, with the fraction assembled by hand.

    MathTex compiles each of its arguments separately, so a \\frac cannot be split
    across arguments - the pieces come back brace-balanced and the glyph indices
    silently stop matching. Stacking a numerator, a rule and a denominator keeps
    every fragment valid LaTeX and hands back the sqrt(d_k) as its own mobject,
    which is what the last beat draws a box around.
    """
    lhs = MathTex(r"\mathrm{Attention}(", "Q", ",", "K", ",", "V",
                  r")=\mathrm{softmax}\Bigl(", font_size=38)
    lhs[1].set_color(BLUE)
    lhs[3].set_color(ORANGE)
    lhs[5].set_color(GREEN)

    num = MathTex("Q", "K^{\\top}", font_size=34)
    num[0].set_color(BLUE)
    num[1].set_color(ORANGE)
    den = MathTex(r"\sqrt{d_k}", font_size=34, color=GOLD_)
    rule = Line(LEFT * 0.5, RIGHT * 0.5, stroke_width=2.0, color=WHITE)
    rule.width = max(num.width, den.width) + 0.16
    frac = VGroup(num, rule, den).arrange(DOWN, buff=0.09)

    rhs = MathTex(r"\Bigr)", "V", font_size=38)
    rhs[1].set_color(GREEN)

    eq = VGroup(lhs, frac, rhs).arrange(RIGHT, buff=0.13)
    return eq, den


# ------------------------------------------------------------------ the scene


class AttentionScaling(Scene):

    def construct(self):
        self.camera.background_color = "#0e0e10"

        eq, den = build_equation()
        eq.to_edge(UP, buff=0.28)
        self.den = den

        head = Text("d_model 32  /  4 heads   ->   d_k = 8   over   T = 60 positions",
                    font_size=20, color=FAINT).next_to(eq, DOWN, buff=0.22)

        self.play(Write(eq), run_time=0.65)
        self.play(FadeIn(head, shift=DOWN * 0.15), run_time=0.3)
        self.wait(0.25)

        self.act_pipeline(head)
        self.act_variance()
        self.act_softmax(eq)
        self.act_close(eq)

    # -- 1 ---------------------------------------------------------------------
    def act_pipeline(self, head):
        """Q K^T -> divide -> row softmax -> times V, with the shape at each step."""
        cap_y = -3.0

        q = block(diverging(Q_FULL[:, :D_K], 3.0), 2.1, "Q", "60 x 8", BLUE)
        kt = block(diverging(K_FULL[:, :D_K].T, 3.0), 2.1 * D_K / T_LEN, "K^{\\top}",
                   "8 x 60", ORANGE)
        s = block(diverging(S_HEAD, 8.0), 2.1, "QK^{\\top}", "60 x 60")
        dot = MathTex(r"\cdot", font_size=40)
        equals = MathTex("=", font_size=40)

        row1 = Group(q, dot, kt, equals, s).arrange(RIGHT, buff=0.42).move_to([0, -0.35, 0])

        cap = Text("entry (i, j) is q_i . k_j : how much position i attends to position j",
                   font_size=23, color=FAINT).move_to([0, cap_y, 0])

        self.play(FadeIn(q, shift=RIGHT * 0.2), FadeIn(kt, shift=LEFT * 0.2),
                  FadeIn(dot), run_time=0.45)
        self.play(FadeIn(equals), FadeIn(s, scale=0.85), run_time=0.45)
        self.play(FadeIn(cap), run_time=0.3)
        self.wait(0.7)

        # the same score matrix, now the first stage of a four-stage pipeline
        s2 = block(diverging(S_HEAD, 8.0), 1.45, "QK^{\\top}", "60 x 60")
        s_scaled = block(diverging(S_HEAD_SCALED, 8.0), 1.45, r"QK^{\top}/\sqrt{d_k}",
                         "60 x 60", GOLD_)
        a_img = block(sequential(A_HEAD, 0.12), 1.45, "A", "60 x 60")
        out = block(diverging(OUT_HEAD, 1.2), 1.45, r"AV", "60 x 8", GREEN)

        def arrow(tex, color=WHITE):
            ar = Arrow(LEFT * 0.42, RIGHT * 0.42, buff=0, stroke_width=3,
                       max_tip_length_to_length_ratio=0.28, color=FAINT)
            lb = MathTex(tex, font_size=25, color=color).next_to(ar, UP, buff=0.1)
            return VGroup(ar, lb)

        a1, a2, a3 = arrow(r"\div\sqrt{d_k}", GOLD_), arrow(r"\mathrm{softmax}"), arrow(r"\cdot\,V", GREEN)
        pipe = Group(s2, a1, s_scaled, a2, a_img, a3, out)
        pipe.arrange(RIGHT, buff=0.34).move_to([0, -0.35, 0])

        # a plain cross-fade rather than FadeTransform: FadeTransform goes through
        # Mobject.become, which an ImageMobject has no colour interpolation for
        self.play(FadeOut(q), FadeOut(kt), FadeOut(dot), FadeOut(equals),
                  FadeOut(s), FadeIn(s2), run_time=0.45)

        # stage 2: the divisor. Same colour scale, so the matrix visibly pales.
        cap2 = Text("divide by sqrt(d_k) = 2.83 -- same colour scale, so the scores really do shrink",
                    font_size=23, color=FAINT).move_to([0, cap_y, 0])
        self.play(GrowArrow(a1[0]), FadeIn(a1[1]), FadeIn(s_scaled),
                  Transform(cap, cap2), run_time=0.5)
        self.wait(0.56)

        # stage 3: softmax along each row
        cap3 = Text("softmax along each row: every row is now a distribution summing to 1",
                    font_size=23, color=FAINT).move_to([0, cap_y, 0])
        self.play(GrowArrow(a2[0]), FadeIn(a2[1]), FadeIn(a_img),
                  Transform(cap, cap3), run_time=0.5)
        self.wait(0.56)

        # stage 4: multiply by V
        cap4 = Text("times V: output at i is the weighted average of all 60 value vectors",
                    font_size=23, color=FAINT).move_to([0, cap_y, 0])
        self.play(GrowArrow(a3[0]), FadeIn(a3[1]), FadeIn(out),
                  Transform(cap, cap4), run_time=0.5)
        self.wait(0.7)

        self.play(FadeOut(pipe), FadeOut(cap), FadeOut(head), run_time=0.35)

    # -- 2 ---------------------------------------------------------------------
    def act_variance(self):
        """Var[q.k] = d_k, shown on the real score histograms."""
        title = Text("why the divisor is sqrt(d_k) and not something else",
                     font_size=24, color=GOLD_).move_to([0, 2.55, 0])

        l1 = MathTex(r"q\cdot k=\sum_{i=1}^{d_k} q_i k_i", font_size=34)
        l2 = MathTex(r"\mathrm{Var}[q_i k_i]=\mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2]=1",
                     font_size=30)
        l3 = MathTex(r"\mathrm{Var}[q\cdot k]=\sum_{i=1}^{d_k} 1 = d_k", font_size=32)
        l4 = MathTex(r"\mathrm{sd}[q\cdot k]=\sqrt{d_k}", font_size=34, color=GOLD_)
        algebra = VGroup(l1, l2, l3, l4).arrange(DOWN, buff=0.34, aligned_edge=LEFT)
        algebra.move_to([-4.1, 0.1, 0])

        self.play(FadeIn(title), run_time=0.3)
        self.play(Write(l1), run_time=0.5)
        self.wait(0.35)
        self.play(Write(l2), run_time=0.5)
        self.wait(0.42)
        self.play(Write(l3), run_time=0.5)
        self.wait(0.32)
        self.play(Write(l4), run_time=0.45)
        self.wait(0.42)

        # measured histograms of all 3600 real scores, raw and divided
        colors = {8: BLUE, 32: PURPLE, 128: ORANGE}

        ax_raw = Axes(x_range=[-36, 36, 18], y_range=[0, 0.16, 0.05], x_length=5.4,
                      y_length=1.5, tips=False,
                      axis_config={"stroke_width": 1.6, "color": FAINT,
                                   "include_ticks": True,
                                   "tick_size": 0.06}).move_to([2.6, 1.35, 0])
        ax_sc = Axes(x_range=[-36, 36, 18], y_range=[0, 0.45, 0.2], x_length=5.4,
                     y_length=1.5, tips=False,
                     axis_config={"stroke_width": 1.6, "color": FAINT,
                                  "include_ticks": True,
                                  "tick_size": 0.06}).move_to([2.6, -1.55, 0])

        t_raw = Text("raw scores q.k", font_size=21, color=FAINT).next_to(ax_raw, UP, buff=0.12)
        t_sc = Text("after / sqrt(d_k)", font_size=21, color=GOLD_).next_to(ax_sc, UP, buff=0.12)

        self.play(Create(ax_raw), FadeIn(t_raw), run_time=0.35)

        raw_curves = VGroup()
        labels = VGroup()
        for dk in DK_SWEEP:
            x, y = density(scores_at(dk).ravel(), -36, 36, 41)
            c = ax_raw.plot_line_graph(x, y, add_vertex_dots=False,
                                       line_color=colors[dk], stroke_width=2.6)
            lab = MathTex(r"d_k{=}%d,\ \mathrm{sd}=%.1f" % (dk, EMP_SD[dk]),
                          font_size=21, color=colors[dk])
            raw_curves.add(c)
            labels.add(lab)
        labels.arrange(DOWN, buff=0.1, aligned_edge=LEFT).next_to(ax_raw, DOWN, buff=0.22)
        labels.shift(RIGHT * 0.1)

        for c, lab in zip(raw_curves, labels):
            self.play(Create(c), FadeIn(lab), run_time=0.3)
        self.wait(0.56)

        # dividing collapses all three onto the same unit-variance curve
        self.play(Create(ax_sc), FadeIn(t_sc), run_time=0.35)
        sc_curves = VGroup()
        for dk in DK_SWEEP:
            x, y = density((scores_at(dk) / np.sqrt(dk)).ravel(), -36, 36, 161)
            sc_curves.add(ax_sc.plot_line_graph(x, y, add_vertex_dots=False,
                                                line_color=colors[dk], stroke_width=2.6))
        self.play(*[Create(c) for c in sc_curves], run_time=0.55)
        note = Text("one curve, sd = 1, for every d_k", font_size=21,
                    color=GOLD_).next_to(ax_sc, DOWN, buff=0.18)
        self.play(FadeIn(note), run_time=0.3)
        self.wait(0.77)

        self.play(FadeOut(title), FadeOut(algebra), FadeOut(ax_raw), FadeOut(ax_sc),
                  FadeOut(raw_curves), FadeOut(sc_curves), FadeOut(labels),
                  FadeOut(t_raw), FadeOut(t_sc), FadeOut(note), run_time=0.35)

    # -- 3 ---------------------------------------------------------------------
    def act_softmax(self, eq):
        """The same scores through softmax with and without the divisor, as d_k grows."""
        pw, ph = 5.0, 1.9            # panel width, and the height that weight 1.0 draws
        y0 = -2.05                   # the baseline both panels stand on

        def panel(x0, title_txt, color):
            org = np.array([x0, y0, 0])
            base = Line(org + LEFT * pw / 2, org + RIGHT * pw / 2,
                        stroke_width=1.6, color=FAINT)
            top = DashedLine(org + LEFT * pw / 2 + UP * ph, org + RIGHT * pw / 2 + UP * ph,
                             stroke_width=1.2, color=FAINT, dash_length=0.06)
            one = MathTex("1", font_size=20, color=FAINT).next_to(top, LEFT, buff=0.1)
            ttl = Text(title_txt, font_size=23, color=color).move_to([x0, 1.35, 0])
            return VGroup(base, top, one, ttl), org

        left_frame, left_org = panel(-3.45, "softmax(q.k)   -- no divisor", ORANGE)
        right_frame, right_org = panel(3.45, "softmax(q.k / sqrt(d_k))", GOLD_)

        sub = Text("the same query's scores over all 60 positions, softmaxed both ways",
                   font_size=21, color=FAINT).move_to([0, 1.95, 0])
        dk_lab = MathTex(r"d_k = 8", font_size=32, color=WHITE).move_to([0, -0.9, 0])

        def make(weights, org, color):
            g = bars(weights, pw, ph, color)
            g.shift(org)
            return g

        def stat(a, org, color):
            t = MathTex(r"\max a = %.2f" % a.max(), font_size=24, color=color)
            j = MathTex(r"\mathrm{tr}\,J = %.2f" % jac_trace(a), font_size=24, color=color)
            g = VGroup(t, j).arrange(DOWN, buff=0.12)
            g.move_to(org + np.array([0, -0.72, 0]))
            return g

        dk0 = DK_SWEEP[0]
        lb = make(ROW_RAW[dk0], left_org, ORANGE)
        rb = make(ROW_SCALED[dk0], right_org, GOLD_)
        ls = stat(ROW_RAW[dk0], left_org, ORANGE)
        rs = stat(ROW_SCALED[dk0], right_org, GOLD_)

        self.play(FadeIn(left_frame), FadeIn(right_frame), FadeIn(sub), FadeIn(dk_lab),
                  run_time=0.35)
        self.play(FadeIn(lb, shift=UP * 0.1), FadeIn(rb, shift=UP * 0.1),
                  FadeIn(ls), FadeIn(rs), run_time=0.4)
        self.wait(0.63)

        # widen the head and watch only the left panel move
        for dk in DK_SWEEP[1:]:
            new_lab = MathTex(r"d_k = %d" % dk, font_size=32, color=WHITE).move_to([0, -0.9, 0])
            self.play(
                Transform(lb, make(ROW_RAW[dk], left_org, ORANGE)),
                Transform(rb, make(ROW_SCALED[dk], right_org, GOLD_)),
                Transform(ls, stat(ROW_RAW[dk], left_org, ORANGE)),
                Transform(rs, stat(ROW_SCALED[dk], right_org, GOLD_)),
                Transform(dk_lab, new_lab),
                run_time=0.55,
            )
            self.wait(0.59)

        # what the collapse does to the gradient
        jac = MathTex(r"J=\mathrm{diag}(a)-aa^{\top},\qquad \mathrm{tr}\,J=1-\sum_i a_i^2",
                      font_size=27).move_to([0, 3.05, 0])
        self.play(FadeOut(eq), FadeIn(jac), run_time=0.35)
        msg = Text("one-hot row -> tr J = 0.02 -> no gradient reaches Q or K",
                   font_size=23, color=ORANGE).move_to([0, -3.32, 0])
        self.play(FadeIn(msg), Indicate(ls, color=ORANGE, scale_factor=1.15), run_time=0.5)
        self.wait(0.84)

        self.play(FadeOut(lb), FadeOut(rb), FadeOut(ls), FadeOut(rs), FadeOut(dk_lab),
                  FadeOut(left_frame), FadeOut(right_frame), FadeOut(sub), FadeOut(msg),
                  FadeOut(jac), run_time=0.35)

    # -- 4 ---------------------------------------------------------------------
    def act_close(self, eq):
        eq.move_to([0, 0.85, 0])
        self.play(FadeIn(eq), run_time=0.3)
        box = SurroundingRectangle(self.den, color=GOLD_, buff=0.09, stroke_width=2.5)
        self.play(Create(box), run_time=0.35)
        line1 = Text("the divisor renormalises the logits to unit variance for any d_k",
                     font_size=25, color=WHITE).move_to([0, -0.75, 0])
        line2 = Text("so the softmax keeps a gradient however wide the head gets",
                     font_size=25, color=GOLD_).move_to([0, -1.35, 0])
        self.play(FadeIn(line1), run_time=0.3)
        self.play(FadeIn(line2), run_time=0.3)
        self.wait(0.91)


# ------------------------------------------------------------------ rendering

def main():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_gif = os.path.join(repo, "animations", "gifs", "attention_scaling.gif")
    script = os.path.abspath(__file__)

    env = dict(os.environ)
    # LaTeX lives in a user tree: texbin gives latex, homebrew gives dvisvgm
    env["PATH"] = "/Library/TeX/texbin:/opt/homebrew/bin:" + env.get("PATH", "")

    work = tempfile.mkdtemp(prefix="attention_scaling_")
    subprocess.run(
        [sys.executable, "-m", "manim", "-qm", "--format=mp4",
         "-o", "attention_scaling", script, "AttentionScaling"],
        cwd=work, env=env, check=True,
    )
    mp4 = os.path.join(work, "media", "videos", "attention_scaling", "720p30",
                       "attention_scaling.mp4")
    subprocess.run(
        ["bash", os.path.join(repo, "animations", "gif_from_mp4.sh"), mp4, out_gif,
         "640", "15"],
        env=env, check=True,
    )

    mb = os.path.getsize(out_gif) / 1e6
    print("wrote %s  %.2f MB" % (out_gif, mb))
    if mb > 1.5:
        raise SystemExit("OVER BUDGET: %.2f MB > 1.5 MB cap" % mb)


if __name__ == "__main__":
    main()
