"""Why maximising the margin is the same as minimising ||w||.

Companion animation for classification/support_vector_machine.ipynb.

The notebook's own `Support_Vector_Machine.fit` is copied in below and run for
real, so every number on screen is the number the notebook reports:
w = [1.1169, 1.1169], b = -3.57, ||w|| = 1.58, margin = 2/||w|| = 1.27, and
exactly two support vectors out of 100 flowers.

The animation walks the derivation in five beats:
  1. several different lines all separate the two species, so "correct" does
     not single one out;
  2. add the two margin lines w.x + b = +-1 and derive margin = 2/||w||,
     cancelling the b's and dividing by ||w|| a term at a time;
  3. drive ||w|| up and down and watch the margin lines open and close;
  4. drag a flower across a margin line and watch y(w.x + b) fall below 1;
  5. the two support vectors are all that hold the line in place.

Run it with no arguments from the repo root to rebuild the GIF:
    /opt/anaconda3/envs/tf_mps/bin/python animations/manim_scenes/svm_margin.py
"""
import numpy as np
from manim import *
from ucimlrepo import fetch_ucirepo

# ---------------------------------------------------------------- look ------
# ggplot-ish light background so this reads as part of the same set as the
# eleven matplotlib GIFs. Colours are the notebook's own class colours.
config.background_color = "#f5f5f5"

INK = "#22252a"        # all text and the decision boundary
SOFT = "#8a8f98"       # axes, ticks, secondary text
C_NEG = "#2a78d6"      # setosa, y = -1
C_POS = "#eb6834"      # versicolor, y = +1
C_MARGIN = "#1baf7a"   # the two margin lines
C_W = "#4a3aa7"        # w, ||w||, and anything measured along w
C_BAD = "#cc2b1d"      # a violated constraint

Text.set_default(color=INK)
Tex.set_default(color=INK)
MathTex.set_default(color=INK)


# ------------------------------------------------------- data and fit -------
def load_iris_two_class():
    """Exactly the notebook's data prep: setosa vs versicolor, petal columns."""
    iris = fetch_ucirepo(id=53)
    X = iris.data.features.values
    y = iris.data.targets.values.ravel()
    mask = (y == 'Iris-setosa') | (y == 'Iris-versicolor')
    X = X[mask][:, [2, 3]]
    y = np.where(y[mask] == 'Iris-setosa', -1, 1)
    return X, y


class Support_Vector_Machine:
    """Copied from the notebook. `visualize` is dropped (matplotlib only);
    `fit` and `predict` are verbatim, so the w and b below are the real ones."""

    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: '#eb6834', -1: '#2a78d6'}

    def fit(self, data):
        self.data = data
        opt_dict = {}

        # try w pointing in all four diagonal directions
        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # coarse pass first, then two finer ones
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001]

        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10

        self.steps = []

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False

            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True

                        # every point has to satisfy y * (w . x + b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    break
                            if not found_option:
                                break

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                else:
                    w = w - step

            # the smallest |w| that still fits every point is the widest margin
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]
            self.steps.append([step, np.linalg.norm(self.w)])

            latest_optimum = opt_choice[0][0] + step * 2

    def predict(self, features):
        # which side of the line the point falls on
        return np.sign(np.dot(np.array(features), self.w) + self.b)


X, Y = load_iris_two_class()
_svm = Support_Vector_Machine(visualization=False)
_svm.fit({-1: X[Y == -1], 1: X[Y == 1]})

W = _svm.w                                  # [1.1169, 1.1169]
B = _svm.b                                  # -3.57
WNORM = float(np.linalg.norm(W))            # 1.5795
NHAT = W / WNORM                            # unit normal to the boundary
COFF = -B / WNORM                           # boundary is nhat . x = COFF

# functional margin of every flower; the smallest ones are the support vectors
FMARGIN = Y * (X @ W + B)
SUPPORT = FMARGIN <= FMARGIN.min() + 0.05   # 2 of 100, as the notebook reports
XPLUS = X[SUPPORT & (Y == 1)][0]            # (3.0, 1.1)  on w.x + b = +1
XMINUS = X[SUPPORT & (Y == -1)][0]          # (1.9, 0.4)  on w.x + b = -1
DRAG_I = int(np.abs(X - np.array([4.5, 1.5])).sum(1).argmin())   # a far flower

# plot window, in petal-length / petal-width units
XLO, XHI = 0.5, 5.75
YLO, YHI = -0.25, 2.25


def clip_line(n, c):
    """Clip the infinite line n . x = c to the plot window.

    Walk a long way along the line in both directions, keep the samples that
    land inside the box, and return the first and last of them. Crude, but it
    handles vertical lines without a special case."""
    d = np.array([-n[1], n[0]])
    p0 = c * np.array(n)
    t = np.linspace(-14, 14, 6001)
    pts = p0[None, :] + t[:, None] * d[None, :]
    inside = ((pts[:, 0] >= XLO) & (pts[:, 0] <= XHI) &
              (pts[:, 1] >= YLO) & (pts[:, 1] <= YHI))
    if not inside.any():
        return None
    hit = pts[inside]
    return hit[0], hit[-1]


class SVMMargin(Scene):
    def construct(self):
        self.build_plot()
        self.beat_many_lines()
        self.beat_derive_margin()
        self.beat_shrink_w()
        self.beat_constraint()
        self.beat_support_vectors()

    # ------------------------------------------------------------ setup ----
    def build_plot(self):
        # equal units on both axes (1.15 screen units per petal-mm) so that
        # "perpendicular" on screen really is perpendicular in the data
        s = 1.15
        self.ax = Axes(
            x_range=[XLO, XHI, 1], y_range=[YLO, YHI, 0.5],
            x_length=(XHI - XLO) * s, y_length=(YHI - YLO) * s,
            tips=False,
            axis_config=dict(color=SOFT, stroke_width=2,
                             tick_size=0.05, font_size=18,
                             decimal_number_config=dict(num_decimal_places=0,
                                                        color=SOFT)),
            x_axis_config=dict(numbers_to_include=[1, 2, 3, 4, 5]),
            y_axis_config=dict(numbers_to_include=[0, 1, 2]),
        ).move_to([-3.3, -0.55, 0])

        xlab = Text("petal length", font_size=17, color=SOFT)
        xlab.next_to(self.ax.x_axis, DOWN, buff=0.28).align_to(self.ax, RIGHT)
        ylab = Text("petal width", font_size=17, color=SOFT).rotate(PI / 2)
        ylab.next_to(self.ax.y_axis, LEFT, buff=0.10)

        # one dot per flower, coloured by the notebook's class colours
        self.dots = VGroup(*[
            Dot(self.ax.c2p(*p), radius=0.045,
                color=(C_POS if lab == 1 else C_NEG), fill_opacity=0.85)
            for p, lab in zip(X, Y)
        ])

        key = VGroup(
            Dot(radius=0.055, color=C_NEG), MathTex(r"\text{setosa } (y=-1)",
                                                    font_size=24, color=C_NEG),
            Dot(radius=0.055, color=C_POS), MathTex(r"\text{versicolor } (y=+1)",
                                                    font_size=24, color=C_POS),
        ).arrange(RIGHT, buff=0.16)
        key[2].shift(RIGHT * 0.3)
        key[3].shift(RIGHT * 0.3)
        key.move_to([-3.4, 1.32, 0])

        rule = Line([0.05, 3.3, 0], [0.05, -3.3, 0],
                    stroke_width=1.2, color="#dcdcdc")

        self.caption = None      # first say() has nothing to fade out yet

        self.play(Create(self.ax), FadeIn(xlab, ylab), run_time=0.31)
        self.play(LaggedStartMap(FadeIn, self.dots, lag_ratio=0.01),
                  FadeIn(key), FadeIn(rule), run_time=0.37)
        self.wait(0.25)

    def say(self, tex, run_time=0.3, wait=0.0):
        """Swap the caption above the plot."""
        new = Tex(tex, font_size=30).move_to([-3.4, 2.35, 0])
        if new.width > 6.1:
            new.scale_to_fit_width(6.1)
        anims = [FadeIn(new, shift=UP * 0.12)]
        if self.caption is not None:
            anims.append(FadeOut(self.caption, shift=UP * 0.12))
        self.play(*anims, run_time=run_time)
        self.caption = new
        if wait:
            self.wait(wait)

    def data_line(self, n, c, **kw):
        a, b = clip_line(np.array(n, dtype=float), c)
        return Line(self.ax.c2p(*a), self.ax.c2p(*b), **kw)

    # ---------------------------------------------- 1. many valid lines ----
    def beat_many_lines(self):
        self.say(r"Many straight lines separate these two species.")

        self.eq0 = MathTex(r"w \cdot x + b = 0", font_size=38)
        self.eq0.move_to([3.35, 2.75, 0])
        self.play(Write(self.eq0), run_time=0.3)

        # three perfectly valid but arbitrary boundaries: a vertical cut, a
        # horizontal one, and a slanted one. Each is the midline of the gap
        # in its own direction, so each classifies all 100 flowers correctly.
        angles = [0.0, 90.0, 65.0]
        line = self.data_line([1.0, 0.0], 2.45, color=SOFT, stroke_width=4)
        self.play(Create(line), run_time=0.3)
        self.wait(0.25)
        for deg in angles[1:]:
            th = np.deg2rad(deg)
            n = np.array([np.cos(th), np.sin(th)])
            proj = X @ n
            c = 0.5 * (proj[Y == -1].max() + proj[Y == 1].min())
            nxt = self.data_line(n, c, color=SOFT, stroke_width=4)
            self.play(ReplacementTransform(line, nxt), run_time=0.3)
            line = nxt
            self.wait(0.25)

        self.say(r"All of them get every flower right, so \\ ``correct'' cannot pick one out.",
                 wait=0.9)

        # settle on the one the notebook's search found
        self.boundary = self.data_line(NHAT, COFF, color=INK, stroke_width=4)
        self.play(ReplacementTransform(line, self.boundary), run_time=0.3)
        self.wait(0.25)

    # -------------------------------------------- 2. derive 2 / ||w|| ------
    def beat_derive_margin(self):
        self.say(r"Ask instead for the widest gap. \\ Put a line at $w\cdot x+b=\pm 1$.")

        self.mp = self.data_line(NHAT, COFF + 1 / WNORM, color=C_MARGIN,
                                 stroke_width=3.5).set_stroke(opacity=0.95)
        self.mm = self.data_line(NHAT, COFF - 1 / WNORM, color=C_MARGIN,
                                 stroke_width=3.5).set_stroke(opacity=0.95)
        self.mp = DashedVMobject(self.mp, num_dashes=18)
        self.mm = DashedVMobject(self.mm, num_dashes=18)

        self.lab_p = MathTex(r"+1", font_size=24, color=C_MARGIN)
        self.lab_m = MathTex(r"-1", font_size=24, color=C_MARGIN)
        self.place_margin_labels(WNORM)

        self.play(Create(self.mp), Create(self.mm),
                  FadeIn(self.lab_p), FadeIn(self.lab_m), run_time=0.3)
        self.wait(0.25)

        # w is the normal: an arrow of length 1/||w|| from the boundary out to
        # the +1 line. Later it grows into the full 2/||w|| double arrow.
        foot = np.array([1.70, np.sqrt(2) * COFF - 1.70])
        tip = foot + NHAT / WNORM
        self.warrow = Arrow(self.ax.c2p(*foot), self.ax.c2p(*tip),
                            buff=0, color=C_W, stroke_width=4,
                            max_tip_length_to_length_ratio=0.28)
        wlab = MathTex(r"w", font_size=28, color=C_W)
        wlab.next_to(self.warrow.get_end(), UP, buff=0.06)
        self.say(r"$w$ points straight across the boundary.")
        self.play(GrowArrow(self.warrow), FadeIn(wlab), run_time=0.3)
        self.wait(0.25)

        # the two support vectors are the x+ and x- of the derivation
        dp = Dot(self.ax.c2p(*XPLUS), radius=0.075, color=C_POS)
        dm = Dot(self.ax.c2p(*XMINUS), radius=0.075, color=C_NEG)
        lp = MathTex(r"x_+", font_size=26, color=C_POS).next_to(dp, UL, buff=0.02)
        lm = MathTex(r"x_-", font_size=26, color=C_NEG).next_to(dm, UR, buff=0.02)
        self.say(r"Take one point on each line.")
        self.play(FadeIn(dp, scale=0.5), FadeIn(dm, scale=0.5),
                  Write(lp), Write(lm), run_time=0.3)

        # the two facts we are about to subtract
        eq1 = MathTex(r"w", r"\cdot", r"x_+", r"+", r"b", r"=", r"+1", font_size=38)
        eq2 = MathTex(r"w", r"\cdot", r"x_-", r"+", r"b", r"=", r"-1", font_size=38)
        for e in (eq1, eq2):
            e[0].set_color(C_W)
            e[4].set_color(SOFT)
        eq1[2].set_color(C_POS)
        eq2[2].set_color(C_NEG)
        eq1.move_to([3.35, 1.80, 0])
        eq2.move_to([3.35, 1.00, 0])
        self.play(Write(eq1), run_time=0.3)
        self.play(Write(eq2), run_time=0.3)
        self.wait(0.25)

        # subtract: the b's cancel and 1 - (-1) = 2
        self.say(r"Subtract. The offset $b$ cancels.")
        strike1 = Line(eq1[4].get_corner(DL), eq1[4].get_corner(UR),
                       color=C_BAD, stroke_width=3).scale(1.4)
        strike2 = Line(eq2[4].get_corner(DL), eq2[4].get_corner(UR),
                       color=C_BAD, stroke_width=3).scale(1.4)
        self.play(Create(strike1), Create(strike2), run_time=0.3)
        self.wait(0.25)
        self.play(FadeOut(strike1, strike2, eq1[3], eq1[4], eq2[3], eq2[4]),
                  run_time=0.3)

        eq3 = MathTex(r"w", r"\cdot", r"(", r"x_+", r"-", r"x_-", r")",
                      r"=", r"2", font_size=38)
        eq3[0].set_color(C_W)
        eq3[3].set_color(C_POS)
        eq3[5].set_color(C_NEG)
        eq3.move_to([3.35, 0.05, 0])

        def merge(a, b, target):
            """Move both copies of a cancelled-down term onto one target."""
            ghost = target.copy().set_opacity(0)
            return [ReplacementTransform(a, target),
                    ReplacementTransform(b, ghost)]

        anims = []
        anims += merge(eq1[0], eq2[0], eq3[0])       # w
        anims += merge(eq1[1], eq2[1], eq3[1])       # dot
        anims += merge(eq1[5], eq2[5], eq3[7])       # =
        anims += merge(eq1[6], eq2[6], eq3[8])       # +1 and -1 make 2
        anims += [ReplacementTransform(eq1[2], eq3[3]),
                  ReplacementTransform(eq2[2], eq3[5]),
                  FadeIn(eq3[2]), FadeIn(eq3[4]), FadeIn(eq3[6])]
        self.play(*anims, run_time=0.34)
        self.wait(0.25)

        # divide both sides by ||w||: the left becomes a projection onto the
        # unit normal, which is the perpendicular width we actually want
        self.say(r"Divide by $\lVert w\rVert$: the left side is now \\ "
                 r"the distance measured across the line.")
        eq4 = MathTex(r"\frac{w}{\lVert w\rVert}", r"\cdot", r"(", r"x_+", r"-",
                      r"x_-", r")", r"=", r"\frac{2}{\lVert w\rVert}", font_size=38)
        eq4[0].set_color(C_W)
        eq4[3].set_color(C_POS)
        eq4[5].set_color(C_NEG)
        eq4[8].set_color(C_W)
        eq4.move_to([3.35, -1.05, 0])
        self.play(*[ReplacementTransform(eq3[i].copy(), eq4[i])
                    for i in range(9)], run_time=0.34)
        self.wait(0.25)

        # and the same quantity, drawn: the arrow along w spans the two lines
        gap = DoubleArrow(self.ax.c2p(*(foot - NHAT / WNORM)),
                          self.ax.c2p(*(foot + NHAT / WNORM)),
                          buff=0, color=C_W, stroke_width=4,
                          max_tip_length_to_length_ratio=0.14)
        glab = MathTex(r"\tfrac{2}{\lVert w\rVert}", font_size=26, color=C_W)
        glab.next_to(gap, LEFT, buff=0.10)
        self.play(ReplacementTransform(self.warrow, gap),
                  ReplacementTransform(wlab, glab), run_time=0.3)
        self.gap, self.glab = gap, glab
        self.wait(0.25)

        eq5 = MathTex(r"\text{margin}", r"=", r"\frac{2}{\lVert w\rVert}",
                      r"=", r"\frac{2}{1.58}", r"=", r"1.27", font_size=38)
        eq5[2].set_color(C_W)
        eq5[6].set_color(C_W)
        eq5.move_to([3.35, -2.35, 0])
        self.play(TransformFromCopy(eq4[8], eq5[2]),
                  FadeIn(eq5[0], eq5[1]), run_time=0.3)
        self.play(FadeIn(eq5[3], eq5[4], eq5[5], eq5[6]), run_time=0.3)
        self.wait(0.35)

        # clear the working, keep the result
        self.eq5 = eq5
        self.play(FadeOut(eq3, eq4, dp, dm, lp, lm),
                  eq5.animate.move_to([3.35, 1.75, 0]), run_time=0.3)

    def place_margin_labels(self, norm):
        """Park the +-1 tags at the point where each margin line leaves the
        bottom of the plot; they slide sideways as ||w|| changes."""
        for lab, sgn in ((self.lab_p, +1), (self.lab_m, -1)):
            a, b = clip_line(NHAT, COFF + sgn / norm)
            end = a if a[1] < b[1] else b
            lab.move_to(self.ax.c2p(*end) + LEFT * 0.20 + UP * 0.34)

    # ------------------------------------------- 3. shrink and grow w ------
    def beat_shrink_w(self):
        norm = ValueTracker(WNORM)

        def redraw(sgn):
            def f(m):
                a, b = clip_line(NHAT, COFF + sgn / norm.get_value())
                base = Line(self.ax.c2p(*a), self.ax.c2p(*b),
                            color=C_MARGIN, stroke_width=3.5)
                m.become(DashedVMobject(base, num_dashes=18))
            return f

        self.mp.add_updater(redraw(+1))
        self.mm.add_updater(redraw(-1))
        self.lab_p.add_updater(lambda m: self.place_margin_labels(norm.get_value()))

        # the drawn 2/||w|| arrow tracks the width too
        foot = np.array([1.70, np.sqrt(2) * COFF - 1.70])

        def redraw_gap(m):
            h = NHAT / norm.get_value()
            m.become(DoubleArrow(self.ax.c2p(*(foot - h)), self.ax.c2p(*(foot + h)),
                                 buff=0, color=C_W, stroke_width=4,
                                 max_tip_length_to_length_ratio=0.14))
        self.gap.add_updater(redraw_gap)
        self.glab.add_updater(lambda m: m.next_to(self.gap, LEFT, buff=0.10))

        # a live readout of ||w|| and the width it buys
        nlab = MathTex(r"\lVert w\rVert =", font_size=36, color=C_W)
        nval = DecimalNumber(WNORM, num_decimal_places=2, font_size=36, color=C_W)
        mlab = MathTex(r"\frac{2}{\lVert w\rVert} =", font_size=36, color=C_W)
        mval = DecimalNumber(2 / WNORM, num_decimal_places=2, font_size=36, color=C_W)
        row1 = VGroup(nlab, nval).arrange(RIGHT, buff=0.16).move_to([3.35, 0.35, 0])
        row2 = VGroup(mlab, mval).arrange(RIGHT, buff=0.16).move_to([3.35, -0.75, 0])
        nval.add_updater(lambda m: m.set_value(norm.get_value()))
        mval.add_updater(lambda m: m.set_value(2 / norm.get_value()))

        self.say(r"Now hold the boundary still and \\ change only the length of $w$.")
        self.play(FadeIn(row1, row2), run_time=0.3)
        self.play(norm.animate.set_value(2.60), run_time=0.3)
        self.wait(0.25)

        self.say(r"Shrink $\lVert w\rVert$ \dots")
        self.play(norm.animate.set_value(WNORM), run_time=0.56, rate_func=linear)
        self.say(r"\dots and the margin lines move apart.", wait=0.8)

        self.say(r"Grow it, and they close in.")
        self.play(norm.animate.set_value(2.60), run_time=0.43, rate_func=linear)
        self.wait(0.25)
        self.play(norm.animate.set_value(WNORM), run_time=0.37, rate_func=linear)
        self.wait(0.25)

        for m in (self.mp, self.mm, self.lab_p, self.gap, self.glab, nval, mval):
            m.clear_updaters()

        self.say(r"So a wide margin is a short $w$.")
        obj = MathTex(r"\max \ \frac{2}{\lVert w\rVert}", r"\iff",
                      r"\min \ \tfrac{1}{2}\lVert w\rVert^{2}", font_size=38)
        obj[0].set_color(C_W)
        obj[2].set_color(C_W)
        obj.move_to([3.35, -0.20, 0])
        if obj.width > 6.1:
            obj.scale_to_fit_width(6.1)
        self.play(FadeOut(row1, row2), FadeIn(obj), run_time=0.3)
        self.wait(0.42)
        self.objective = obj

    # ------------------------------------------------ 4. the constraint ----
    def beat_constraint(self):
        self.say(r"What stops $\lVert w\rVert$ shrinking forever?")
        con = MathTex(r"y_i\,(w \cdot x_i + b)", r"\ \ge\ ", r"1", font_size=38)
        con.move_to([3.35, -1.30, 0])
        self.play(FadeIn(con), run_time=0.3)
        self.wait(0.25)

        # drag one versicolor flower in towards the boundary and watch its
        # functional margin fall through 1
        start = X[DRAG_I].astype(float)
        end = np.array([2.55, 0.80])
        t = ValueTracker(0.0)

        def pos():
            return start + (end - start) * t.get_value()

        def fval():
            p = pos()
            return float(np.dot(W, p) + B)         # y = +1 for this flower

        self.dots[DRAG_I].set_opacity(0.0)
        mover = always_redraw(lambda: Dot(
            self.ax.c2p(*pos()), radius=0.085,
            color=(C_BAD if fval() < 1 else C_POS)))
        def make_trail():
            # nothing to draw until the flower has actually moved
            if np.linalg.norm(pos() - start) < 1e-3:
                return VGroup()
            return DashedLine(self.ax.c2p(*start), self.ax.c2p(*pos()),
                              color=SOFT, stroke_width=2, dash_length=0.06)

        trail = always_redraw(make_trail)

        readout = VGroup(
            MathTex(r"y_i\,(w \cdot x_i + b) =", font_size=34),
            DecimalNumber(fval(), num_decimal_places=2, font_size=34),
        ).arrange(RIGHT, buff=0.16).move_to([3.35, -2.30, 0])
        def refresh(m):
            m.set_value(fval())
            m.set_color(C_BAD if fval() < 1 else INK)

        readout[1].add_updater(refresh)
        readout[0].add_updater(
            lambda m: m.set_color(C_BAD if fval() < 1 else INK))

        warn = Tex(r"constraint violated", font_size=28, color=C_BAD)
        warn.move_to([3.35, -3.00, 0])
        warn.add_updater(lambda m: m.set_opacity(0.0 if fval() >= 1 else 1.0))

        self.play(FadeIn(mover), FadeIn(readout), run_time=0.3)
        self.add(trail, warn)
        self.say(r"Push one flower towards the line.")
        self.play(t.animate.set_value(1.0), run_time=0.81, rate_func=linear)
        self.play(Indicate(con, color=C_BAD, scale_factor=1.12), run_time=0.3)
        self.say(r"Below 1, and this $w$ is no longer allowed.", wait=1.1)

        # put it back
        self.play(t.animate.set_value(0.0), run_time=0.34, rate_func=linear)
        self.wait(0.25)
        for m in (readout[0], readout[1], warn):
            m.clear_updaters()
        self.play(FadeOut(mover, trail, readout, warn, con), run_time=0.3)
        self.dots[DRAG_I].set_opacity(0.85)

    # -------------------------------------------- 5. the support vectors ---
    def beat_support_vectors(self):
        self.say(r"Only the flowers touching the margin \\ hold the line in place.")

        rings = VGroup(*[
            Circle(radius=0.13, color=INK, stroke_width=3).move_to(self.ax.c2p(*p))
            for p in X[SUPPORT]
        ])
        others = VGroup(*[d for i, d in enumerate(self.dots) if not SUPPORT[i]])
        keep = VGroup(*[d for i, d in enumerate(self.dots) if SUPPORT[i]])

        self.play(Create(rings), run_time=0.3)
        self.play(others.animate.set_opacity(0.22),
                  *[d.animate.scale(1.5) for d in keep], run_time=0.3)

        count = Tex(r"2 support vectors out of 100", font_size=32)
        count.move_to([3.35, -1.35, 0])
        self.play(FadeIn(count), run_time=0.3)
        self.wait(0.35)

        self.say(r"Delete the other 98 and the \\ boundary does not move.")
        self.play(FadeOut(others), run_time=0.31)
        self.wait(0.77)


# ------------------------------------------------------------- render -------
if __name__ == '__main__':
    import glob
    import os
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    here = Path(__file__).resolve()
    repo = here.parents[2]
    gif = repo / 'animations' / 'gifs' / 'svm_margin.gif'
    gif.parent.mkdir(parents=True, exist_ok=True)

    # LaTeX lives in a user tree; manim needs both of these on PATH
    env = dict(os.environ)
    env['PATH'] = '/Library/TeX/texbin:/opt/homebrew/bin:' + env.get('PATH', '')

    # This machine's homebrew ffmpeg is linked against an x265 dylib that has
    # been upgraded out from under it, so `ffmpeg` dies in dyld before it runs.
    # The library is still on disk in an older Cellar directory, and pointing
    # DYLD_FALLBACK_LIBRARY_PATH at it is enough. /bin/bash strips DYLD_* under
    # SIP, so the setting has to travel in a tiny wrapper on PATH rather than in
    # the environment. If ffmpeg already works, none of this happens.
    shim = None
    probe = subprocess.run(['ffmpeg', '-version'], env=env,
                           capture_output=True)
    if probe.returncode != 0:
        libs = sorted(glob.glob('/opt/homebrew/Cellar/x265/*/lib'))
        shim = Path(tempfile.mkdtemp(prefix='ffmpeg_shim_'))
        wrapper = shim / 'ffmpeg'
        wrapper.write_text(
            '#!/bin/sh\n'
            f'export DYLD_FALLBACK_LIBRARY_PATH={":".join(libs)}\n'
            'exec /opt/homebrew/bin/ffmpeg "$@"\n')
        wrapper.chmod(0o755)
        env['PATH'] = str(shim) + ':' + env['PATH']

    media = Path(tempfile.mkdtemp(prefix='manim_svm_'))
    try:
        subprocess.run(
            ['/opt/anaconda3/envs/tf_mps/bin/python', '-m', 'manim',
             '-qm', '--format=mp4', '--media_dir', str(media),
             '-o', 'svm_margin', str(here), 'SVMMargin'],
            check=True, env=env, cwd=str(repo))
        mp4 = media / 'videos' / here.stem / '720p30' / 'svm_margin.mp4'
        # two-pass palette encode; a naive per-frame palette is 3-5x bigger
        subprocess.run(
            ['bash', str(repo / 'animations' / 'gif_from_mp4.sh'),
             str(mp4), str(gif), '640', '15'],
            check=True, env=env)
    finally:
        shutil.rmtree(media, ignore_errors=True)
        if shim is not None:
            shutil.rmtree(shim, ignore_errors=True)

    mb = gif.stat().st_size / 1e6
    print(f'wrote {gif}  {mb:.2f} MB')
    if mb > 1.5:
        raise SystemExit(f'OVER BUDGET: {mb:.2f} MB > 1.5 MB cap')
