"""Backpropagation through time is a product, and products behave geometrically.

Companion animation for `neural_networks/rnn_lstm.ipynb`.

The notebook reads 60-base DNA windows one base at a time, so the gradient that
reaches the first base has passed through 59 intermediate steps and picked up a
Jacobian at every one of them. That product is the subject here: it is
geometric, so it either decays toward zero or blows up, and only a factor
within a hair of 1 leaves it intact.

Every number on screen is measured, not drawn. The notebook's own `RNN` and
`LSTM` classes are copied in verbatim below, together with its data loading and
its training loop, and both classes already record
`norms[t] = ||dJ/dh_t||` on the way past when called with `keep_norms=True`.
The scene plays back three things:

  1. the same untrained cell at three settings of ||W_h|| (0.7x, 1x and 1.6x
     the notebook's initialisation), which gives measured per-step factors of
     0.690, 0.938 and 1.093 - the decaying and the exploding regime of one
     expression;
  2. the gradient fading as it travels back through the chain, driven by the
     norms measured at the notebook's own initialisation;
  3. both models trained exactly as the notebook trains them, whose measured
     per-step factors are 1.04 for the RNN and 1.00 for the LSTM.

On the notebook's task the RNN scores *higher* than the LSTM (0.8918 against
0.8386), which the closing card says out loud: the splice motif sits in the
middle of the window, about 30 steps from the end, which is inside the reach a
plain RNN still has. The animation is about the mechanism, not a scoreboard.

Run from the repo root with no arguments to regenerate the GIF:

    /opt/anaconda3/envs/tf_mps/bin/python animations/manim_scenes/bptt_vanishing.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from manim import *

REPO = Path(__file__).resolve().parents[2]
NAME = "bptt_vanishing"

# --- shared look -----------------------------------------------------------
# The matplotlib animations use ggplot on white with a fixed five-colour cycle;
# this scene keeps the same palette so the whole set reads as one thing. The
# notebook draws the RNN in blue and the LSTM in orange, and so does this.
INK = "#22242a"
PAPER = "#ffffff"
BLUE = "#2a78d6"     # the RNN's hidden chain
ORANGE = "#eb6834"   # the LSTM's cell state
GOLD = "#eda100"     # the gradient travelling backwards
RED = "#c0392b"      # a product that is running away
GREY = "#9aa0aa"

config.background_color = PAPER


# ===========================================================================
# The notebook's code, unchanged.
# ===========================================================================
def softmax(scores):
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cross_entropy(probs, y):
    return -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()


# a forward pass keeps every timestep in memory, so whole-dataset passes go in chunks
def predict_probs(model, X, batch=256):
    out = np.zeros((len(X), 3))
    for s in range(0, len(X), batch):
        _, p = model.forward(X[s:s + batch])
        out[s:s + batch] = p
    return out


class RNN:

    def __init__(self, input_size, hidden_size, n_classes, seed=0):
        r = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(hidden_size)
        self.Wx = r.normal(0, s, (input_size, hidden_size))
        self.Wh = r.normal(0, s, (hidden_size, hidden_size))
        self.b = np.zeros(hidden_size)
        self.Wy = r.normal(0, s, (hidden_size, n_classes))
        self.by = np.zeros(n_classes)
        self.hidden_size = hidden_size

    def params(self):
        return [self.Wx, self.Wh, self.b, self.Wy, self.by]

    def forward(self, X):
        n, T, _ = X.shape
        H = np.zeros((T + 1, n, self.hidden_size))     # H[0] is the zero initial state
        for t in range(T):
            H[t + 1] = np.tanh(X[:, t] @ self.Wx + H[t] @ self.Wh + self.b)
        probs = softmax(H[T] @ self.Wy + self.by)
        return H, probs

    def backward(self, X, y, H, probs, keep_norms=False):
        n, T, _ = X.shape

        # softmax with cross-entropy collapses to p - onehot(y)
        dscores = probs.copy()
        dscores[np.arange(n), y] -= 1.0
        dscores /= n

        dWy = H[T].T @ dscores
        dby = dscores.sum(axis=0)

        # one shared set of gradients that every timestep adds into
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)

        dh = dscores @ self.Wy.T
        norms = np.zeros(T + 1)
        norms[T] = np.linalg.norm(dh)

        for t in range(T - 1, -1, -1):
            dz = dh * (1.0 - H[t + 1] ** 2)            # through the tanh
            dWx += X[:, t].T @ dz
            dWh += H[t].T @ dz
            db += dz.sum(axis=0)
            dh = dz @ self.Wh.T                        # one step further back
            norms[t] = np.linalg.norm(dh)

        grads = [dWx, dWh, db, dWy, dby]
        if keep_norms:
            return grads, norms
        return grads

    def predict(self, X, batch=256):
        return predict_probs(self, X, batch).argmax(axis=1)


class LSTM:

    def __init__(self, input_size, hidden_size, n_classes, seed=0):
        r = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(hidden_size + input_size)
        # one matrix holding [Wf, Wi, Wo, Wc] side by side
        self.W = r.normal(0, s, (hidden_size + input_size, 4 * hidden_size))
        self.b = np.zeros(4 * hidden_size)
        self.b[:hidden_size] = 1.0                     # forget gates start open
        self.Wy = r.normal(0, 1.0 / np.sqrt(hidden_size), (hidden_size, n_classes))
        self.by = np.zeros(n_classes)
        self.hidden_size = hidden_size

    def params(self):
        return [self.W, self.b, self.Wy, self.by]

    def forward(self, X):
        n, T, d = X.shape
        H = self.hidden_size
        h = np.zeros((T + 1, n, H))
        c = np.zeros((T + 1, n, H))
        gates = np.zeros((T, n, 4 * H))
        cat = np.zeros((T, n, H + d))

        for t in range(T):
            cat[t] = np.hstack([h[t], X[:, t]])
            a = cat[t] @ self.W + self.b
            g = np.empty_like(a)
            g[:, :3 * H] = sigmoid(a[:, :3 * H])       # f, i, o
            g[:, 3 * H:] = np.tanh(a[:, 3 * H:])       # candidate
            gates[t] = g
            f, i, o, cbar = g[:, :H], g[:, H:2 * H], g[:, 2 * H:3 * H], g[:, 3 * H:]
            c[t + 1] = f * c[t] + i * cbar
            h[t + 1] = o * np.tanh(c[t + 1])

        probs = softmax(h[T] @ self.Wy + self.by)
        return (h, c, gates, cat), probs

    def backward(self, X, y, cache, probs, keep_norms=False):
        h, c, gates, cat = cache
        n, T, _ = X.shape
        H = self.hidden_size

        dscores = probs.copy()
        dscores[np.arange(n), y] -= 1.0
        dscores /= n
        dWy = h[T].T @ dscores
        dby = dscores.sum(axis=0)

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dh = dscores @ self.Wy.T
        dc = np.zeros((n, H))
        norms = np.zeros(T + 1)
        norms[T] = np.linalg.norm(dh)

        for t in range(T - 1, -1, -1):
            g = gates[t]
            f, i, o, cbar = g[:, :H], g[:, H:2 * H], g[:, 2 * H:3 * H], g[:, 3 * H:]
            tc = np.tanh(c[t + 1])

            do = dh * tc
            dcell = dc + dh * o * (1.0 - tc ** 2)      # from h plus the next step
            df = dcell * c[t]
            di = dcell * cbar
            dcbar = dcell * i
            dc = dcell * f                             # the additive path back in time

            da = np.empty((n, 4 * H))
            da[:, :H] = df * f * (1 - f)
            da[:, H:2 * H] = di * i * (1 - i)
            da[:, 2 * H:3 * H] = do * o * (1 - o)
            da[:, 3 * H:] = dcbar * (1 - cbar ** 2)

            dW += cat[t].T @ da
            db += da.sum(axis=0)
            dh = (da @ self.W.T)[:, :H]                # first H columns of d[h, x]
            norms[t] = np.linalg.norm(dh)

        grads = [dW, db, dWy, dby]
        if keep_norms:
            return grads, norms
        return grads

    def predict(self, X, batch=256):
        return predict_probs(self, X, batch).argmax(axis=1)


def clip_grads(grads, limit):
    total = np.sqrt(sum((g ** 2).sum() for g in grads))
    if limit is not None and total > limit:
        scale = limit / total
        for g in grads:
            g *= scale
    return total


def train(model, X_tr, y_tr, epochs, lr, batch=64, clip=5.0, seed=7):
    """The notebook's training loop with the per-epoch reporting stripped out.

    The random draws are identical to the notebook's, so the weights this
    produces are the weights the notebook trains.
    """
    r = np.random.default_rng(seed)
    for _ in range(epochs):
        order = r.permutation(len(X_tr))
        for s in range(0, len(X_tr), batch):
            sel = order[s:s + batch]
            cache, probs = model.forward(X_tr[sel])
            grads = model.backward(X_tr[sel], y_tr[sel], cache, probs)
            clip_grads(grads, clip)
            for p, g in zip(model.params(), grads):
                p -= lr * g
    return model


def load_splice():
    """The notebook's data prep: 3,190 windows of 60 bases, one-hot over 5 symbols."""
    from ucimlrepo import fetch_ucirepo

    splice = fetch_ucirepo(id=69)
    frame = splice.data.features.apply(lambda col: col.str.strip())
    labels = splice.data.targets.values.ravel()

    BASES = ['A', 'C', 'G', 'T']
    CLASSES = ['EI', 'IE', 'N']

    # index 4 is UNK, so anything that is not ACGT lands there
    chars = frame.values
    codes = np.full(chars.shape, 4, dtype=int)
    for k, base in enumerate(BASES):
        codes[chars == base] = k

    X_all = np.eye(5)[codes]
    y_all = np.array([CLASSES.index(c) for c in labels])

    rng = np.random.default_rng(0)
    order = rng.permutation(len(X_all))
    split = int(0.8 * len(X_all))
    return (X_all[order[:split]], y_all[order[:split]],
            X_all[order[split:]], y_all[order[split:]])


# ===========================================================================
# The measurements the animation plays back.
# ===========================================================================
HIDDEN = 48        # the notebook's hidden size
EPOCHS = 40        # the notebook's training length
SCALES = (1.6, 1.0, 0.7)   # multipliers on the initialised ||W_h||


def measure():
    X_train, y_train, X_test, y_test = load_splice()
    X_probe, y_probe = X_test[:128], y_test[:128]      # the notebook's probe batch

    out = {'norms': {}, 'gamma': {}}

    # --- one untrained cell at three weight scales --------------------------
    # Only W_h changes, so any difference in the backward pass comes from the
    # size of the factor the product picks up at each of the 60 steps.
    for scale in SCALES:
        model = RNN(5, HIDDEN, 3, seed=1)
        model.Wh *= scale
        H, probs = model.forward(X_probe)
        _, norms = model.backward(X_probe, y_probe, H, probs, keep_norms=True)
        out['norms'][scale] = norms
        out['gamma'][scale] = (norms[0] / norms[60]) ** (1 / 60)
        if scale == 1.0:
            # the average tanh' seen along the sequence: every factor carries one
            out['tanh_prime'] = float((1.0 - H[1:] ** 2).mean())

    # --- both models trained exactly as the notebook trains them ------------
    rnn = train(RNN(5, HIDDEN, 3, seed=1), X_train, y_train, EPOCHS, lr=0.02)
    H, probs = rnn.forward(X_probe)
    _, out['norms']['rnn'] = rnn.backward(X_probe, y_probe, H, probs, keep_norms=True)

    lstm = train(LSTM(5, HIDDEN, 3, seed=1), X_train, y_train, EPOCHS, lr=0.5)
    cache, probs = lstm.forward(X_probe)
    _, out['norms']['lstm'] = lstm.backward(X_probe, y_probe, cache, probs, keep_norms=True)
    out['forget'] = float(cache[2][:, :, :HIDDEN].mean())     # mean forget gate

    for key in ('rnn', 'lstm'):
        n = out['norms'][key]
        out['gamma'][key] = (n[0] / n[60]) ** (1 / 60)

    return out


def sci(value, digits=1):
    """Format a number for LaTeX: plain when it is readable, powers of ten when not."""
    if 1e-2 <= abs(value) < 1e3:
        return f"{value:.3g}"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0 ** exponent
    return rf"{mantissa:.{digits}f} \times 10^{{{exponent}}}"


# ===========================================================================
# The scene.
# ===========================================================================
NODE_X = [-4.7, -2.35, 0.0, 2.35, 4.7]      # five drawn steps of the sixty
NODE_T = [1, 2, 30, 59, 60]                 # the timestep each one stands for
CHAIN_Y = 1.85
CELL_Y = 2.78
LOSS_X = 6.1
RADIUS = 0.36

EQ1_Y = -0.45
EQ2_Y = -1.5
EQ3_Y = -2.35
CAP_Y = -3.15


class BpttVanishing(Scene):

    def construct(self):
        MathTex.set_default(color=INK)
        Tex.set_default(color=INK)
        Text.set_default(color=INK)

        M = measure()

        # ---- the unrolled chain --------------------------------------------
        node_labels = ["h_1", "h_2", r"\cdots", "h_{59}", "h_{60}"]
        nodes, labels = VGroup(), VGroup()
        for x, text in zip(NODE_X, node_labels):
            circle = Circle(radius=RADIUS, stroke_color=BLUE, stroke_width=2.2,
                            fill_color=PAPER, fill_opacity=1.0)
            if text == r"\cdots":
                # the 56 steps there is no room to draw
                circle = DashedVMobject(circle, num_dashes=18)
                circle.set_stroke(GREY, width=1.8)
            circle.move_to([x, CHAIN_Y, 0])
            nodes.add(circle)
            labels.add(MathTex(text, font_size=26).move_to([x, CHAIN_Y, 0]))

        arrows = VGroup(*[
            Arrow([NODE_X[i] + RADIUS, CHAIN_Y, 0], [NODE_X[i + 1] - RADIUS, CHAIN_Y, 0],
                  buff=0.04, stroke_width=2.4, color=BLUE, tip_length=0.15)
            for i in range(4)
        ])

        inputs = VGroup()
        for i in (0, 1, 3, 4):
            inputs.add(Arrow([NODE_X[i], 0.95, 0], [NODE_X[i], CHAIN_Y - RADIUS, 0],
                             buff=0.03, stroke_width=1.8, color=GREY, tip_length=0.12))
            inputs.add(MathTex(f"x_{{{NODE_T[i]}}}", font_size=22,
                               color=GREY).move_to([NODE_X[i], 0.7, 0]))
        skipped = Tex("56 steps", font_size=20, color=GREY).move_to([0.0, 1.05, 0])

        loss_box = RoundedRectangle(width=0.78, height=0.62, corner_radius=0.1,
                                    stroke_color=GREY, stroke_width=2,
                                    fill_color=PAPER, fill_opacity=1
                                    ).move_to([LOSS_X, CHAIN_Y, 0])
        loss_label = MathTex("J", font_size=28).move_to([LOSS_X, CHAIN_Y, 0])
        loss_arrow = Arrow([NODE_X[4] + RADIUS, CHAIN_Y, 0], [LOSS_X - 0.39, CHAIN_Y, 0],
                           buff=0.04, stroke_width=2.4, color=BLUE, tip_length=0.15)

        chain = VGroup(nodes, labels, arrows, inputs, skipped)

        title = Tex(r"Backprop through time multiplies one Jacobian 59 times",
                    font_size=36).move_to([0, 3.5, 0])

        self.play(Write(title), run_time=0.45)
        self.play(LaggedStart(*[FadeIn(n) for n in nodes], lag_ratio=0.12),
                  LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.12),
                  LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.12),
                  run_time=0.6)
        self.play(FadeIn(inputs), FadeIn(skipped), run_time=0.3)

        forward = MathTex(r"h_t", r"=\tanh\!\left(", r"W_x^{\top} x_t",
                          r"+", r"W_h^{\top}", r"h_{t-1}", r"+\, b\right)",
                          font_size=36).move_to([0, EQ1_Y, 0])
        forward[4].set_color(BLUE)
        sub = Tex(r"one 60-base window, the same $W_h$ at every step",
                  font_size=26, color=GREY).move_to([0, EQ2_Y, 0])
        self.play(FadeIn(forward), run_time=0.3)
        self.play(FadeIn(sub), run_time=0.3)
        self.wait(0.39)

        # ---- the backward pass picks up a factor at every step --------------
        self.play(FadeIn(loss_box), FadeIn(loss_label), GrowArrow(loss_arrow),
                  run_time=0.3)

        arcs, arc_labels = VGroup(), VGroup()
        for i in range(4):
            arc = CurvedArrow(np.array([NODE_X[i + 1], CHAIN_Y + RADIUS + 0.05, 0]),
                              np.array([NODE_X[i] + 0.02, CHAIN_Y + RADIUS + 0.05, 0]),
                              angle=-PI / 2.6, color=GOLD, stroke_width=2.4,
                              tip_length=0.15)
            arcs.add(arc)
            arc_labels.add(MathTex(r"W_h^{\top} D_i", font_size=24, color=GOLD)
                           .move_to([(NODE_X[i] + NODE_X[i + 1]) / 2, CHAIN_Y + 1.15, 0]))

        product = MathTex(r"\frac{\partial h_{60}}{\partial h_{1}}",
                          r"=\prod_{i=2}^{60}", r"\frac{\partial h_i}{\partial h_{i-1}}",
                          r"=\prod_{i=2}^{60}", r"W_h^{\top}",
                          r"\operatorname{diag}\!\left(1 - h_i^2\right)",
                          font_size=36).move_to([0, EQ1_Y, 0])
        product[4].set_color(BLUE)
        product[1].set_color(GOLD)
        product[3].set_color(GOLD)

        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.2),
                  Transform(forward, product), run_time=0.75)
        self.play(FadeIn(arc_labels), run_time=0.3)

        bound = MathTex(r"D_i = \operatorname{diag}\!\left(1 - h_i^2\right),",
                        r"\quad \tanh'(z) = 1 - \tanh^2(z) \le 1",
                        font_size=30, color=GREY).move_to([0, EQ2_Y, 0])
        note = Tex(r"every factor carries a $\tanh'$, so it shrinks by default: "
                   r"measured mean $%.2f$ here" % M['tanh_prime'],
                   font_size=26, color=GREY).move_to([0, EQ3_Y, 0])
        self.play(Transform(sub, bound), run_time=0.35)
        self.play(FadeIn(note), run_time=0.3)
        self.wait(0.49)

        # ---- watch it fade on the way back ---------------------------------
        # Radius and opacity follow the norms measured at the notebook's own
        # initialisation: 6.9e-02 leaving the loss, 1.5e-03 arriving at h_1.
        init_norms = M['norms'][1.0]
        self.play(FadeOut(note), run_time=0.3)

        def packet_at(i, norms, lo, hi, color=GOLD):
            """A dot whose size and opacity track log ||dJ/dh_t||."""
            n = norms[NODE_T[i]]
            frac = float(np.clip((np.log10(n) - lo) / (hi - lo), 0.0, 1.0))
            return Dot([NODE_X[i], CHAIN_Y, 0], radius=0.09 + 0.19 * frac,
                       color=color, fill_opacity=0.10 + 0.90 * frac)

        lo, hi = np.log10(init_norms[1]), np.log10(init_norms[60])

        def readout(i, norms):
            return MathTex(r"\left\|\frac{\partial J}{\partial h_t}\right\|_{t=%d} = %s"
                           % (NODE_T[i], sci(norms[NODE_T[i]])),
                           font_size=30, color=GOLD).move_to([0, EQ3_Y, 0])

        packet = Dot([LOSS_X, CHAIN_Y, 0], radius=0.28, color=GOLD, fill_opacity=1.0)
        meter = readout(4, init_norms)
        self.play(FadeIn(packet, scale=0.5), FadeIn(meter), run_time=0.3)
        for i in (3, 2, 1, 0):
            self.play(Transform(packet, packet_at(i, init_norms, lo, hi)),
                      Transform(meter, readout(i, init_norms)),
                      run_time=0.31)
        gone = Tex(r"the first bases get $%.0f\times$ less gradient than the last: "
                   r"whatever they hold, the weights barely learn it"
                   % (init_norms[60] / init_norms[0]),
                   font_size=26, color=GREY).move_to([0, CAP_Y, 0])
        self.play(FadeIn(gone), run_time=0.3)
        self.wait(0.52)

        # ---- one product, two regimes --------------------------------------
        title2 = Tex(r"One product, two regimes: $\;\|\partial h_{60} / \partial h_1\| "
                     r"\approx \gamma^{\,59}$", font_size=36).move_to([0, 3.5, 0])
        self.play(FadeOut(forward), FadeOut(sub), FadeOut(gone), FadeOut(meter),
                  FadeOut(packet), FadeOut(arc_labels), FadeOut(arcs),
                  Transform(title, title2), run_time=0.4)
        self.play(chain.animate.set_opacity(0.30),
                  VGroup(loss_box, loss_label, loss_arrow).animate.set_opacity(0.30),
                  run_time=0.3)

        axes = Axes(x_range=[0, 60, 20], y_range=[-12, 4, 4],
                    x_length=7.0, y_length=2.6,
                    axis_config={"include_tip": False, "stroke_width": 2,
                                 "color": GREY, "font_size": 20},
                    x_axis_config={"numbers_to_include": [0, 20, 40, 60],
                                   "decimal_number_config": {"num_decimal_places": 0}},
                    ).move_to([-1.7, -1.9, 0])
        y_ticks = VGroup(*[
            MathTex(rf"10^{{{v}}}", font_size=20, color=GREY)
            .next_to(axes.c2p(0, v), LEFT, buff=0.12)
            for v in (-12, -8, -4, 0, 4)
        ])
        x_title = Tex(r"timestep $t$ the backward pass has reached",
                      font_size=22, color=GREY).next_to(axes, DOWN, buff=0.18)
        y_title = MathTex(r"\left\|\partial J / \partial h_t\right\|",
                          font_size=24, color=GREY).next_to(axes.c2p(0, 4), UP, buff=0.10)
        self.play(Create(axes), FadeIn(y_ticks), FadeIn(x_title), FadeIn(y_title),
                  run_time=0.5)

        curve_colors = {1.6: RED, 1.0: BLUE, 0.7: "#1f4f8f"}
        curve_notes = {1.6: r"$\|W_h\| \times 1.6$", 1.0: r"as initialised",
                       0.7: r"$\|W_h\| \times 0.7$"}
        curves, curve_labels = VGroup(), VGroup()
        for scale in SCALES:
            norms = M['norms'][scale]
            # points ordered from t = 60 down to 0, so Create draws them the way
            # the backward pass walks: right to left
            points = [axes.c2p(t, np.log10(max(norms[t], 1e-13)))
                      for t in range(60, -1, -1)]
            curve = VMobject(stroke_color=curve_colors[scale], stroke_width=3)
            curve.set_points_as_corners(points)
            curves.add(curve)

            total = norms[0] / norms[60]
            label = VGroup(
                MathTex(r"\gamma = %.3f" % M['gamma'][scale], font_size=26,
                        color=curve_colors[scale]),
                Tex(curve_notes[scale] + r", $\;\times %s$ over 60 steps" % sci(total),
                    font_size=22, color=GREY),
            ).arrange(RIGHT, buff=0.22)
            label.next_to(points[-1], UP if scale != 1.6 else DOWN, buff=0.18)
            label.shift(RIGHT * 1.9)
            curve_labels.add(label)

        for curve, label in zip(curves, curve_labels):
            self.play(Create(curve), run_time=0.42)
            self.play(FadeIn(label), run_time=0.3)
        side = Tex(r"same cell, same batch,\\ only $\|W_h\|$ changes",
                   font_size=24, color=GREY).move_to([4.5, -1.35, 0])
        gamma_note = Tex(r"$\gamma$ is the measured\\ per-step factor",
                         font_size=24, color=GREY).move_to([4.5, -2.6, 0])
        self.play(FadeIn(side), FadeIn(gamma_note), run_time=0.3)
        self.wait(0.56)

        # ---- the LSTM's cell state is an additive path -----------------------
        title3 = Tex(r"The LSTM adds a path with no repeated matrix on it",
                     font_size=36).move_to([0, 3.5, 0])
        self.play(FadeOut(axes), FadeOut(y_ticks), FadeOut(x_title), FadeOut(y_title),
                  FadeOut(curves), FadeOut(curve_labels), FadeOut(side),
                  FadeOut(gamma_note), Transform(title, title3), run_time=0.4)
        self.play(chain.animate.set_opacity(1.0),
                  VGroup(loss_box, loss_label, loss_arrow).animate.set_opacity(1.0),
                  run_time=0.3)

        cell_line = Line([NODE_X[0] - 0.55, CELL_Y, 0], [NODE_X[4] + 0.55, CELL_Y, 0],
                         stroke_color=ORANGE, stroke_width=4)
        cell_dots = VGroup(*[Dot([x, CELL_Y, 0], radius=0.07, color=ORANGE)
                             for x in NODE_X])
        cell_tag = MathTex(r"c_t", font_size=28, color=ORANGE).move_to([-6.0, CELL_Y, 0])
        gate_marks = VGroup(*[
            MathTex(r"\odot f", font_size=22, color=ORANGE)
            .move_to([(NODE_X[i] + NODE_X[i + 1]) / 2, CELL_Y + 0.30, 0])
            for i in range(4)
        ])
        write_arrow = Arrow([NODE_X[1], CHAIN_Y + RADIUS, 0], [NODE_X[1], CELL_Y - 0.10, 0],
                            buff=0.03, stroke_width=1.8, color=ORANGE, tip_length=0.12)
        write_tag = MathTex(r"i \odot \tilde{c}", font_size=22,
                            color=ORANGE).next_to(write_arrow, LEFT, buff=0.08)
        read_arrow = Arrow([NODE_X[3], CELL_Y - 0.10, 0], [NODE_X[3], CHAIN_Y + RADIUS, 0],
                           buff=0.03, stroke_width=1.8, color=ORANGE, tip_length=0.12)
        read_tag = MathTex(r"o", font_size=22, color=ORANGE).next_to(read_arrow, RIGHT,
                                                                    buff=0.08)
        cell_path = VGroup(cell_line, cell_dots, cell_tag, gate_marks,
                           write_arrow, write_tag, read_arrow, read_tag)

        cell_eq = MathTex(r"c_t", r"=", r"f_t", r"\odot c_{t-1}", r"\;+\;",
                          r"i_t \odot \tilde{c}_t", r",\qquad h_t = o_t \odot \tanh(c_t)",
                          font_size=34).move_to([0, EQ1_Y, 0])
        for k in (0, 2, 3, 5):
            cell_eq[k].set_color(ORANGE)

        self.play(Create(cell_line), FadeIn(cell_dots), FadeIn(cell_tag), run_time=0.35)
        self.play(FadeIn(gate_marks), GrowArrow(write_arrow), FadeIn(write_tag),
                  GrowArrow(read_arrow), FadeIn(read_tag), run_time=0.3)
        self.play(FadeIn(cell_eq), run_time=0.3)
        self.wait(0.35)

        deriv = MathTex(r"\frac{\partial c_t}{\partial c_{t-1}} = \operatorname{diag}(f_t)",
                        r"\quad\Longrightarrow\quad",
                        r"\frac{\partial c_{60}}{\partial c_{1}} = "
                        r"\prod_{i=2}^{60} \operatorname{diag}(f_i)",
                        font_size=34).move_to([0, EQ2_Y, 0])
        deriv[0].set_color(ORANGE)
        deriv[2].set_color(ORANGE)
        additive = Tex(r"an element-wise multiply the gates control, "
                       r"not a repeated $W_h^{\top}$: "
                       r"hold $f$ near 1 and the path stays open",
                       font_size=26, color=GREY).move_to([0, EQ3_Y, 0])
        self.play(FadeIn(deriv), run_time=0.35)
        self.play(FadeIn(additive), run_time=0.3)
        self.wait(0.49)

        # ---- both trained, both measured ------------------------------------
        rnn_norms, lstm_norms = M['norms']['rnn'], M['norms']['lstm']
        self.play(FadeOut(cell_eq), FadeOut(deriv), FadeOut(additive), run_time=0.3)

        lo_r = min(np.log10(rnn_norms[t]) for t in NODE_T)
        hi_r = max(np.log10(rnn_norms[t]) for t in NODE_T)
        lo_l, hi_l = lo_r, hi_r        # the same scale, so the two are comparable

        def cell_packet(i):
            n = lstm_norms[NODE_T[i]]
            frac = float(np.clip((np.log10(n) - lo_l) / (hi_l - lo_l), 0.0, 1.0))
            return Dot([NODE_X[i], CELL_Y, 0], radius=0.10 + 0.18 * frac,
                       color=GOLD, fill_opacity=0.25 + 0.75 * frac)

        def hidden_packet(i):
            n = rnn_norms[NODE_T[i]]
            frac = float(np.clip((np.log10(n) - lo_r) / (hi_r - lo_r), 0.0, 1.0))
            return Dot([NODE_X[i], CHAIN_Y, 0], radius=0.10 + 0.18 * frac,
                       color=interpolate_color(ManimColor(GOLD), ManimColor(RED), frac),
                       fill_opacity=0.25 + 0.75 * frac)

        top = Dot([NODE_X[4], CELL_Y, 0], radius=0.10, color=GOLD, fill_opacity=0.3)
        bottom = Dot([NODE_X[4], CHAIN_Y, 0], radius=0.10, color=GOLD, fill_opacity=0.3)
        both = Tex(r"both trained, one batch, the notebook's settings",
                   font_size=26, color=GREY).move_to([0, EQ1_Y, 0])
        self.play(FadeIn(top), FadeIn(bottom), FadeIn(both), run_time=0.3)
        for i in (3, 2, 1, 0):
            self.play(Transform(top, cell_packet(i)), Transform(bottom, hidden_packet(i)),
                      run_time=0.3)

        verdict = VGroup(
            Tex(r"LSTM cell path: $\times %s$ over 60 steps, per-step $%.2f$"
                % (sci(lstm_norms[0] / lstm_norms[60]), M['gamma']['lstm']),
                font_size=27, color=ORANGE),
            Tex(r"RNN hidden path: $\times %s$, per-step $%.2f$ -- the same product, "
                r"drifting the other way"
                % (sci(rnn_norms[0] / rnn_norms[60]), M['gamma']['rnn']),
                font_size=27, color=BLUE),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT).move_to([0, EQ2_Y - 0.1, 0])
        self.play(FadeIn(verdict), run_time=0.3)
        self.wait(0.42)

        gates = Tex(r"$f$ keeps the old cell state, $i$ writes the proposal, "
                    r"$o$ reads it out --- here the trained gates sit at "
                    r"$\bar{f} = %.2f$" % M['forget'],
                    font_size=26, color=GREY).move_to([0, EQ3_Y - 0.15, 0])
        caveat = Tex(r"on this task the RNN still scores higher, 0.8918 to 0.8386: "
                     r"the splice motif sits mid-window,\\ about 30 steps back, "
                     r"which is inside the reach a plain RNN still has",
                     font_size=25, color=GREY).move_to([0, CAP_Y - 0.05, 0])
        self.play(FadeIn(gates), run_time=0.3)
        self.play(FadeIn(caveat), run_time=0.3)
        self.wait(0.84)


# ---------------------------------------------------------------------------
# Render to mp4, then convert with the repo's two-pass palette encoder. Manim's
# own --format=gif writes a per-frame palette and comes out several times
# larger for the same picture.
# ---------------------------------------------------------------------------
def _encoder_env():
    """PATH for latex + dvisvgm, plus a workaround for this machine's ffmpeg.

    Homebrew's ffmpeg 7.1_4 is linked against libx265.215 while the x265 keg on
    PATH now ships .216, so `ffmpeg` aborts on load. The 4.1 keg is still
    installed, so pointing DYLD_FALLBACK_LIBRARY_PATH at it fixes the load
    without touching the Homebrew tree. macOS strips DYLD_* variables when it
    execs a protected binary such as /bin/bash, which is why the encoder script
    is sourced by a shell that sets the variable itself rather than inheriting
    it (see the call below).
    """
    env = dict(os.environ)
    env["PATH"] = "/Library/TeX/texbin:/opt/homebrew/bin:" + env.get("PATH", "")
    return env


def main():
    script = str(Path(__file__).resolve())
    out_gif = REPO / "animations" / "gifs" / f"{NAME}.gif"
    env = _encoder_env()

    work = tempfile.mkdtemp(prefix=f"{NAME}_")
    subprocess.run(
        [sys.executable, "-m", "manim", "-qm", "--format=mp4", "--disable_caching",
         "-o", NAME, script, "BpttVanishing"],
        cwd=work, env=env, check=True)

    mp4 = Path(work) / "media" / "videos" / NAME / "720p30" / f"{NAME}.mp4"
    encoder = REPO / "animations" / "gif_from_mp4.sh"
    fallback = "/opt/homebrew/Cellar/x265/4.1/lib"
    if os.path.isdir(fallback):
        # source the encoder from a shell that exports the fallback itself
        cmd = ["bash", "-c",
               f'export DYLD_FALLBACK_LIBRARY_PATH="{fallback}"; '
               'source "$1" "$2" "$3" "$4" "$5"',
               "_", str(encoder), str(mp4), str(out_gif), "640", "13"]
    else:
        cmd = ["bash", str(encoder), str(mp4), str(out_gif), "640", "13"]
    subprocess.run(cmd, env=env, check=True)

    mb = out_gif.stat().st_size / 1e6
    print(f"wrote {out_gif}  {mb:.2f} MB")
    if mb > 1.5:
        raise SystemExit(f"OVER BUDGET: {mb:.2f} MB > 1.5 MB cap")


if __name__ == "__main__":
    main()
