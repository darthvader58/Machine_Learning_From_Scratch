"""Backpropagation is the chain rule applied to a composition.

Companion animation for `neural_networks/mlp.ipynb`.

The notebook's network is [64, 64, 32, 10], which has 6,570 weights and far too
many edges to draw. This script trains the notebook's own `Neural_Network`
class - copied in verbatim below - on a 3-4-4-3 architecture so every unit and
every edge is visible, then replays one real forward and backward pass through
it frame by frame.

The three inputs are the leading three principal components of the same UCI
handwritten digits the notebook uses (numpy SVD, no sklearn), restricted to the
digits 0, 1 and 7 so the softmax has three outputs. Every number on screen -
the activations, the softmax probabilities, delta at each layer, the ReLU gates
and the weight gradient - comes out of that network's own forward/backward, not
out of a script that draws what backpropagation is supposed to look like.

Run from the repo root with no arguments to regenerate the GIF:

    /opt/anaconda3/envs/tf_mps/bin/python animations/manim_scenes/backprop_chain.py
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from manim import *

REPO = Path(__file__).resolve().parents[2]
NAME = "backprop_chain"

# --- shared look -----------------------------------------------------------
# The matplotlib animations use ggplot on white with a fixed five-colour cycle;
# this scene keeps the same palette so the set reads as one thing.
INK = "#22242a"
PAPER = "#ffffff"
BLUE = "#2a78d6"     # forward signal: activations
ORANGE = "#eb6834"   # backward signal: the error delta
GREEN = "#1baf7a"    # a ReLU gate that is open
RED = "#c0392b"      # a ReLU gate that is shut
GREY = "#9aa0aa"

config.background_color = PAPER


# ===========================================================================
# The notebook's code, unchanged.
# ===========================================================================
def softmax(scores):
    # subtract the row maximum first so exp cannot overflow
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y):
    return -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()


class Neural_Network:
    def __init__(self, layer_sizes, learning_rate=0.1, activation=True, seed=0):
        r = np.random.default_rng(seed)

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        self.weights = []
        self.biases = []
        self.history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He initialisation: sqrt(2 / fan_in)
            self.weights.append(r.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out))

    def forward(self, X):
        self.activations = [X]
        self.zs = []
        last = len(self.weights) - 1
        a = X

        for i in range(len(self.weights)):
            z = a @ self.weights[i] + self.biases[i]
            self.zs.append(z)

            if i == last:
                a = softmax(z)
            elif self.activation:
                a = np.maximum(0, z)
            else:
                a = z

            self.activations.append(a)

        return a

    def backward(self, probs, y):
        n = len(y)

        # softmax and cross-entropy differentiated together: p - y
        delta = probs.copy()
        delta[np.arange(n), y] -= 1
        delta /= n

        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        for i in range(len(self.weights) - 1, -1, -1):
            grads_w[i] = self.activations[i].T @ delta
            grads_b[i] = delta.sum(axis=0)

            if i > 0:
                delta = delta @ self.weights[i].T
                if self.activation:
                    delta = delta * (self.zs[i - 1] > 0)

        return grads_w, grads_b

    def step(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def score(self, X, y):
        probs = self.forward(X)
        return cross_entropy(probs, y), (probs.argmax(axis=1) == y).mean()

    def fit(self, X, y, X_test, y_test, epochs=40, batch_size=32, seed=0):
        for epoch in range(epochs):
            batch_order = np.random.default_rng(seed + epoch).permutation(len(X))

            for s in range(0, len(batch_order), batch_size):
                batch = batch_order[s:s + batch_size]
                probs = self.forward(X[batch])
                grads_w, grads_b = self.backward(probs, y[batch])
                self.step(grads_w, grads_b)

            tl, ta = self.score(X, y)
            vl, va = self.score(X_test, y_test)
            self.history['train_loss'].append(tl)
            self.history['test_loss'].append(vl)
            self.history['train_acc'].append(ta)
            self.history['test_acc'].append(va)

        return self


# ===========================================================================
# One real pass through a network small enough to draw.
# ===========================================================================
DIGITS = (0, 1, 7)      # the three classes the tiny softmax chooses between
SAMPLE = 1402           # index into the shuffled order; picked because one ReLU
                        # unit in each hidden layer is off for it, which is the
                        # point the gate beat needs to make


def run_network():
    """Train the notebook's class on 3 features, then record one sample's pass."""
    from ucimlrepo import fetch_ucirepo

    digits = fetch_ucirepo(id=80)
    X_all = digits.data.features.values.astype(float) / 16.0
    y_all = digits.data.targets.values.ravel()

    # keep three digit classes so the output layer has three units
    mask = np.isin(y_all, DIGITS)
    labels = np.searchsorted(np.array(DIGITS), y_all[mask])
    X = X_all[mask]

    # three inputs instead of 64: the leading principal components of the pixels,
    # rescaled to [0, 1] the way the notebook rescales raw pixel values
    centred = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    F = centred @ Vt[:3].T
    F = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))

    rng = np.random.default_rng(0)
    order = rng.permutation(len(F))
    split = int(0.8 * len(F))
    train_idx, test_idx = order[:split], order[split:]

    net = Neural_Network([3, 4, 4, 3], learning_rate=0.1, seed=2)
    net.fit(F[train_idx], labels[train_idx], F[test_idx], labels[test_idx],
            epochs=2, batch_size=32, seed=0)

    # one sample, one forward pass; batch of size 1 so delta is not averaged
    i = SAMPLE
    x = F[i:i + 1]
    label = int(labels[i])
    probs = net.forward(x)[0]

    # the backward recurrence, unrolled so each delta can be drawn on its own
    d3 = probs.copy()
    d3[label] -= 1                                        # delta^(L) = p - y
    d2 = (d3 @ net.weights[2].T) * (net.zs[1][0] > 0)     # (W delta) o g'(z)
    d1 = (d2 @ net.weights[1].T) * (net.zs[0][0] > 0)

    return dict(
        a0=net.activations[0][0], a1=net.activations[1][0], a2=net.activations[2][0],
        z1=net.zs[0][0], z2=net.zs[1][0],
        p=probs, y=np.eye(3)[label], label=label,
        d3=d3, d2=d2, d1=d1,
        gate1=(net.zs[0][0] > 0).astype(int), gate2=(net.zs[1][0] > 0).astype(int),
        gw1=np.outer(net.activations[0][0], d1),
        acc=net.history['test_acc'][-1],
    )


def vec(values, decimals=2):
    """A row vector of numbers, formatted the way the notebook prints them."""
    inner = ",\\ ".join(f"{v:.{decimals}f}" for v in values)
    return f"({inner})"


class BackpropChain(Scene):
    def construct(self):
        MathTex.set_default(color=INK)
        Tex.set_default(color=INK)
        Text.set_default(color=INK)

        S = run_network()

        # ---- geometry -----------------------------------------------------
        xs = [-5.5, -2.9, -0.3, 2.3]
        sizes = [3, 4, 4, 3]
        radius = 0.19
        centre_y = 1.15

        def ys(n):
            # index 0 at the top, evenly spaced about centre_y
            gap = 0.85 if n == 3 else 0.85
            return [centre_y + (n - 1 - 2 * k) * gap / 2 for k in range(n)]

        layers = []
        for x, n in zip(xs, sizes):
            col = VGroup(*[
                Circle(radius=radius, stroke_color=INK, stroke_width=1.6,
                       fill_color=BLUE, fill_opacity=0.0).move_to([x, y, 0])
                for y in ys(n)
            ])
            layers.append(col)

        edges = []      # edges[l] holds the lines from layer l to layer l+1
        for l in range(3):
            group = VGroup()
            for src in layers[l]:
                for dst in layers[l + 1]:
                    group.add(Line(src.get_center(), dst.get_center(),
                                   stroke_color=GREY, stroke_width=1.1,
                                   stroke_opacity=0.75)
                              .set_length(np.linalg.norm(
                                  dst.get_center() - src.get_center()) - 2 * radius))
            edges.append(group)

        net_group = VGroup(*edges, *layers)

        captions = VGroup(
            MathTex(r"a^{(0)}", font_size=30).move_to([xs[0], -0.75, 0]),
            MathTex(r"a^{(1)}", font_size=30).move_to([xs[1], -0.75, 0]),
            MathTex(r"a^{(2)}", font_size=30).move_to([xs[2], -0.75, 0]),
            MathTex(r"p", font_size=30).move_to([xs[3], -0.75, 0]),
        )
        sublabels = VGroup(
            Tex("input", font_size=24, color=GREY).move_to([xs[0], -1.15, 0]),
            Tex("ReLU", font_size=24, color=GREY).move_to([xs[1], -1.15, 0]),
            Tex("ReLU", font_size=24, color=GREY).move_to([xs[2], -1.15, 0]),
            Tex("softmax", font_size=24, color=GREY).move_to([xs[3], -1.15, 0]),
        )

        title = Tex(r"Backpropagation is the chain rule, run once", font_size=38)
        title.move_to([0, 3.35, 0])

        # ---- opening ------------------------------------------------------
        self.play(Write(title), run_time=0.35)
        self.play(LaggedStart(*[Create(g) for g in edges], lag_ratio=0.25, run_time=0.45),
                  LaggedStart(*[FadeIn(c) for c in layers], lag_ratio=0.25, run_time=0.45))
        self.play(FadeIn(captions), FadeIn(sublabels), run_time=0.3)
        self.wait(0.25)

        # ---- 1. the forward pass -------------------------------------------
        fwd_eq = MathTex(r"z^{(l)} = W^{(l)\top} a^{(l-1)} + b^{(l)}",
                         r"\qquad",
                         r"a^{(l)} = \max\!\left(0,\, z^{(l)}\right)",
                         font_size=36).move_to([0, -1.85, 0])
        fwd_eq[0].set_color(BLUE)
        self.play(FadeIn(fwd_eq), run_time=0.3)

        numbers = MathTex(r"a^{(0)} = " + vec(S['a0']), font_size=34, color=BLUE)
        numbers.move_to([0, -2.75, 0])

        def fill_layer(col, values):
            top = max(values.max(), 1e-9)
            return [c.animate.set_fill(BLUE, opacity=float(0.85 * v / top))
                    for c, v in zip(col, values)]

        self.play(*fill_layer(layers[0], S['a0']), FadeIn(numbers), run_time=0.3)
        self.wait(0.25)

        for l, (acts, name) in enumerate([(S['a1'], "a^{(1)}"), (S['a2'], "a^{(2)}")]):
            flashes = [ShowPassingFlash(e.copy().set_stroke(BLUE, width=3.5, opacity=1),
                                        time_width=0.6) for e in edges[l]]
            self.play(LaggedStart(*flashes, lag_ratio=0.02), run_time=0.33)
            new_numbers = MathTex(name + " = " + vec(acts), font_size=34,
                                  color=BLUE).move_to([0, -2.75, 0])
            self.play(*fill_layer(layers[l + 1], acts),
                      Transform(numbers, new_numbers), run_time=0.3)
            self.wait(0.25)

        # last layer: the softmax, whose outputs get their own column
        flashes = [ShowPassingFlash(e.copy().set_stroke(BLUE, width=3.5, opacity=1),
                                    time_width=0.6) for e in edges[2]]
        self.play(LaggedStart(*flashes, lag_ratio=0.02), run_time=0.33)

        out_y = ys(3)
        digit_labels = VGroup(*[
            MathTex(str(d), font_size=24,
                    color=PAPER if pv > 0.45 else INK).move_to([xs[3], y, 0])
            for d, y, pv in zip(DIGITS, out_y, S['p'])
        ])
        p_head = MathTex(r"p", font_size=28, color=BLUE).move_to([3.75, 2.55, 0])
        p_col = VGroup(*[
            MathTex(f"{v:.2f}", font_size=30, color=BLUE).move_to([3.75, y, 0])
            for v, y in zip(S['p'], out_y)
        ])
        new_numbers = MathTex(r"p = \mathrm{softmax}\left(z^{(3)}\right) = " + vec(S['p']),
                              font_size=34, color=BLUE).move_to([0, -2.75, 0])
        self.play(*fill_layer(layers[3], S['p']),
                  FadeIn(digit_labels), FadeIn(p_head), FadeIn(p_col),
                  Transform(numbers, new_numbers), run_time=0.3)
        self.wait(0.35)

        # ---- 2. the output error, and why it is this simple ----------------
        y_head = MathTex(r"y", font_size=28, color=INK).move_to([4.75, 2.55, 0])
        y_col = VGroup(*[
            MathTex(f"{v:.0f}", font_size=30,
                    color=INK if v else GREY).move_to([4.75, yy, 0])
            for v, yy in zip(S['y'], out_y)
        ])
        self.play(FadeOut(fwd_eq), FadeOut(numbers), run_time=0.3)
        self.play(FadeIn(y_head), FadeIn(y_col), run_time=0.3)

        loss_eq = MathTex(r"J = -\sum_k y_k \log p_k", font_size=34)
        loss_eq.move_to([0, -1.8, 0])
        self.play(FadeIn(loss_eq), run_time=0.3)
        self.wait(0.25)

        chain_eq = MathTex(r"\frac{\partial J}{\partial z_j}", "=", r"-\sum_k y_k",
                           r"\frac{1}{p_k}", r"\cdot", r"p_k",
                           r"\left(\delta_{kj} - p_j\right)", font_size=34)
        chain_eq.move_to([0, -2.65, 0])
        note = Tex(r"cross-entropy $\times$ softmax derivative", font_size=24, color=GREY)
        note.next_to(chain_eq, DOWN, buff=0.18)
        self.play(FadeIn(chain_eq), FadeIn(note), run_time=0.3)
        self.wait(0.35)

        # the 1/p_k from the loss and the p_k from the softmax cancel
        strikes = VGroup(*[
            Line(part.get_corner(DL) + (DOWN + LEFT) * 0.07,
                 part.get_corner(UR) + (UP + RIGHT) * 0.07,
                 stroke_color=RED, stroke_width=3)
            for part in (chain_eq[3], chain_eq[5])
        ])
        self.play(Indicate(chain_eq[3], color=RED, scale_factor=1.15),
                  Indicate(chain_eq[5], color=RED, scale_factor=1.15), run_time=0.3)
        self.play(Create(strikes), run_time=0.3)
        self.wait(0.25)

        result = MathTex(r"= -y_j + p_j \sum_k y_k = p_j - y_j", font_size=34)
        result.move_to([0, -3.45, 0])
        self.play(FadeOut(note), FadeIn(result), run_time=0.3)
        self.wait(0.32)

        delta_box = VGroup(
            MathTex(r"\delta^{(L)} = p - y", font_size=36, color=ORANGE)
        )
        delta_box.move_to([4.4, -1.95, 0])
        frame = SurroundingRectangle(delta_box, color=ORANGE, buff=0.18,
                                     stroke_width=2)
        d_head = MathTex(r"\delta^{(L)}", font_size=28, color=ORANGE).move_to([5.95, 2.55, 0])
        d_col = VGroup(*[
            MathTex(f"{v:+.2f}", font_size=30, color=ORANGE).move_to([5.95, yy, 0])
            for v, yy in zip(S['d3'], out_y)
        ])
        halos3 = VGroup(*[
            Circle(radius=radius + 0.09, stroke_color=ORANGE,
                   stroke_width=1.5 + 6 * abs(v) / np.abs(S['d3']).max(),
                   fill_opacity=0).move_to(c.get_center())
            for c, v in zip(layers[3], S['d3'])
        ])
        self.play(FadeIn(delta_box), Create(frame),
                  FadeIn(d_head), FadeIn(d_col), Create(halos3), run_time=0.3)
        self.wait(0.42)

        band = VGroup(loss_eq, chain_eq, strikes, result)
        self.play(FadeOut(band), FadeOut(delta_box), FadeOut(frame), run_time=0.3)

        # ---- 3. the backward recurrence ------------------------------------
        rec = MathTex(r"\delta^{(l)}", "=", r"\left(W^{(l+1)} \delta^{(l+1)}\right)",
                      r"\odot", r"g'\!\left(z^{(l)}\right)", font_size=38)
        rec.move_to([-0.6, -1.75, 0])
        rec[0].set_color(ORANGE)
        rec[2].set_color(ORANGE)
        rec[4].set_color(GREEN)

        back_note = Tex(r"the same edges, backwards: \texttt{delta @ W.T}",
                        font_size=25, color=GREY)
        back_note.move_to([-3.9, -2.9, 0])
        gate_note = Tex(r"1 if the unit fired, 0 if it did not",
                        font_size=25, color=GREEN)
        gate_note.move_to([3.3, -2.9, 0])
        arrows = VGroup(
            Arrow(back_note.get_top(), rec[2].get_bottom(), buff=0.08,
                  stroke_width=2, max_tip_length_to_length_ratio=0.12, color=GREY),
            Arrow(gate_note.get_top(), rec[4].get_bottom(), buff=0.08,
                  stroke_width=2, max_tip_length_to_length_ratio=0.12, color=GREEN),
        )
        self.play(FadeIn(rec), run_time=0.3)
        self.play(FadeIn(back_note), FadeIn(gate_note), Create(arrows), run_time=0.3)
        self.wait(0.42)

        halos = {3: halos3}
        for l, (deltas, gates, zs_vals) in enumerate(
                [(S['d2'], S['gate2'], S['z2']), (S['d1'], S['gate1'], S['z1'])]):
            src_layer = 2 - l          # index of the layer receiving the error
            edge_group = edges[src_layer]

            # the error travels back along exactly the edges it came forward on
            flashes = [
                ShowPassingFlash(
                    Line(e.get_end(), e.get_start()).set_stroke(ORANGE, width=3.5),
                    time_width=0.6)
                for e in edge_group
            ]
            self.play(LaggedStart(*flashes, lag_ratio=0.02), run_time=0.35)

            # then the ReLU gate decides which units are allowed to keep it
            gate_labels = VGroup()
            for c, g in zip(layers[src_layer], gates):
                spot = c.get_center() + np.array([0.42, 0.0, 0.0])
                badge = Circle(radius=0.155, stroke_width=0,
                               fill_color=PAPER, fill_opacity=1).move_to(spot)
                lab = MathTex(str(int(g)), font_size=26,
                              color=GREEN if g else RED).move_to(spot)
                gate_labels.add(badge, lab)
            gate_vec = MathTex(r"g'\!\left(z^{(%d)}\right) = " % (2 - l)
                               + "(" + ",\\ ".join(str(int(g)) for g in gates) + ")",
                               font_size=32, color=GREEN)
            gate_vec.move_to([0, -3.6, 0])
            if l == 0:
                self.play(FadeIn(gate_labels), FadeIn(gate_vec), run_time=0.3)
            else:
                self.play(FadeIn(gate_labels), Transform(prev_gate_vec, gate_vec),
                          run_time=0.3)
            if l == 0:
                prev_gate_vec = gate_vec
            self.wait(0.25)

            top = max(np.abs(deltas).max(), 1e-9)
            new_halos = VGroup()
            blocked = VGroup()
            for c, v, g in zip(layers[src_layer], deltas, gates):
                if g:
                    new_halos.add(Circle(radius=radius + 0.09, stroke_color=ORANGE,
                                         stroke_width=1.5 + 6 * abs(v) / top,
                                         fill_opacity=0).move_to(c.get_center()))
                else:
                    cross = VGroup(
                        Line(c.get_center() + np.array([-0.16, -0.16, 0]),
                             c.get_center() + np.array([0.16, 0.16, 0]),
                             stroke_color=RED, stroke_width=3),
                        Line(c.get_center() + np.array([-0.16, 0.16, 0]),
                             c.get_center() + np.array([0.16, -0.16, 0]),
                             stroke_color=RED, stroke_width=3),
                    )
                    blocked.add(cross)
            self.play(Create(new_halos), Create(blocked), run_time=0.3)
            halos[src_layer] = new_halos
            self.wait(0.32)

        self.wait(0.25)
        self.play(FadeOut(rec), FadeOut(back_note), FadeOut(gate_note),
                  FadeOut(arrows), FadeOut(prev_gate_vec), run_time=0.3)

        # ---- 4. the parameter gradients ------------------------------------
        grad_eq = MathTex(r"\frac{\partial J}{\partial W^{(l)}}", "=",
                          r"a^{(l-1)}", r"\delta^{(l)\top}", font_size=38)
        grad_eq.move_to([-2.6, -2.0, 0])
        grad_eq[2].set_color(BLUE)
        grad_eq[3].set_color(ORANGE)
        outer = Tex(r"an outer product: the gradient for one weight is the error at its "
                    r"destination\\ times the activation at its source",
                    font_size=26, color=GREY)
        outer.move_to([0, -3.05, 0])
        self.play(FadeIn(grad_eq), FadeIn(outer), run_time=0.3)

        # one concrete edge: input unit 3 into the first hidden unit
        src = layers[0][2]
        dst = layers[1][0]
        edge = Line(src.get_center(), dst.get_center(), stroke_color=INK, stroke_width=4)
        one_weight = MathTex(
            r"\frac{\partial J}{\partial W^{(1)}_{31}} = "
            + f"{S['a0'][2]:.2f}" + r"\times" + f"{S['d1'][0]:.2f}" + " = "
            + f"{S['gw1'][2, 0]:.2f}", font_size=34)
        one_weight.move_to([2.7, -2.0, 0])
        one_weight.set_color_by_tex("times", INK)
        self.play(Create(edge), Indicate(src, color=BLUE, scale_factor=1.2),
                  Indicate(dst, color=ORANGE, scale_factor=1.2), run_time=0.3)
        self.play(FadeIn(one_weight), run_time=0.3)
        self.wait(0.56)

        # ---- 5. why one sweep is enough -------------------------------------
        self.play(FadeOut(grad_eq), FadeOut(outer), FadeOut(one_weight),
                  FadeOut(edge), run_time=0.3)

        fan_in = VGroup(*[
            Line(s.get_center(), dst.get_center(), stroke_color=ORANGE, stroke_width=3)
            for s in layers[0]
        ])
        closing = Tex(r"Each $\delta$ is computed once and reused by every weight feeding "
                      r"that unit,\\ so one backward sweep costs about what one forward "
                      r"sweep costs.", font_size=30)
        closing.move_to([0, -2.3, 0])
        self.play(LaggedStart(*[ShowPassingFlash(f.copy(), time_width=0.7) for f in fan_in],
                              lag_ratio=0.12),
                  FadeIn(closing), run_time=0.5)
        self.wait(0.56)

        # ---- 6. and what the ReLU is holding apart --------------------------
        self.play(FadeOut(closing), run_time=0.3)
        collapse = MathTex(r"W^{(3)\top}\!\left(W^{(2)\top}\!\left(W^{(1)\top}x + b_1\right)"
                           r"+ b_2\right) + b_3 \;=\; W'^{\top} x + b'", font_size=34)
        collapse.move_to([0, -2.15, 0])
        collapse_note = Tex(r"take the ReLUs out and the notebook's 6{,}570 weights "
                            r"multiply into 650:\\ the same probabilities to "
                            r"$3 \times 10^{-15}$, and depth buys nothing",
                            font_size=27, color=GREY)
        collapse_note.move_to([0, -3.15, 0])
        self.play(FadeIn(collapse), run_time=0.3)
        self.play(FadeIn(collapse_note), run_time=0.3)
        self.wait(0.77)


# ---------------------------------------------------------------------------
# Render to mp4, then convert with the repo's two-pass palette encoder. Manim's
# own --format=gif writes a per-frame palette and comes out several times
# larger for the same picture.
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    scratch = REPO / "animations" / ".manim_build"
    scratch.mkdir(exist_ok=True)

    env = dict(os.environ)
    env["PATH"] = "/Library/TeX/texbin:/opt/homebrew/bin:" + env.get("PATH", "")

    subprocess.run(
        [sys.executable, "-m", "manim", "-qm", "--format=mp4", "--disable_caching",
         "-o", NAME, str(Path(__file__).resolve()), "BackpropChain"],
        cwd=scratch, env=env, check=True)

    mp4 = scratch / "media" / "videos" / NAME / "720p30" / f"{NAME}.mp4"
    gif = REPO / "animations" / "gifs" / f"{NAME}.gif"

    # Homebrew's ffmpeg is linked against whichever x265 was installed when it
    # was built, and an x265 upgrade leaves it looking for a dylib that is only
    # in the older keg. Point dyld at every keg that is still on disk so the
    # right one is found. macOS strips DYLD_* variables when it launches the
    # SIP-protected /bin/bash, and again on any re-exec of it, so the variable
    # is set inside that shell and the encoder is sourced into the same shell
    # rather than run as a child. It is a no-op where ffmpeg already works.
    prelude = ('for d in /opt/homebrew/Cellar/x265/*/lib; do '
               'DYLD_FALLBACK_LIBRARY_PATH="$d:$DYLD_FALLBACK_LIBRARY_PATH"; done; '
               'export DYLD_FALLBACK_LIBRARY_PATH="'
               '$DYLD_FALLBACK_LIBRARY_PATH/usr/local/lib:/usr/lib"; '
               'source "$0" "$@"')
    subprocess.run(["bash", "-c", prelude, str(REPO / "animations" / "gif_from_mp4.sh"),
                    str(mp4), str(gif), "640", "13"], env=env, check=True)

    mb = gif.stat().st_size / 1e6
    print(f"wrote {gif}  {mb:.2f} MB")
    if mb > 1.5:
        raise SystemExit(f"OVER BUDGET: {mb:.2f} MB > 1.5 MB cap")
