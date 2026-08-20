"""Why a recommender factorises the rating matrix instead of filling it in.

The argument in one picture: R has 22 million cells and only 4.5% of them carry
a rating, but P and Q together hold 312 thousand numbers. Every prediction is
forced through that bottleneck, which is both why the model cannot memorise and
why the empty cells get values at all.

Needs LaTeX. Run from the repo root:
    python animations/manim_scenes/mf_factorization.py
"""
import os
import subprocess
import sys

from manim import (Scene, VGroup, Rectangle, Text, MathTex, Create, Write,
                   FadeIn, FadeOut, Transform, Indicate, config,
                   BLUE_D, ORANGE, GREEN_D, YELLOW_D, GREY_B, WHITE, LEFT, RIGHT, UP, DOWN)
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
NAME = 'mf_factorization'


class MFFactorization(Scene):
    def construct(self):
        np.random.seed(0)

        title = Text('A rating matrix is mostly holes', font_size=30)
        self.play(Write(title), run_time=0.6)
        self.play(title.animate.to_edge(UP, buff=0.35).scale(0.8), run_time=0.4)

        # --- the sparse matrix ---
        R = Rectangle(width=4.4, height=3.0, color=GREY_B, stroke_width=2)
        R.shift(LEFT * 3.6 + UP * 0.55)
        rlab = MathTex(r'R', font_size=34, color=WHITE).next_to(R, UP, buff=0.12)
        rdim = MathTex(r'6{,}040 \times 3{,}706', font_size=20, color=GREY_B).next_to(R, DOWN, buff=0.12)

        # scatter a few filled cells to show the sparsity
        dots = VGroup()
        for _ in range(90):
            x = np.random.uniform(-2.1, 2.1)
            y = np.random.uniform(-1.4, 1.4)
            c = Rectangle(width=0.11, height=0.11, fill_opacity=0.9,
                          fill_color=BLUE_D, stroke_width=0)
            c.move_to(R.get_center() + np.array([x, y, 0]))
            dots.add(c)

        self.play(Create(R), Write(rlab), run_time=0.5)
        self.play(FadeIn(dots, lag_ratio=0.01), Write(rdim), run_time=0.8)

        fill = MathTex(r'\text{only } 4.5\% \text{ filled}', font_size=22, color=BLUE_D)
        fill.next_to(rdim, DOWN, buff=0.15)
        self.play(Write(fill), run_time=0.4)
        self.wait(0.4)

        # --- the factorisation ---
        approx = MathTex(r'\approx', font_size=40).next_to(R, RIGHT, buff=0.4)
        P = Rectangle(width=0.95, height=3.0, color=ORANGE, stroke_width=2,
                      fill_opacity=0.22, fill_color=ORANGE).next_to(approx, RIGHT, buff=0.35)
        plab = MathTex(r'P', font_size=32, color=ORANGE).next_to(P, UP, buff=0.12)
        pdim = MathTex(r'6{,}040 \times 32', font_size=17, color=ORANGE).next_to(P, DOWN, buff=0.12)

        Qt = Rectangle(width=3.0, height=0.95, color=GREEN_D, stroke_width=2,
                       fill_opacity=0.22, fill_color=GREEN_D).next_to(P, RIGHT, buff=0.3)
        qlab = MathTex(r'Q^\top', font_size=32, color=GREEN_D).next_to(Qt, UP, buff=0.12)
        qdim = MathTex(r'32 \times 3{,}706', font_size=17, color=GREEN_D).next_to(Qt, DOWN, buff=0.12)

        self.play(Write(approx), run_time=0.3)
        self.play(Create(P), Write(plab), Write(pdim), run_time=0.5)
        self.play(Create(Qt), Write(qlab), Write(qdim), run_time=0.5)
        self.wait(0.5)

        # --- the count ---
        self.play(FadeOut(dots), FadeOut(fill),
                  FadeOut(rdim), FadeOut(pdim), FadeOut(qdim), run_time=0.3)
        count = VGroup(
            MathTex(r'6{,}040 \times 3{,}706 = 22{,}384{,}240 \ \text{cells}',
                    font_size=25, color=WHITE),
            MathTex(r'(6{,}040 + 3{,}706) \times 32 = 311{,}872 \ \text{numbers}',
                    font_size=25, color=YELLOW_D),
            MathTex(r'1.4\% \ \text{as many}', font_size=27, color=YELLOW_D),
        ).arrange(DOWN, buff=0.24).shift(DOWN * 2.35)
        self.play(Write(count[0]), run_time=0.5)
        self.play(Write(count[1]), run_time=0.5)
        self.play(Write(count[2]), run_time=0.4)
        self.play(Indicate(count[2], color=YELLOW_D, scale_factor=1.15), run_time=0.5)
        self.wait(0.5)

        # --- one prediction ---
        self.play(FadeOut(count), FadeOut(title), run_time=0.35)
        pred_title = Text('One prediction is one row against one column',
                          font_size=25).to_edge(UP, buff=0.35)
        self.play(Write(pred_title), run_time=0.5)

        prow = Rectangle(width=0.95, height=0.16, fill_opacity=0.95, fill_color=ORANGE,
                         stroke_width=0).move_to(P.get_center() + UP * 0.7)
        qcol = Rectangle(width=0.16, height=0.95, fill_opacity=0.95, fill_color=GREEN_D,
                         stroke_width=0).move_to(Qt.get_center() + RIGHT * 0.6)
        cell = Rectangle(width=0.16, height=0.16, fill_opacity=0.95, fill_color=YELLOW_D,
                         stroke_width=0)
        cell.move_to(R.get_center() + np.array([0.6, 0.7, 0]))

        self.play(FadeIn(prow), FadeIn(qcol), run_time=0.4)
        self.play(FadeIn(cell), run_time=0.3)

        eq = MathTex(r'\hat{r}_{ui}', r'=', r'\mu + b_u + b_i', r'+',
                     r'p_u^\top q_i', font_size=30).shift(DOWN * 2.6)
        eq[0].set_color(YELLOW_D)
        eq[4].set_color(WHITE)
        self.play(Write(eq), run_time=0.7)
        self.wait(0.6)

        # --- the payoff ---
        self.play(FadeOut(eq), FadeOut(prow), FadeOut(qcol), FadeOut(cell),
                  FadeOut(pred_title), run_time=0.35)

        final = VGroup(
            Text('Every cell gets a value, including the 95.5% never rated.',
                 font_size=21, color=WHITE),
            Text('312k numbers cannot memorise 22M cells - so it has to generalise.',
                 font_size=21, color=YELLOW_D),
        ).arrange(DOWN, buff=0.26).shift(DOWN * 2.6)
        self.play(Write(final[0]), run_time=0.6)
        self.play(Write(final[1]), run_time=0.6)

        allfill = VGroup()
        for a in range(-19, 20, 2):
            for b in range(-13, 14, 2):
                c = Rectangle(width=0.1, height=0.1, fill_opacity=0.55,
                              fill_color=BLUE_D, stroke_width=0)
                c.move_to(R.get_center() + np.array([a * 0.11, b * 0.11, 0]))
                allfill.add(c)
        self.play(FadeIn(allfill, lag_ratio=0.002), run_time=1.2)
        self.wait(1.0)


def main():
    build = os.path.join(REPO, 'animations', '_manim_media')
    subprocess.run([sys.executable, '-m', 'manim', '-qm', '--format=mp4',
                    '--media_dir', build, '-o', NAME, __file__, 'MFFactorization'],
                   check=True, cwd=REPO)
    mp4 = os.path.join(build, 'videos', NAME, '720p30', f'{NAME}.mp4')
    gif = os.path.join(REPO, 'animations', 'gifs', f'{NAME}.gif')
    subprocess.run(['bash', os.path.join(REPO, 'animations', 'gif_from_mp4.sh'),
                    mp4, gif, '640', '13'], check=True)
    print('size: %.2f MB' % (os.path.getsize(gif) / 1e6))


if __name__ == '__main__':
    main()
