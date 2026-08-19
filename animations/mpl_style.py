"""Shared look for every matplotlib animation.

The notebooks use `style.use('ggplot')` and a fixed five-colour cycle, so the
animations inherit both. Keeping this in one place means all 11 matplotlib
GIFs read as one set rather than eleven separate efforts.
"""
import matplotlib
matplotlib.use('Agg')          # no display needed; we only ever write files
import matplotlib.pyplot as plt
from matplotlib import style, animation

style.use('ggplot')

COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#4a3aa7']
GREY = '#7f7f7f'

# 640px wide at 100 dpi keeps GIFs legible on GitHub without blowing the budget.
FIGSIZE = (6.4, 4.4)
DPI = 100
FPS = 12


def save_gif(anim, out_path, fps=FPS):
    """Write a FuncAnimation to disk and report its size."""
    import os
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    mb = os.path.getsize(out_path) / 1e6
    print(f"wrote {out_path}  {mb:.2f} MB")
    if mb > 1.5:
        raise SystemExit(f"OVER BUDGET: {mb:.2f} MB > 1.5 MB cap")
    return mb
