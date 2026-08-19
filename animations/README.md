# Animations

One animation per notebook, each showing the single mathematical idea that matters most for
that algorithm. Every one replays the notebook's own from-scratch NumPy code - the classes are
copied verbatim, with only a snapshot list added to record state per iteration - so what the
animation shows is the algorithm the notebook actually runs, not a scripted approximation.

## Layout

```
mpl_scenes/     matplotlib FuncAnimation builders - algorithm dynamics on the real data
manim_scenes/   Manim Scene subclasses - symbolic and geometric derivations
gifs/           the rendered artifacts, embedded in the notebooks and shown in the root README
mpl_style.py    shared colours, figure size and the size-checked GIF writer
gif_from_mp4.sh two-pass ffmpeg palettegen/paletteuse encoder
inject.py       inserts a GIF into a notebook without re-running the notebook
render_all.sh   rebuilds every GIF from source
```

## Rebuilding

```bash
bash animations/render_all.sh
```

Each script is standalone and seeded, so this reproduces the committed GIFs byte for byte.
A single one can be rebuilt directly:

```bash
/opt/anaconda3/envs/tf_mps/bin/python animations/mpl_scenes/kmeans_lloyd.py
```

## Why the GIFs live on disk rather than only in the notebooks

`IPython.display.Image(filename=...)` embeds the file's raw bytes base64 under an `image/gif`
MIME key - the same output mechanism that already carries the notebooks' static figures, so it
renders anywhere those render. The copy on disk and the copy inside the `.ipynb` are the same
bytes, which is what lets the root README show them on GitHub without a second render.

Base64 costs 4/3 of the file size, so a 600 KB GIF adds about 800 KB to a notebook. The
budget is 1.5 MB per GIF, enforced by `save_gif` in `mpl_style.py`.

## Injecting into a notebook

`inject.py` starts a kernel, runs **only** the new cell, and splices its output in. It never
re-executes the notebook - several of these take from minutes to an hour to run, and re-running
them risks landing on different numbers than the README reports.

```bash
export JUPYTER_PATH=<dir containing a working kernelspec>
python animations/inject.py <notebook> <gif-name> <anchor-heading> <<'TXT'
## Markdown heading for the cell that introduces the animation
TXT
```

Both inserted cells are tagged `metadata.animation = <gif-name>`, so re-running replaces them
in place instead of stacking duplicates.

## Manim prerequisites

The Manim scenes typeset with LaTeX, which is not part of the Python install:

```bash
brew install --cask basictex
sudo /Library/TeX/texbin/tlmgr update --self
sudo /Library/TeX/texbin/tlmgr install standalone preview doublestroke ms rsfs relsize \
  fundus-calligra wasy wasysym physics mathastext cbfonts-fd dvisvgm
pip install manim==0.19.2
```

`standalone.cls` is the one to check for - `kpsewhich standalone.cls` should print a path.
The matplotlib scenes need none of this; they use matplotlib's own mathtext.
