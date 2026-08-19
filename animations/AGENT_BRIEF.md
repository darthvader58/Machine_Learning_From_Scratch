# Brief for animation-authoring agents

You are writing ONE animation for ONE notebook in this from-scratch ML repo.

## Environment
- Python: `/opt/anaconda3/envs/tf_mps/bin/python` (3.11, arm64). Use this interpreter ONLY.
- Repo root: `/Users/shashwatraj/Machine_Learning_From_Scratch`
- ffmpeg is on PATH. `animations/gif_from_mp4.sh in.mp4 out.gif [width] [fps]` is the encoder.
- Shared matplotlib look: `animations/mpl_style.py` (imports `COLORS`, `FIGSIZE`, `DPI`,
  `save_gif`). Import it; do not restyle.

## Hard rules
1. **Allowed imports: `numpy`, `matplotlib`, `manim`, stdlib, and `ucimlrepo` for data.**
   NO `sklearn`, `torch`, `tensorflow`, `scipy`, `seaborn`. A repo-wide grep enforces this.
2. **The animation must show the REAL algorithm.** Reuse the notebook's own from-scratch class
   verbatim where practical - copy it into your script rather than reimplementing it
   differently. If the notebook's class records iteration state (several store `self.history`),
   replay that. Do not fake the dynamics with a scripted tween.
3. **Size: 1.5 MB HARD CAP per GIF, ~600 KB target.** `save_gif` raises if you exceed it.
   Levers: fewer frames, lower fps, smaller figsize, fewer plotted points.
4. **Do NOT touch any `.ipynb` file.** Notebook injection is done centrally, later. You produce
   a script and a GIF, nothing else.
5. **Do NOT run any git command.**
6. Set every random seed. The GIF must be byte-reproducible across runs.

## Deliverables (exactly two files)
- `animations/mpl_scenes/<name>.py`  OR  `animations/manim_scenes/<name>.py`
- `animations/gifs/<name>.gif`

`<name>` is given to you. The script must be standalone and re-runnable:
`/opt/anaconda3/envs/tf_mps/bin/python animations/mpl_scenes/<name>.py` regenerates the GIF
from the repo root, with no arguments.

## Quality bar
- Legible at 640px wide. Font sizes >= 9pt. Do not crowd the frame.
- Include the governing equation on the figure via matplotlib mathtext (`$...$`) or Manim
  `MathTex`. The point of these is *mathematical* explanation.
- 3-8 seconds of motion. End on a readable final state; do not cut mid-transition.
- Add a brief hold (repeat the last frame ~6 times) so the loop does not snap.
- Comment the script the way the notebooks comment code: plain, explaining what the step does.

## Datasets
Datasets already used by the notebooks are cached. Check what the target notebook loads and use
the same data so the animation matches the numbers the notebook already reports. `data/` holds
cifar-10 and modelnet10. UCI sets come via `fetch_ucirepo(id=...)` and are cached.

## When done
Print: the script path, the GIF path, its size in MB, frame count, and one sentence on what the
animation shows. If you could not hit the size budget or the concept did not work, say so
plainly rather than shipping something misleading.

---

# Manim addendum

Manim 0.19.2 is installed and verified working. LaTeX lives in a **user tree**, so your
environment must carry both paths:

```bash
export PATH="/Library/TeX/texbin:/opt/homebrew/bin:$PATH"
```

`/Library/TeX/texbin` gives `latex`; `/opt/homebrew/bin` gives `dvisvgm`. Without the second,
Manim raises `FileNotFoundError: 'dvisvgm'` only when it first tries to typeset - the scene
builds fine up to that point, so the failure looks unrelated. The packages are installed under
`~/Library/texmf` via `tlmgr --usermode`, which is already on kpathsea's search path.

## Rendering to a GIF

Do NOT use Manim's `--format=gif`; it writes a naive per-frame palette and the file comes out
several times larger than it needs to be. Render mp4, then convert:

```bash
export PATH="/Library/TeX/texbin:/opt/homebrew/bin:$PATH"
cd <your scratch dir>
/opt/anaconda3/envs/tf_mps/bin/python -m manim -qm --format=mp4 -o <name> scene.py <SceneClass>
bash /Users/shashwatraj/Machine_Learning_From_Scratch/animations/gif_from_mp4.sh \
  media/videos/scene/720p30/<name>.mp4 \
  /Users/shashwatraj/Machine_Learning_From_Scratch/animations/gifs/<name>.gif 640 15
```

`-qm` is 720p30. The helper does a two-pass palettegen/paletteuse downscale to 640px at 15 fps,
prints the final size, and is what keeps these inside the 1.5 MB cap.

## Structure your script so it renders itself

Your deliverable `animations/manim_scenes/<name>.py` must regenerate the GIF when run with no
arguments from the repo root, same contract as the matplotlib scenes. Put the Manim invocation
and the ffmpeg conversion in a `if __name__ == '__main__':` block that shells out, so
`render_all.sh` picks it up.

## Manim style notes

- Default frame is 14.22 x 8 units, origin at centre. Keep content inside roughly x in [-6.5, 6.5]
  and y in [-3.5, 3.5] or it will be cropped at 640px.
- `MathTex` for equations, `Tex` for prose, `Text` for plain labels. `MathTex` takes LaTeX, so
  `\tfrac` and friends are fine here - unlike matplotlib mathtext.
- Break equations into separate `MathTex` substrings when you want to `Transform`, `Indicate` or
  recolour one part. Transforming a whole equation into another whole equation reads as a cut;
  transforming matched sub-terms is what makes the algebra look inevitable.
- Colour the same symbol the same way every time it appears. That consistency is most of what
  makes this style readable.
- 5-10 seconds. Use `self.wait()` generously - viewers need time to read an equation.
- Seed any randomness so the render is reproducible.
