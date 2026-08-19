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
