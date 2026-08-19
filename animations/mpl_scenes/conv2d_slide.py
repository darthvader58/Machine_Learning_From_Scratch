"""Convolution as a sliding dot product - and why it is not a dense layer.

Three panels, matching the way `2d_cnn.ipynb` builds the idea up:

  left    a real CIFAR-10 photograph with the 3x3 window sliding over it in
          row-major order
  middle  the patch sitting under the window right now, the kernel beside it,
          the nine products, and the single number they sum to
  right   the feature map filling in one pixel at a time

The image is `X_col_train[26]` from the notebook's CIFAR subset (seed 1) and the
kernel is the notebook's hand-made `vertical` edge filter, so the feature map is
a real edge response rather than noise.

The point the animation is making: the nine weights in the middle never change.
The same kernel is applied at all 900 positions, which is why the layer costs 9
weights and a bias instead of the 921,600 a dense layer would need.

Run from the repo root:
    /opt/anaconda3/envs/tf_mps/bin/python animations/mpl_scenes/conv2d_slide.py
"""
import os
import pickle
import sys

import numpy as np
import matplotlib
from matplotlib import animation
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # animations/ holds mpl_style
from mpl_style import plt, COLORS, GREY, DPI, save_gif  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
BATCH_DIR = os.path.join(REPO, 'data', 'cifar-10', 'cifar-10-batches-py')
OUT = os.path.join(REPO, 'animations', 'gifs', 'conv2d_slide.gif')

np.random.seed(0)


# ----------------------------------------------------------------------------
# the notebook's own forward pass, copied verbatim from 2d_cnn.ipynb
# ----------------------------------------------------------------------------
def conv_forward(X, kernels, bias):
    n = len(X)
    f, kh, kw = kernels.shape
    out_h = X.shape[1] - kh + 1
    out_w = X.shape[2] - kw + 1

    out = np.zeros((n, out_h, out_w, f))
    flat_kernels = kernels.reshape(f, kh * kw)

    for i in range(out_h):
        for j in range(out_w):
            # the patch under the filter, taken from every image at once
            region = X[:, i:i + kh, j:j + kw].reshape(n, kh * kw)
            out[:, i, j, :] = region @ flat_kernels.T + bias

    return out


# ----------------------------------------------------------------------------
# the data: the same CIFAR-10 subset the notebook builds in section 2
# ----------------------------------------------------------------------------
def load_batch(name):
    with open(os.path.join(BATCH_DIR, name), 'rb') as f:
        return pickle.load(f, encoding='bytes')


train_batches = [load_batch(f'data_batch_{i}') for i in range(1, 6)]
X_cifar_all = np.concatenate([b[b'data'] for b in train_batches]).reshape(-1, 3, 32, 32)
y_cifar_all = np.concatenate([np.array(b[b'labels']) for b in train_batches])

# the notebook keeps airplane, automobile, bird and cat, 750 training images each
cifar_classes = [0, 1, 2, 3]
r = np.random.default_rng(1)
chosen = [r.permutation(np.where(y_cifar_all == c)[0])[:750] for c in cifar_classes]
chosen = r.permutation(np.concatenate(chosen))
X_col_train = X_cifar_all[chosen].astype(float) / 255.0

# image 26 of that subset is a dark car on a light background: a clean edge test.
# a filter in the notebook reads all three colour planes at once; to keep the
# arithmetic on screen readable this animates one plane, the green one.
image = X_col_train[26, 1]

# the notebook's hand-made vertical edge filter: bright on the left, dark on the
# right gives a large positive response
kernel = np.array([[1., 0., -1.],
                   [1., 0., -1.],
                   [1., 0., -1.]])
bias = 0.0

# one call to the notebook's function gives every output the animation reveals
feature_map = conv_forward(image[None], kernel[None], np.array([bias]))[0, :, :, 0]
OUT_H, OUT_W = feature_map.shape                    # 30 x 30
N_POS = OUT_H * OUT_W                               # 900 window positions
LIMIT = np.abs(feature_map).max()

positions = [(i, j) for i in range(OUT_H) for j in range(OUT_W)]

# parameter counts for the caption
conv_params = kernel.size + 1                        # 9 weights + 1 bias
dense_params = image.size * feature_map.size         # 1024 inputs -> 900 outputs


# ----------------------------------------------------------------------------
# frame schedule: step through the first positions one at a time so the pattern
# is unmistakable, then fast-forward in blocks so all 900 still get covered
# ----------------------------------------------------------------------------
SLOW = 16
BLOCK = 24
frame_idx = list(range(SLOW))
k = SLOW
while k < N_POS:
    k = min(N_POS, k + BLOCK)
    frame_idx.append(k - 1)
frame_idx += [N_POS - 1] * 6                         # hold so the loop does not snap


# ----------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(7.6, 4.4), dpi=DPI)
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.3, 1.0],
                      left=0.03, right=0.985, top=0.80, bottom=0.13, wspace=0.18)
ax_img = fig.add_subplot(gs[0])
ax_calc = fig.add_subplot(gs[1])
ax_map = fig.add_subplot(gs[2])

fig.suptitle('One kernel, slid over the image: $O_{ij} = \\sum_m \\sum_n '
             'K_{mn}\\,X_{i+m,\\,j+n} + b$', fontsize=11.5, y=0.95)

# --- left: the photograph with the window on it ---
ax_img.imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
ax_img.set_title('CIFAR-10 automobile\n$X$, 32x32 (green plane)', fontsize=9.5)
ax_img.set_xticks([]); ax_img.set_yticks([]); ax_img.grid(False)
window = Rectangle((-0.5, -0.5), 3, 3, fill=False, lw=2.2, ec=COLORS[1], zorder=5)
ax_img.add_patch(window)

# --- right: the feature map, revealed one output pixel at a time ---
cmap = matplotlib.colormaps['gray_r'].copy()
cmap.set_bad('#dcd8d5')                              # not computed yet
shown = np.full_like(feature_map, np.nan)
map_art = ax_map.imshow(np.ma.masked_invalid(shown), cmap=cmap,
                        vmin=-LIMIT, vmax=LIMIT, interpolation='nearest')
ax_map.set_title('feature map $O$\n30x30, one pixel per position', fontsize=9.5)
ax_map.set_xticks([]); ax_map.set_yticks([]); ax_map.grid(False)
map_dot = Rectangle((-0.5, -0.5), 1, 1, fill=False, lw=1.8, ec=COLORS[1], zorder=5)
ax_map.add_patch(map_dot)

# --- middle: the arithmetic for the current position ---
ax_calc.set_xlim(0, 10); ax_calc.set_ylim(0, 11)
ax_calc.set_aspect('equal')
ax_calc.axis('off')
ax_calc.grid(False)

CELL = 1.15


def build_grid(x0, y_top, fs=9):
    """Nine boxes with a number in each; returns the rectangles and the texts."""
    rects, texts = [], []
    for a in range(3):
        for b in range(3):
            x = x0 + b * CELL
            y = y_top - (a + 1) * CELL
            rect = Rectangle((x, y), CELL, CELL, facecolor='white',
                             edgecolor=GREY, lw=0.8)
            ax_calc.add_patch(rect)
            txt = ax_calc.text(x + CELL / 2, y + CELL / 2, '', ha='center',
                               va='center', fontsize=fs)
            rects.append(rect); texts.append(txt)
    return rects, texts


PATCH_X, KERN_X, TOP = 0.35, 5.75, 9.6
patch_rects, patch_texts = build_grid(PATCH_X, TOP)
kern_rects, kern_texts = build_grid(KERN_X, TOP)
prod_rects, prod_texts = build_grid(PATCH_X, 4.9)

ax_calc.text(PATCH_X + 1.5 * CELL, TOP + 0.35, 'patch under window',
             ha='center', fontsize=9, color=GREY)
ax_calc.text(KERN_X + 1.5 * CELL, TOP + 0.35, 'kernel $K$ (fixed)',
             ha='center', fontsize=9, color=COLORS[0])
ax_calc.text((PATCH_X + 3 * CELL + KERN_X) / 2, TOP - 1.5 * CELL, r'$\times$',
             ha='center', va='center', fontsize=13)
ax_calc.text(PATCH_X + 1.5 * CELL, 5.15, 'elementwise products',
             ha='center', fontsize=9, color=GREY)

# the right half carries the one number the nine products add up to
ax_calc.text(7.1, 4.35, 'add the nine up:', ha='center', fontsize=9, color=GREY)
ax_calc.text(7.1, 1.9, 'the same $K$ is used\nat every position',
             ha='center', va='center', fontsize=9, color=COLORS[0])

# the kernel numbers are written once and never touched again - that is the point
for cell, value in zip(kern_texts, kernel.ravel()):
    cell.set_text(f'{value:+.0f}')
for rect, value in zip(kern_rects, kernel.ravel()):
    rect.set_facecolor({1.0: '#cfe0f5', 0.0: 'white', -1.0: '#f8d9c8'}[value])

sum_text = ax_calc.text(7.1, 3.35, '', ha='center', va='center', fontsize=10.5)
pos_text = ax_calc.text(5.0, 10.75, '', ha='center', va='center', fontsize=9.5,
                        color=COLORS[1])

fig.text(0.5, 0.045,
         f'The same {kernel.size} weights are reused at all {N_POS} positions: '
         f'{conv_params} parameters in total.\n'
         f'A dense layer from {image.size} pixels to {feature_map.size} outputs '
         f'would need {dense_params:,} weights.',
         ha='center', fontsize=9.5, color='#333333')


def update(idx):
    """Draw the state after window position `idx` has been evaluated."""
    i, j = positions[idx]

    # left panel: move the window
    window.set_xy((j - 0.5, i - 0.5))

    # middle panel: this patch, times the same kernel, summed
    patch = image[i:i + 3, j:j + 3]
    products = patch * kernel
    for rect, txt, value in zip(patch_rects, patch_texts, patch.ravel()):
        txt.set_text(f'{value:.2f}')
        rect.set_facecolor(str(1.0 - 0.75 * value))   # grey box, same as the pixel
        txt.set_color('white' if value > 0.55 else 'black')
    for rect, txt, value in zip(prod_rects, prod_texts, products.ravel()):
        txt.set_text(f'{value:+.2f}' if value else '0')
        rect.set_facecolor('#cfe0f5' if value > 0 else
                           ('#f8d9c8' if value < 0 else 'white'))

    total = products.sum() + bias
    sum_text.set_text(f'$O_{{{i},{j}}} = {total:+.2f}$')
    stage = 'stepping' if idx < SLOW else 'fast-forward'
    pos_text.set_text(f'position {idx + 1} of {N_POS}   ({stage})')

    # right panel: reveal every output computed so far
    flat = shown.ravel()
    flat[:idx + 1] = feature_map.ravel()[:idx + 1]
    map_art.set_array(np.ma.masked_invalid(shown))
    map_dot.set_xy((j - 0.5, i - 0.5))

    return []


anim = animation.FuncAnimation(fig, update, frames=frame_idx, interval=1000 / 12,
                               blit=False)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_gif(anim, OUT, fps=12)
print('frames:', len(frame_idx))
