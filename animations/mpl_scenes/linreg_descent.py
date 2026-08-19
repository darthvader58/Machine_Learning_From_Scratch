"""Gradient descent on the weight-only linear regression from
`regression/linear_regression.ipynb`, shown as one walk in two places at once.

Left  : the training cars with the current line y_hat = w x + b drawn over them.
Right : the loss J(w, b) as filled contours, with the parameter trajectory and an
        arrow along the current negative gradient.

The data, the standardisation, the learning rate (alpha = 0.1) and the
initialisation (w = 0, b = 0) are the notebook's, so the last frame lands on the
coefficients the notebook prints: w = -6.3305, b = 23.2955, J = 9.7929.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpl_style import COLORS, GREY, FIGSIZE, DPI, save_gif, plt   # shared look
from matplotlib import animation
from ucimlrepo import fetch_ucirepo

np.random.seed(42)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'gifs', 'linreg_descent.gif')


# ----------------------------------------------------------------- the model
class Linear_Regression:
    """Copied from the notebook. The only addition is `self.path`, which stores
    (w, b, dw, db) at the top of every iteration so the animation can replay the
    real steps instead of tweening between them."""

    def __init__(self, alpha=0.01, n_iter=1000):
        self.alpha = alpha
        self.n_iter = n_iter

    def fit(self, X, y):
        n, D = X.shape
        self.w = np.zeros(D)
        self.b = 0.0
        self.history = []
        self.path = []

        for i in range(self.n_iter):
            y_hat = X.dot(self.w) + self.b
            error = y_hat - y

            cost = np.sum(error ** 2) / (2 * n)
            self.history.append(cost)
            if not np.isfinite(cost):
                break

            # dJ/dw = (1/n) X^T (y_hat - y),  dJ/db = (1/n) sum(y_hat - y)
            dw = X.T.dot(error) / n
            db = np.sum(error) / n

            self.path.append((self.w[0], self.b, dw[0], db))   # added for the animation

            self.w = self.w - self.alpha * dw
            self.b = self.b - self.alpha * db

        self.history = np.array(self.history)
        self.path = np.array(self.path)
        return self

    def predict(self, X):
        return X.dot(self.w) + self.b


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# ------------------------------------------------------------------- the data
# Auto MPG, exactly as the notebook loads it: join features and target, then drop
# the 6 rows whose horsepower is missing so features and target stay aligned.
auto = fetch_ucirepo(id=9)
feature_names = list(auto.data.features.columns)
X_all = auto.data.features.values.astype(float)
y_all = auto.data.targets['mpg'].values.astype(float)

complete = ~np.isnan(np.hstack([X_all, y_all.reshape(-1, 1)])).any(axis=1)
X_all, y_all = X_all[complete], y_all[complete]

# the notebook's split: an independently seeded shuffle, first 80% for training
rng = np.random.RandomState(0)
order = rng.permutation(len(X_all))
n_train = int(0.8 * len(X_all))
train_idx = order[:n_train]

w_col = feature_names.index('weight')
x_train_w = X_all[train_idx][:, [w_col]]
y_train = y_all[train_idx]

# standardise the single column, using training statistics only
mu_w = x_train_w.mean()
sigma_w = x_train_w.std()
x_std = (x_train_w - mu_w) / sigma_w

model = Linear_Regression(alpha=0.1, n_iter=2000).fit(x_std, y_train)
path = model.path
alpha = model.alpha
n = len(y_train)
x_flat = x_std.ravel()

print('final: w = %.4f  b = %.4f  J = %.4f  R2 = %.4f'
      % (model.w[0], model.b, model.history[-1], r_squared(y_train, model.predict(x_std))))


# ------------------------------------------------------- the loss surface grid
# J(w, b) = (1/2n) sum_i (w x_i + b - y_i)^2 evaluated over a grid of parameters.
W_LO, W_HI = -15.5, 8.5
B_LO, B_HI = -4.0, 28.0
gw = np.linspace(W_LO, W_HI, 140)
gb = np.linspace(B_LO, B_HI, 140)
GW, GB = np.meshgrid(gw, gb)
resid = GW[:, :, None] * x_flat[None, None, :] + GB[:, :, None] - y_train[None, None, :]
J_grid = np.sum(resid ** 2, axis=2) / (2 * n)
del resid

# Because each column was standardised, (1/n) X^T X is the identity here, so the
# contours are exact circles about the minimum. Levels evenly spaced in radius
# keep the rings evenly spaced on screen.
J_min = model.history[-1]
radii = np.linspace(0, np.sqrt(2 * (J_grid.max() - J_min)), 13)
levels = J_min + 0.5 * radii ** 2


# ------------------------------------------------------------------- the frame
# real iterations, sampled finely at the start where the steps are large
iters = list(range(0, 26)) + [27, 29, 31, 34, 38, 43, 50, 60, 80, 120, 250, 700, 1999]
HOLD = 7
frames = iters + [iters[-1]] * HOLD

fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
ax_fit = fig.add_axes([0.085, 0.135, 0.40, 0.66])
ax_loss = fig.add_axes([0.615, 0.135, 0.345, 0.66])

# the governing equations, in the notebook's 1/(2n) convention
fig.text(0.5, 0.955,
         r'$J(w,b)=\frac{1}{2n}\sum_i(\hat{y}_i-y_i)^2$'
         r'$,\qquad \hat{y}_i = w\,x_i + b$',
         ha='center', va='center', fontsize=11, color='#3b3b3b')
fig.text(0.5, 0.865,
         r'$w \leftarrow w-\alpha\frac{\partial J}{\partial w}$'
         r'$,\quad \frac{\partial J}{\partial w}=\frac{1}{n}\sum_i(\hat{y}_i-y_i)\,x_i$'
         r'$,\quad \alpha=0.1$',
         ha='center', va='center', fontsize=11, color='#3b3b3b')
fig.text(0.5, 0.028,
         'J is convex in (w, b): one bowl, one minimum, so the walk cannot get stuck.',
         ha='center', va='center', fontsize=9, color=GREY)

# --- left: data and the current line
ax_fit.scatter(x_flat, y_train, s=9, color=COLORS[0], alpha=0.55,
               linewidths=0, label='training cars', zorder=2)
(fit_line,) = ax_fit.plot([], [], color=COLORS[1], linewidth=2.4,
                          label=r'$\hat{y}=w\,x+b$', zorder=3)
ax_fit.set_xlim(-2.0, 2.9)
ax_fit.set_ylim(-3.5, 50)
ax_fit.set_xlabel('weight (standardised)', fontsize=9)
ax_fit.set_ylabel('mpg', fontsize=9)
ax_fit.tick_params(labelsize=8)
ax_fit.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
fit_text = ax_fit.text(0.04, 0.06, '', transform=ax_fit.transAxes, fontsize=9,
                       va='bottom', ha='left', color='#2b2b2b',
                       bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=GREY, alpha=0.85))

# --- right: the loss surface
ax_loss.contourf(GW, GB, J_grid, levels=levels, cmap='Blues', extend='max', zorder=1)
ax_loss.contour(GW, GB, J_grid, levels=levels, colors='white',
                linewidths=0.5, alpha=0.6, zorder=2)
ax_loss.plot(model.w[0], model.b, marker='*', markersize=11, color='#ffe08a',
             markeredgecolor='#4a3a00', markeredgewidth=0.6, zorder=5)
ax_loss.text(model.w[0] - 1.0, model.b - 1.9, 'minimum', fontsize=8.5, color='#2b2b2b',
             ha='right', va='top')
(trail,) = ax_loss.plot([], [], color=COLORS[1], linewidth=1.4, marker='o',
                        markersize=2.6, zorder=6)
(here,) = ax_loss.plot([], [], marker='o', markersize=6.5, color=COLORS[1],
                       markeredgecolor='white', markeredgewidth=0.9, zorder=7)
arrow = ax_loss.annotate('', xy=(0, 0), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='-|>', color=COLORS[2],
                                         linewidth=2.0, mutation_scale=13), zorder=8)
arrow_label = ax_loss.text(0, 0, '', fontsize=10, color=COLORS[2], zorder=9,
                           ha='left', va='center')
ax_loss.set_xlim(W_LO, W_HI)
ax_loss.set_ylim(B_LO, B_HI)
ax_loss.set_aspect('equal', adjustable='box')
ax_loss.set_xlabel('w', fontsize=9)
ax_loss.set_ylabel('b', fontsize=9)
ax_loss.tick_params(labelsize=8)
ax_loss.grid(False)
loss_text = ax_loss.text(0.04, 0.04, '', transform=ax_loss.transAxes, fontsize=9,
                         va='bottom', ha='left', color='#2b2b2b',
                         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=GREY, alpha=0.85))

line_x = np.array([-2.0, 2.9])
g0 = np.hypot(path[0, 2], path[0, 3])


def draw(k):
    """Draw the state of the fit at iteration k of the real gradient descent run."""
    w, b, dw, db = path[k]

    # left: the line these parameters give, and how much variance it explains
    fit_line.set_data(line_x, w * line_x + b)
    r2 = r_squared(y_train, x_flat * w + b)
    fit_text.set_text('iteration %d\n$R^2$ = %.3f' % (k, r2))

    # right: where the parameters have been, where they are, and where the
    # negative gradient points from here
    trail.set_data(path[:k + 1, 0], path[:k + 1, 1])
    here.set_data([w], [b])

    # the arrow is drawn along -grad J with a length proportional to its size, so
    # it shrinks away as the walk settles: at the minimum the gradient is zero
    g = np.hypot(dw, db)
    length = min(5.5, 0.45 * g)
    ux, uy = -dw / g, -db / g
    arrow.set_position((w, b))
    arrow.xy = (w + length * ux, b + length * uy)
    arrow_label.set_position((w + length * ux + 0.6, b + length * uy))
    arrow_label.set_text(r'$-\nabla J$' if length > 0.9 else '')

    loss_text.set_text('w = %+.4f\nb = %+.4f\nJ = %.3f' % (w, b, model.history[k]))
    return fit_line, fit_text, trail, here, arrow, arrow_label, loss_text


anim = animation.FuncAnimation(fig, draw, frames=frames, interval=90, blit=False)
save_gif(anim, OUT, fps=12)
print('frames:', len(frames))
