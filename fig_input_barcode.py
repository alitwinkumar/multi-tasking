"""Phase diagrams for rule inputs along barcode A, barcode B, and their union."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

output_dir = Path(__file__).resolve().parent / 'plots' / Path(__file__).stem
output_dir.mkdir(parents=True, exist_ok=True)


def savefig(name, **kwargs):
    fig = plt.gcf()
    stem = Path(name).stem
    fig.savefig(output_dir / (stem + '.pdf'), **kwargs)

from scipy.special import ndtr, ndtri
from contourpy import contour_generator
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

theta = 1.0
alpha_A = 1.4
alpha_B = 1.9
s_values = np.linspace(0, 3, 601)
f_values = np.linspace(0, 1, 501)

contrast = 1.0 - 1.0 / alpha_A
f_A_boundary = alpha_A / alpha_B
f_B_at_threshold = contrast + 1.0 / alpha_B
f_B_at_infinite_input = 2.0 * (contrast + 1.0 / alpha_B - 0.5)

def s_B_boundary(f):
    """Exact B-transverse-stability boundary for threshold-linear units."""
    g_s = 1.0 - contrast / f
    g_0 = (1.0 / alpha_B - f + contrast) / (1.0 - f)
    return theta * (1.0 - ndtri(g_s) / ndtri(g_0))

def f_B_existence(s):
    """Exact minimization formula for existence of a nonzero B-only state."""
    if s <= theta:
        return np.nan
    t = np.geomspace(1e-4, 1e4, 20000)
    g_0 = ndtr(-t)
    g_s = ndtr((s / theta - 1.0) * t)
    required_f = (1.0 / alpha_B - g_0) / (g_s - g_0)
    return np.min(required_f)

t_curve = np.geomspace(1e-3, 1e3, 1600)
g_0_curve = ndtr(-t_curve)
g_s_curve = (
    1.0 / alpha_B - g_0_curve / alpha_A
) / (contrast + 1.0 / alpha_B - g_0_curve)
f_curve = contrast / (1.0 - g_s_curve)
s_curve = 1.0 + ndtri(g_s_curve) / t_curve
visible = (s_curve >= s_values[0] / theta) & (s_curve <= s_values[-1] / theta)

x = np.geomspace(1e-3, 200.0, 1100)[:, None]
y = np.geomspace(1e-3, 200.0, 1100)[None, :]
gain_A_only = ndtr(x)
gain_B_only = ndtr(-y)
inverse_gap = (1.0 / alpha_A - 1.0 / alpha_B) / (
    gain_A_only - gain_B_only
)
f_map = 1.0 - inverse_gap
common_gain = (1.0 / alpha_A - inverse_gap * gain_A_only) / f_map
with np.errstate(invalid="ignore", divide="ignore"):
    z = ndtri(np.clip(common_gain, 1e-14, 1.0 - 1e-14))
    delta_map = x * z / (y * np.sqrt(x * x - z * z))
    s_map = 1.0 + delta_map
valid_map = (
    (f_map > 0.0)
    & (f_map < 1.0)
    & (common_gain > 0.5)
    & (common_gain < gain_A_only)
    & np.isfinite(s_map)
    & (s_map >= 1.0)
    & (s_map <= s_values[-1] / theta)
)

normal_pdf = lambda value: np.exp(-0.5 * value * value) / np.sqrt(2.0 * np.pi)
with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
    r_A_map = theta * delta_map / x
    r_B_map = theta / y
    radius_map = np.hypot(r_A_map, r_B_map)
    dg_s_A = -(s_map - 1.0) * theta * normal_pdf(x) / r_A_map**2
    dg_s_R = -(s_map - 1.0) * theta * normal_pdf(z) / radius_map**2
    dg_0_B = theta * normal_pdf(y) / r_B_map**2
    ratio_A = r_A_map / radius_map
    ratio_B = r_B_map / radius_map

    dh_A_dA = (1.0 - f_map) * dg_s_A + f_map * dg_s_R * ratio_A
    dh_A_dB = f_map * dg_s_R * ratio_B
    dh_B_dA = f_map * dg_s_R * ratio_A
    dh_B_dB = (1.0 - f_map) * dg_0_B + f_map * dg_s_R * ratio_B

    jac_11 = alpha_A * r_A_map * dh_A_dA
    jac_12 = alpha_A * r_A_map * dh_A_dB
    jac_21 = alpha_B * r_B_map * dh_B_dA
    jac_22 = alpha_B * r_B_map * dh_B_dB
    jac_trace = jac_11 + jac_22
    jac_det = jac_11 * jac_22 - jac_12 * jac_21

stable_simultaneous_map = valid_map & (jac_trace < 0.0) & (jac_det > 0.0)

det_contours = contour_generator(
    np.log(x[:, 0]),
    np.log(y[0, :]),
    np.where(valid_map, jac_det, np.nan).T,
    name="mpl2014",
    corner_mask=True,
)
fold_curves = []
for vertices in det_contours.lines(0.0)[0]:
    x_fold = np.exp(vertices[:, 0])
    y_fold = np.exp(vertices[:, 1])
    gain_A_fold = ndtr(x_fold)
    gain_B_fold = ndtr(-y_fold)
    inverse_gap_fold = (1.0 / alpha_A - 1.0 / alpha_B) / (
        gain_A_fold - gain_B_fold
    )
    f_fold = 1.0 - inverse_gap_fold
    common_fold = (
        1.0 / alpha_A - inverse_gap_fold * gain_A_fold
    ) / f_fold
    z_fold = ndtri(np.clip(common_fold, 1e-14, 1.0 - 1e-14))
    with np.errstate(invalid="ignore", divide="ignore"):
        s_fold = 1.0 + x_fold * z_fold / (
            y_fold * np.sqrt(x_fold * x_fold - z_fold * z_fold)
        )
    physical = (
        np.isfinite(s_fold)
        & (s_fold >= 1.0)
        & (s_fold <= s_values[-1] / theta)
        & (f_fold > 0.0)
        & (f_fold < 1.0)
    )
    if (
        np.count_nonzero(physical) > 100
        and np.ptp(s_fold[physical]) > 0.2
        and np.ptp(f_fold[physical]) > 0.005
    ):
        fold_curves.append((s_fold[physical], f_fold[physical]))

active = s_values[None, :] > 1.0
a_stable = active & (f_values[:, None] < f_A_boundary)

order = np.argsort(s_curve)
f_B_boundary = np.interp(
    s_values,
    s_curve[order],
    f_curve[order],
)
b_stable = active & (f_values[:, None] > f_B_boundary[None, :])

simultaneous_stable = np.zeros((len(f_values), len(s_values)), dtype=bool)
stable_s = s_map[stable_simultaneous_map]
stable_f = np.broadcast_to(f_map, s_map.shape)[
    stable_simultaneous_map
]

s_step = s_values[1] - s_values[0]
f_step = f_values[1] - f_values[0]

for column, s in enumerate(s_values):
    if s <= 1.0:
        continue

    nearby = np.sort(
        stable_f[np.abs(stable_s - s) <= max(1.5 * s_step, 0.0075)]
    )

    if len(nearby) < 2:
        continue

    gaps = np.flatnonzero(np.diff(nearby) > 2.5 * f_step)
    starts = np.r_[0, gaps + 1]
    stops = np.r_[gaps, len(nearby) - 1]

    for start, stop in zip(starts, stops):
        if stop - start < 1:
            continue

        lower = nearby[start] - 0.55 * f_step
        upper = nearby[stop] + 0.55 * f_step
        simultaneous_stable[:, column] |= (
            (f_values >= lower) & (f_values <= upper)
        )

phase = np.zeros((len(f_values), len(s_values)), dtype=int)

phase[active & a_stable & ~b_stable & ~simultaneous_stable] = 1
phase[active & ~a_stable & b_stable & ~simultaneous_stable] = 2
phase[active & ~a_stable & ~b_stable & simultaneous_stable] = 3
phase[active & a_stable & b_stable] = 4
phase[active & a_stable & ~b_stable & simultaneous_stable] = 5
phase[active & ~a_stable & b_stable & simultaneous_stable] = 6

phase[active & (phase == 0)] = 3

# Omit the barely resolved B/A+B coexistence strip from the manuscript plot.
# This is a display simplification: retain the calculated classification here.
# The separate parameter-search script shows all stable-attractor combinations.
phase_full = phase.copy()
phase[phase == 6] = 2

colors = [
    "#e8e8e8",
    "#315a9e",
    "#b23a48",
    "#e5b849",
    "#8064a2",
    "#5aa9d6",
    "#e07a5f",
]

labels = [
    "silent",
    r"$A$",
    r"$B$",
    r"$A{+}B$",
    r"$A$, $B$",
    r"$A$, $A{+}B$",
    r"$B$, $A{+}B$",
]

def phase_at(s, f):
    s = np.nan_to_num(s, nan=s_values[0])
    f = np.nan_to_num(f, nan=f_values[0])
    s_index = np.clip(
        np.rint((s - s_values[0]) / s_step).astype(int),
        0,
        len(s_values) - 1,
    )
    f_index = np.clip(
        np.rint((f - f_values[0]) / f_step).astype(int),
        0,
        len(f_values) - 1,
    )
    return phase[f_index, s_index]

def plot_phase_boundary(ax, s, f, delta_f=0.0045):
    s = np.asarray(s)
    f = np.asarray(f)
    valid = (
        np.isfinite(s)
        & np.isfinite(f)
        & (s >= 1.0)
        & (s <= 3.0)
        & (f >= 0.0)
        & (f <= 1.0)
    )

    lower = phase_at(s, np.clip(f - delta_f, 0.0, 1.0))
    upper = phase_at(s, np.clip(f + delta_f, 0.0, 1.0))
    valid &= lower != upper

    ax.plot(
        np.where(valid, s, np.nan),
        np.where(valid, f, np.nan),
        color="black",
        lw=0.7,
        ls="-",
    )

def plot_fold_next_to_multistability(ax, s, f):
    """Keep fold segments bordering A/A+B or B/A+B phases only."""
    s = np.asarray(s)
    f = np.asarray(f)
    finite = np.isfinite(s) & np.isfinite(f)
    s_safe = np.nan_to_num(s, nan=s_values[0])
    f_safe = np.nan_to_num(f, nan=f_values[0])
    s_index = np.clip(
        np.rint((s_safe - s_values[0]) / s_step).astype(int),
        0,
        len(s_values) - 1,
    )
    f_index = np.clip(
        np.rint((f_safe - f_values[0]) / f_step).astype(int),
        0,
        len(f_values) - 1,
    )

    adjacent = np.zeros(s.shape, dtype=bool)
    for delta_s in range(-2, 3):
        for delta_f in range(-2, 3):
            sample = phase[
                np.clip(f_index + delta_f, 0, len(f_values) - 1),
                np.clip(s_index + delta_s, 0, len(s_values) - 1),
            ]
            adjacent |= (sample == 5) | (sample == 6)

    visible = (
        finite
        & adjacent
        & (s >= 1.0)
        & (s <= 3.0)
        & (f >= 0.0)
        & (f <= 1.0)
    )
    ax.plot(
        np.where(visible, s, np.nan),
        np.where(visible, f, np.nan),
        color="black",
        lw=0.7,
        ls="-",
    )

# Three input directions: A, B, and the union of both barcodes.
phase_A = phase.copy()
phase_B = np.zeros_like(phase)
phase_B[:, s_values > theta] = 2
phase_union = np.zeros_like(phase)
f_union = (1.0 - 1.0 / alpha_A) / (1.0 - 1.0 / alpha_B)
phase_union[(s_values[None, :] > theta) & (f_values[:, None] < f_union)] = 3
phase_union[(s_values[None, :] > theta) & (f_values[:, None] >= f_union)] = 2

plt.rcParams.update({'font.family': 'Arial', 'font.size': 10,
                     'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': 0.7})
titles = [r'$I_i \propto b_i^A$', r'$I_i \propto b_i^B$', r'$I_i \propto \max(b_i^A,b_i^B)$']
for i, (data, title) in enumerate(zip([phase_A, phase_B, phase_union], titles)):
    fig, ax = plt.subplots(figsize=(1.5 * 2.2, 1.5 * 2))
    ax.imshow(data, origin='lower', aspect='auto', interpolation='nearest',
              extent=[0, 3, 0, 1], cmap=ListedColormap(colors), vmin=-0.5, vmax=6.5)
    ax.axvline(1, color='k', lw=0.7)
    if i == 0:
        ax.plot([1, 3], [f_A_boundary, f_A_boundary], color='k', lw=0.7)
        plot_phase_boundary(ax, s_curve, f_curve)
        for sf, ff in fold_curves:
            plot_fold_next_to_multistability(ax, sf, ff)
    if i == 2:
        ax.plot([1, 3], [f_union, f_union], color='k', lw=0.7)
    ids = [1, 2, 3, 4, 5] if i == 0 else ([2] if i == 1 else [2, 3])
    ax.legend(handles=[Patch(facecolor=colors[j], label=labels[j]) for j in ids],
              loc='upper left', frameon=False, fontsize=5, handlelength=1.5)
    ax.set(xlim=(0, 3), ylim=(0, 1), xlabel=r'rule-input strength $s/\theta$',
           ylabel=r'barcode sparsity $f$', title=title)
    ax.text(-0.18, 1.07, 'ABC'[i], transform=ax.transAxes, weight='bold')
    fig.tight_layout()
    savefig('input_barcode_' + ['A', 'B', 'union'][i], bbox_inches='tight')

plt.show()
