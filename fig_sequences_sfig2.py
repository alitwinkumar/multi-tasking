"""Sequence tasks: K_A=20, K_B=30, gains 1.55 and 1.25; mixed initial state 1/4 A + 3/4 B."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

output_dir = Path(__file__).resolve().parent / 'plots' / Path(__file__).stem
output_dir.mkdir(parents=True, exist_ok=True)


def savefig(name, **kwargs):
    fig = plt.gcf()
    stem = Path(name).stem
    fig.savefig(output_dir / (stem + '.pdf'), **kwargs)

def set_plot():
    plt.style.use("ggplot")

    fig_width = 2.5 * 1.3
    fig_height = 2
    fig_size = [fig_width, fig_height]
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["pdf.fonttype"] = 42

    plt.rcParams["lines.linewidth"] = 1.2
    plt.rcParams["lines.markeredgewidth"] = 0.003
    plt.rcParams["lines.markersize"] = 3
    plt.rcParams["font.size"] = 14
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["axes.facecolor"] = "1"
    plt.rcParams["axes.edgecolor"] = "0"
    plt.rcParams["axes.linewidth"] = "0.7"

    plt.rcParams["axes.labelcolor"] = "0"
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["xtick.color"] = "0"
    plt.rcParams["ytick.color"] = "0"
    plt.rcParams["xtick.major.size"] = 2
    plt.rcParams["ytick.major.size"] = 2

    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["axes.grid"] = False

def new_axis(figsize=None):
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    clean_axis(ax)
    return fig, ax

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

set_plot()

# Sequence lengths, gains, and sampling seed.
seed = 11
rng = np.random.default_rng(seed)

N = 10000
f = 1.0
delta_m = 0.1

K1 = 20
K2 = 30

T = 40.0
dt = 0.04
tau = 1.0

theta = -0.3
gI = 1.6

gain1 = 1.55
gain2 = 1.25         
initial_amplitude = 1.0

out_prefix = "two_sequence_mixture_readouts"

def phi(x):
    return np.maximum(0.0, x)

def gaussian_patterns(K):
    return rng.normal(size=(N, K))

def make_exact_mask():
    mask = np.zeros(N, dtype=bool)
    active = rng.choice(N, size=int(round(f * N)), replace=False)
    mask[active] = True
    return mask

def apply_barcode_to_patterns(raw, mask):
    m_barcode = np.where(mask, 1.0, delta_m)
    n_barcode = mask.astype(float)
    m_patterns = raw * m_barcode[:, None]
    n_patterns = raw * n_barcode[:, None]
    return m_patterns, n_patterns

def make_sequence_patterns(K):
    mask = make_exact_mask()
    raw = gaussian_patterns(K)
    m_patterns, n_patterns = apply_barcode_to_patterns(raw, mask)
    return m_patterns, n_patterns, mask

def pattern_readout(m_patterns, y):
    norm = np.sum(m_patterns * m_patterns, axis=0)
    norm = np.maximum(norm, 1e-12)
    return 10*m_patterns.T @ y / norm

def n_readout(n_patterns, r):
    return n_patterns.T @ r / (f * N)

def sequence_input(m_patterns, n_patterns, gain, r):
    q = n_readout(n_patterns, r)
    return gain * (m_patterns[:, 1:] @ q[:-1])

# Euler integration of dx/dt=-x+J ReLU(x-theta)-gI mean(r).
def simulate(m_list, n_list, gain_list, x0):
    n_steps = int(round(T / dt))
    t = np.arange(n_steps + 1) * dt

    x = x0.copy()
    all_readouts = [
        np.zeros((m.shape[1], n_steps + 1))
        for m in m_list
    ]

    r = phi(x - theta)
    for i, m in enumerate(m_list):
        all_readouts[i][:, 0] = pattern_readout(m, r)

    for step in range(n_steps):
        r = phi(x - theta)

        recurrent = np.zeros(N)
        for m, n, gain in zip(m_list, n_list, gain_list):
            recurrent += sequence_input(m, n, gain, r)

        inhibition = gI * np.mean(r)
        dx = -x + recurrent - inhibition
        x = x + (dt / tau) * dx

        r = phi(x - theta)
        for i, m in enumerate(m_list):
            all_readouts[i][:, step + 1] = pattern_readout(m, r)

    return t, all_readouts

def plot_sequence_readouts(ax, t, q, cmap_name):
    cmap = plt.get_cmap(cmap_name)
    K = q.shape[0]
    for k in range(K):
        color = cmap(0.15 + 0.4 * k / max(K - 1, 1))
        ax.plot(t, q[k], color=color)
    ax.set_xlabel("time")
    ax.set_ylabel("outputs")
    ax.set_ylim([0, 1.7])

m1, n1, mask1 = make_sequence_patterns(K1)
m2, n2, mask2 = make_sequence_patterns(K2)

x0_1 = initial_amplitude * m1[:, 0]
x0_2 = initial_amplitude * m2[:, 0]
x0_mix = initial_amplitude * (0.25 * m1[:, 0] + 0.75 * m2[:, 0])

t1, q_single_1 = simulate([m1], [n1], [gain1], x0_1)
t2, q_single_2 = simulate([m2], [n2], [gain2], x0_2)

t_mix, q_mix = simulate([m1, m2], [n1, n2], [gain1, gain2], x0_mix)

q_mix_initial_1 = pattern_readout(m1, phi(x0_mix - theta))[0]
q_mix_initial_2 = pattern_readout(m2, phi(x0_mix - theta))[0]


fig, ax = new_axis()
plot_sequence_readouts(ax, t1, q_single_1[0], "winter")
ax.set_ylabel("outputs task A")
plt.tight_layout()
savefig(out_prefix + "_task1_output.pdf")

fig, ax = new_axis()
plot_sequence_readouts(ax, t2, q_single_2[0], "autumn")
ax.set_ylabel("outputs task B")
plt.tight_layout()
savefig(out_prefix + "_task2_output.pdf")

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=[2.5 * 1.3, 3])

for ax_ in ax:
    clean_axis(ax_)

plot_sequence_readouts(ax[0], t_mix, q_mix[0], "winter")
plot_sequence_readouts(ax[1], t_mix, q_mix[1], "autumn")
ax[0].set_xlabel("")
ax[1].set_xlabel("time")
ax[0].set_ylabel("outputs A")
ax[1].set_ylabel("outputs B")
ax[0].set_ylim([0, 1.3])
ax[1].set_ylim([0, 1.3])

plt.tight_layout()
savefig(out_prefix + "_task1and2_output.pdf")

J1_plot = gain1 * (m1[:20, 1:] @ n1[:20, :-1].T) / (f * N)
J2_plot = gain2 * (m2[:20, 1:] @ n2[:20, :-1].T) / (f * N)

fig = plt.figure(figsize=[2, 2])
ax = fig.add_subplot(111)
ax.pcolor(J1_plot, cmap="bwr", rasterized=True, vmin=-19 / N, vmax=19 / N)
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.axis('off')
savefig(out_prefix + "_task1_connectivity.pdf")

fig = plt.figure(figsize=[2, 2])
ax = fig.add_subplot(111)
ax.pcolor(J2_plot, cmap="bwr", rasterized=True, vmin=-19 / N, vmax=19 / N)
ax.tick_params(left=True, bottom=False, labelleft=False, labelbottom=False)
plt.axis('off')
savefig(out_prefix + "_task2_connectivity.pdf")

plt.show()
