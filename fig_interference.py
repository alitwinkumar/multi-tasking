"""ReLU interference: separate and combined task readouts and connectivity blocks."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

output_dir = Path(__file__).resolve().parent / 'plots' / Path(__file__).stem
output_dir.mkdir(parents=True, exist_ok=True)


def savefig(name, **kwargs):
    fig = plt.gcf()
    stem = Path(name).stem
    fig.savefig(output_dir / (stem + '.pdf'), **kwargs)

# dx/dt = -x + J phi(x), with phi(x) = max(x - theta, 0).
N = 1000
rank = 2
dt = 0.1
theta = -1.0  # Tonic drive; theta=0 gives unshifted ReLU and decaying activity.


def phi(x):
    return np.maximum(x - theta, 0.0)

sA = 1.3
omA = 0.45

sB = 1.7
omB = 1.7

# Keep this draw order: A then B.
np.random.seed(22)

Sigma = np.eye(2 * rank)
Sigma_mn = np.array([[sA, omA], [-omA, sA]])
Sigma[:rank, rank:] = Sigma_mn
Sigma[rank:, :rank] = Sigma_mn.T
while np.min(np.linalg.eigvalsh(Sigma)) < 0:
    Sigma[rank:, rank:] *= 1.1

X = np.random.multivariate_normal(np.zeros(2 * rank), Sigma, size=N)
mA = X[:, :rank]
nA = X[:, rank:]
JA = mA @ nA.T / N
wA = mA[:, 0] / (0.8* N)

Sigma = np.eye(2 * rank)
Sigma_mn = np.array([[sB, omB], [-omB, sB]])
Sigma[:rank, rank:] = Sigma_mn
Sigma[rank:, :rank] = Sigma_mn.T
while np.min(np.linalg.eigvalsh(Sigma)) < 0:
    Sigma[rank:, rank:] *= 1.1

X = np.random.multivariate_normal(np.zeros(2 * rank), Sigma, size=N)
mB = X[:, :rank]
nB = X[:, rank:]
JB = mB @ nB.T / N
wB = mB[:, 0] / (3*N)

x0 = 0.1 * np.random.randn(N)

def run(m, n, T):
    time = np.arange(0, T, dt)
    x = x0.copy()
    outA = np.zeros(len(time))
    outB = np.zeros(len(time))

    def drift(x):
        # J = m @ n.T / N; apply its factors without forming a dense product.
        return -x + m @ (n.T @ phi(x)) / N

    for it in range(len(time)):
        x += dt * drift(x)
        r = phi(x)
        outA[it] = wA @ r
        outB[it] = wB @ r
    return time, outA, outB

# Isolated networks: retain the final 50 time units.
time, outA, _ = run(mA, nA, 500)
_, _, outB = run(mB, nB, 500)
keep = time > 450

# Combined network: retain the initial transient.
timeC, outCA, outCB = run(np.column_stack((mA, mB)), np.column_stack((nA, nB)), 50)
keepC = timeC > 0

# Common limits within each task preserve the separate/combined comparison.
y_limits = {
    'A': 1.1 * max(np.max(np.abs(outA[keep])), np.max(np.abs(outCA[keepC]))),
    'B': 1.1 * max(np.max(np.abs(outB[keep])), np.max(np.abs(outCB[keepC]))),
}

blue = '#4034d4'
pink = '#ad0864'
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'lines.linewidth': 1.2,
})

traces = [
    (time[keep] - 450, outA[keep], 'A', blue, 'task_A'),
    (timeC[keepC], outCA[keepC], 'A', blue, 'combined_A'),
    (time[keep] - 450, outB[keep], 'B', pink, 'task_B'),
    (timeC[keepC], outCB[keepC], 'B', pink, 'combined_B'),
]

for t, y, task, color, name in traces:
    fig, ax = plt.subplots(figsize=(3.25, 2))
    ax.plot(t, y, color=color)
    ax.set(xlim=(0, 50), ylim=(-y_limits[task], y_limits[task]), xticks=[0, 20, 40], xlabel='time (s)', ylabel='output task ' + task)
    fig.tight_layout()
    savefig(name, transparent=True)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.25, 3))
for ax, (t, y, task, color, name) in zip(axes, [traces[1], traces[3]]):
    ax.plot(t, y, color=color)
    ax.set(xlim=(0, 50), ylim=(-y_limits[task], y_limits[task]), xticks=[0, 20, 40], xlabel='time (s)', ylabel='output task ' + task)
axes[0].set_xlabel('')
fig.tight_layout()
savefig('interference_outputs')

# Connectivity blocks, with the same scale for both tasks.
for J, task in [(JA, 'A'), (JB, 'B')]:
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.pcolor(J[:20, :20], cmap='bwr', rasterized=True, vmin=-9/N, vmax=9/N)
    ax.axis('off')
    savefig('connectivity_' + task, transparent=True)

plt.show()
