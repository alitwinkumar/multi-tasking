"""Unit-variance distributions, Stein kernels, and transverse gain beta_P."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

output_dir = Path(__file__).resolve().parent / 'plots' / Path(__file__).stem
output_dir.mkdir(parents=True, exist_ok=True)


def savefig(name, **kwargs):
    fig = plt.gcf()
    stem = Path(name).stem
    fig.savefig(output_dir / (stem + '.pdf'), **kwargs)

from scipy.optimize import brentq
from scipy.special import ndtr

plt.style.use("ggplot")

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.figsize"] = [1.5 * 2.2, 1.5 * 2]
plt.rcParams["figure.autolayout"] = True

plt.rcParams["lines.linewidth"] = 1.2
plt.rcParams["lines.markeredgewidth"] = 0.003
plt.rcParams["lines.markersize"] = 3

plt.rcParams["font.size"] = 14
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["font.sans-serif"] = "Arial"

plt.rcParams["axes.facecolor"] = "1"
plt.rcParams["axes.edgecolor"] = "0"
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["axes.labelcolor"] = "0"
plt.rcParams["axes.labelsize"] = 12

plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["xtick.color"] = "0"
plt.rcParams["ytick.color"] = "0"
plt.rcParams["xtick.major.size"] = 2
plt.rcParams["ytick.major.size"] = 2

g_I = 0.5
theta = -1.0

mmin, mmax = -2.5, 2.5
sigma_min, sigma_max = 1.0, 2.0

a_arc = np.sqrt(2.0)
a_unif = np.sqrt(3.0)
b_lap = 1.0 / np.sqrt(2.0)

w_sparse = 0.15
eps_sparse = 0.10

var_small_uniform = eps_sparse**2 / 3.0
sigma_sparse = np.sqrt((1.0 - w_sparse * var_small_uniform) / (1.0 - w_sparse))

colors = {
    "Arcsine": "#b23a48",
    "Uniform": "#e07a5f",
    "Gaussian": "black",
    "Laplace": "#315a9e",
    "Sparse": "#5aa9d6",
}

order = ["Arcsine", "Uniform", "Gaussian", "Laplace", "Sparse"]

def gaussian_pdf(x, sigma=1.0):
    return np.exp(-0.5 * (x / sigma)**2) / (np.sqrt(2.0 * np.pi) * sigma)

def arcsine_pdf(x):
    out = np.zeros_like(x, dtype=float)
    inside = np.abs(x) < a_arc
    out[inside] = 1.0 / (np.pi * np.sqrt(a_arc**2 - x[inside]**2))
    return out

def uniform_pdf(x):
    out = np.zeros_like(x, dtype=float)
    inside = np.abs(x) <= a_unif
    out[inside] = 1.0 / (2.0 * a_unif)
    return out

def laplace_pdf(x):
    return np.exp(-np.abs(x) / b_lap) / (2.0 * b_lap)

def sparse_pdf(x):
    narrow = np.where(np.abs(x) <= eps_sparse, 1.0 / (2.0 * eps_sparse), 0.0)
    broad = gaussian_pdf(x, sigma_sparse)
    return w_sparse * narrow + (1.0 - w_sparse) * broad

PDF = {
    "Arcsine": arcsine_pdf,
    "Uniform": uniform_pdf,
    "Gaussian": lambda x: gaussian_pdf(x, 1.0),
    "Laplace": laplace_pdf,
    "Sparse": sparse_pdf,
}

# S(c)=P(m>c), T(c)=E[m 1(m>c)], U(c)=E[m^2 1(m>c)].
def gaussian_STU(c, sigma=1.0):
    z = c / sigma
    phi = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)

    S = 1.0 - ndtr(z)
    T = sigma * phi
    U = sigma**2 * (S + z * phi)

    return S, T, U

def uniform_STU(c, a=a_unif):
    if c <= -a:
        return 1.0, 0.0, a**2 / 3.0

    if c >= a:
        return 0.0, 0.0, 0.0

    S = (a - c) / (2.0 * a)
    T = (a**2 - c**2) / (4.0 * a)
    U = (a**3 - c**3) / (6.0 * a)

    return S, T, U

def arcsine_STU(c):
    if c <= -a_arc:
        return 1.0, 0.0, 1.0

    if c >= a_arc:
        return 0.0, 0.0, 0.0

    z = c / a_arc
    angle = np.arcsin(z)

    S = np.arccos(z) / np.pi
    T = np.sqrt(a_arc**2 - c**2) / np.pi
    U = a_arc**2 / np.pi * (np.pi / 4.0 - angle / 2.0 + np.sin(2.0 * angle) / 4.0)

    return S, T, U

def laplace_STU(c):
    b = b_lap

    if c >= 0:
        e = np.exp(-c / b)

        S = 0.5 * e
        T = 0.5 * (c + b) * e
        U = 0.5 * (c**2 + 2.0 * b * c + 2.0 * b**2) * e

    else:
        e = np.exp(c / b)

        S = 1.0 - 0.5 * e
        T = 0.5 * (-c + b) * e
        U = 2.0 * b**2 - 0.5 * (c**2 - 2.0 * b * c + 2.0 * b**2) * e

    return S, T, U

def sparse_STU(c):
    S1, T1, U1 = uniform_STU(c, a=eps_sparse)
    S2, T2, U2 = gaussian_STU(c, sigma=sigma_sparse)

    S = w_sparse * S1 + (1.0 - w_sparse) * S2
    T = w_sparse * T1 + (1.0 - w_sparse) * T2
    U = w_sparse * U1 + (1.0 - w_sparse) * U2

    return S, T, U

STU = {
    "Arcsine": arcsine_STU,
    "Uniform": lambda c: uniform_STU(c, a_unif),
    "Gaussian": lambda c: gaussian_STU(c, 1.0),
    "Laplace": laplace_STU,
    "Sparse": sparse_STU,
}

def tau_gaussian(x):
    return np.ones_like(x)

def tau_arcsine(x):
    out = np.full_like(x, np.nan)
    inside = np.abs(x) <= a_arc
    out[inside] = a_arc**2 - x[inside]**2
    return out

def tau_uniform(x):
    out = np.full_like(x, np.nan)
    inside = np.abs(x) <= a_unif
    out[inside] = (a_unif**2 - x[inside]**2) / 2.0
    return out

def tau_laplace(x):
    return b_lap**2 + b_lap * np.abs(x)

def tau_sparse(x):
    p = sparse_pdf(x)
    T = np.array([sparse_STU(xx)[1] for xx in x])
    return T / p

TAU = {
    "Arcsine": tau_arcsine,
    "Uniform": tau_uniform,
    "Gaussian": tau_gaussian,
    "Laplace": tau_laplace,
    "Sparse": tau_sparse,
}

# Single-task fixed point: U(c)-c T(c)=1/sigma_A; beta_P=sigma_A S(c).
def D_of_c(c, stu):
    S, T, U = stu(c)
    return U - c * T

def cutoff_from_sigma(sigma_A, stu):
    target = 1.0 / sigma_A
    return brentq(lambda c: D_of_c(c, stu) - target, -20.0, 20.0)

def fixed_point_from_sigma(sigma_A, stu):
    c = cutoff_from_sigma(sigma_A, stu)
    S, T, U = stu(c)

    beta = sigma_A * S
    denom = c * (1.0 + g_I * S) - g_I * T

    if denom >= 0:
        return c, np.nan, np.nan, np.nan, False

    kappa = theta / denom
    xbar = theta - kappa * c

    return c, kappa, xbar, beta, True

m = np.linspace(mmin, mmax, 3001)
sigma_values = np.linspace(1.001, sigma_max, 600)

beta_curves = {}

for name in order:
    stu = STU[name]

    sigmas = []
    betas = []

    for sigma_A in sigma_values:
        c, kappa, xbar, beta, exists = fixed_point_from_sigma(sigma_A, stu)

        if exists:
            sigmas.append(sigma_A)
            betas.append(beta)

    beta_curves[name] = np.asarray(sigmas), np.asarray(betas)

plt.figure()

for name in order:
    plt.plot(m, PDF[name](m), lw=2.0, color=colors[name], label=name)

plt.xlim(mmin, mmax)
plt.ylim(0, 2.4)

plt.xlabel(r"$m$")
plt.ylabel(r"$P(m)$")

plt.legend(frameon=False, loc=9)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

savefig("fs_nongaussian_distributions.pdf", bbox_inches="tight")

plt.figure()

for name in order:
    plt.plot(m, TAU[name](m), lw=2.0, color=colors[name])

plt.axhline(1.0, color="0.7", lw=0.8, zorder=0)

plt.xlim(mmin, mmax)
plt.ylim(bottom=0)

plt.xlabel(r"$m$")
plt.ylabel(r"$\tau_P(m)$")

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

savefig("fs_nongaussian_stein_kernels.pdf", bbox_inches="tight")

plt.figure()

for name in order:
    sigmas, betas = beta_curves[name]
    plt.plot(sigmas, betas, lw=2.0, color=colors[name])

plt.axhline(1.0, color="0.7", lw=0.8, zorder=0)
plt.axvline(2.0, color="0.7", lw=0.8, ls="--", zorder=0)

plt.xlim(sigma_min, sigma_max)

plt.xlabel(r"$\sigma_A$")
plt.ylabel(r"$\beta_P$")

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

savefig("fs_nongaussian_beta_vs_sigma.pdf", bbox_inches="tight")

plt.show()
