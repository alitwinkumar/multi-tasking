"""Projected flow at input strengths 0, 0.1, 0.2, 0.3 and the zero-input 3D attractors."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

output_dir = Path(__file__).resolve().parent / 'plots' / Path(__file__).stem
output_dir.mkdir(parents=True, exist_ok=True)


def savefig(name, **kwargs):
    fig = plt.gcf()
    stem = Path(name).stem
    fig.savefig(output_dir / (stem + '.pdf'), **kwargs)

import itertools
from matplotlib.colors import LogNorm
from scipy.optimize import brentq, least_squares, root

def_colors = plt.rcParams['axes.prop_cycle'].by_key()["color"]

plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['lines.linewidth'] = 0.7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.major.pad'] = 2
plt.rcParams['ytick.major.pad'] = 2
plt.rcParams['xtick.minor.size'] = 1.5
plt.rcParams['ytick.minor.size'] = 1.5
plt.rcParams['xtick.minor.width'] = 0.4
plt.rcParams['ytick.minor.width'] = 0.4
plt.rcParams['figure.figsize'] = [6.5*2.5/2.54, 5/2.54]#[7.5/2.54, 5/2.54]
plt.rcParams['path.simplify'] = True
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['image.aspect'] = 'auto'
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['legend.frameon'] = False
plt.rcParams['patch.edgecolor'] = 'none'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.formatter.limits'] = (-3,3)

seed = 13
rng = np.random.default_rng(seed)

theta = -0.3
dt = 0.04
tau = 1.0
epoch_T = 21.0
x0_scale = 0.03

N = 10000

np.random.seed(100)

sigma1 = 1.5
sigma2= 1.3
omega1= 1.2

thetas = 2*np.pi*np.arange(N)/N

def phi(x):
    return(np.maximum(x+1,0))

def dyns_step(x, alpha=0.0):
    dx = -x
    r = phi(x)
    dx += (1/N)*Ms.dot(Ns.T.dot(r))
    dx += alpha*Ms[:,2]
    return dx

def cart_to_polar(Ks):
    Nt = np.shape(Ks)[1]
    Rhos = np.zeros((2,Nt))
    Thetas = np.zeros((1,Nt))
    Rhos[0,:] = np.sqrt(Ks[0,:]**2+Ks[1,:]**2)
    Rhos[1,:] = Ks[2,:]
    Thetas[0,:]= np.arctan2(Ks[1,:], Ks[0,:])
    return(Rhos, Thetas)

T = 60
dt = 0.1
time = np.arange(0, T, step=dt)

Sigma = np.eye(6)
Sigma[0:3,3:] = np.array(((sigma1, omega1, 0), (-omega1, sigma1, 0), (0,0, sigma2)))
Sigma[3:,0:3] = np.array(((sigma1, omega1, 0), (-omega1, sigma1, 0), (0,0, sigma2))).T

eigs = np.min(np.linalg.eigvals(Sigma))
maxiter = 40
i0 = 0
while eigs<0 and i0<maxiter:
    Sigma[3:,3:] = 1.1*Sigma[3:,3:]
    eigs = np.min(np.linalg.eigvals(Sigma))
    i0 +=1

XX = np.random.multivariate_normal(np.zeros(6), Sigma, size=N)

Ms = XX[:,0:3]
Ns = XX[:, 3:]

Ks = np.zeros((3, len(time)))

Trials = 30

KKs = np.zeros((3, len(time), Trials))
RRhos   = np.zeros((2, len(time), Trials))
TThetas = np.zeros((1, len(time), Trials))

def projected_flow(kappa, alpha=0.0):
    x = Ms.dot(kappa)
    return Ms.T.dot(dyns_step(x, alpha=alpha))/N

alphas = [0.0, 0.1, 0.2, 0.3]
kappa2_grid = np.linspace(-2.5, 2.4, 220)

def plane_flow(kappa02, alpha=0.0):
    dK = projected_flow(np.array([kappa02[0], 0.0, kappa02[1]]), alpha=alpha)
    return np.array([dK[0], dK[2]])

def plane_jacobian(kappa02, alpha=0.0, eps=1e-4):
    jac = np.zeros((2, 2))
    for ii in range(2):
        step = np.zeros(2)
        step[ii] = eps
        jac[:, ii] = (plane_flow(kappa02 + step, alpha) - plane_flow(kappa02 - step, alpha))/(2*eps)
    return jac

def projected_jacobian(kappa, alpha=0.0, eps=1e-4):
    jac = np.zeros((3, 3))
    for ii in range(3):
        step = np.zeros(3)
        step[ii] = eps
        jac[:, ii] = (projected_flow(kappa + step, alpha) - projected_flow(kappa - step, alpha))/(2*eps)
    return jac

def is_stable_fixed_point(kappa02, alpha=0.0):
    eigs = np.linalg.eigvals(plane_jacobian(kappa02, alpha=alpha))
    return np.all(np.real(eigs) < 0)

def all_fixed_points(alpha, kappa0_range, kappa2_range):
    guesses = []
    for k0 in np.linspace(kappa0_range[0], kappa0_range[1], 17):
        for k2 in np.linspace(kappa2_range[0], kappa2_range[1], 17):
            guesses.append(np.array([k0, k2]))

    fixed_points = []
    for guess in guesses:
        sol = root(lambda z: plane_flow(z, alpha=alpha), guess)
        if not sol.success:
            continue
        point = sol.x
        if not (kappa0_range[0] <= point[0] <= kappa0_range[1] and kappa2_range[0] <= point[1] <= kappa2_range[1]):
            continue
        if np.linalg.norm(plane_flow(point, alpha=alpha)) > 1e-4:
            continue
        if any(np.linalg.norm(point - old_point) < 0.05 for old_point, _ in fixed_points):
            continue
        fixed_points.append((point, is_stable_fixed_point(point, alpha=alpha)))
    return fixed_points

cc_lc = '#3332CB'
cc_fixed = '#9F2D69'

kappa0_grid = np.linspace(-3.0, 3.0, 49)
kappa2_grid = np.linspace(-2.5, 2.5, 49)
K0, K2 = np.meshgrid(kappa0_grid, kappa2_grid)

fig, axs = plt.subplots(1, len(alphas), figsize=(1.7*len(alphas), 1.8),
                        sharex=True, sharey=True, constrained_layout=True)
speed_floor = 5e-3
flow_fields = []
speed_max = speed_floor
for alpha in alphas:
    dK0 = np.zeros_like(K0)
    dK2 = np.zeros_like(K2)
    speed = np.zeros_like(K0)

    for index in np.ndindex(K0.shape):
        dK = projected_flow(np.array([K0[index], 0.0, K2[index]]), alpha=alpha)
        dK0[index] = dK[0]
        dK2[index] = dK[2]
        speed[index] = np.hypot(dK0[index], dK2[index])

    speed_max = max(speed_max, np.max(speed))
    flow_fields.append((alpha, dK0, dK2, speed))

speed_norm = LogNorm(vmin=speed_floor, vmax=speed_max)
images = []
for ax, (alpha, dK0, dK2, speed) in zip(np.ravel(axs), flow_fields):
    image = ax.imshow(np.maximum(speed, speed_floor),
                      extent=[kappa0_grid[0], kappa0_grid[-1], kappa2_grid[0], kappa2_grid[-1]],
                      origin="lower", cmap="bone", norm=speed_norm, alpha=0.9)
    images.append(image)
    ax.streamplot(K0, K2, dK0, dK2, color="0.3", density=0.55, linewidth=0.8, arrowsize=0.8)
    ax.axhline(0, color="0.85", lw=0.5)
    ax.axvline(0, color="0.85", lw=0.5)
    for point, stable in all_fixed_points(alpha, (kappa0_grid[0], kappa0_grid[-1]), (kappa2_grid[0], kappa2_grid[-1])):
        if alpha>0.15:
            cc_fixed='k'
        facecolor = cc_lc if stable else "w"
        edgecolor= "w" if stable else cc_fixed
        if alpha>0.28:
            facecolor = cc_fixed
        ax.scatter(point[0], point[1], s=30, facecolor=facecolor, edgecolor=edgecolor, linewidth=0.7, zorder=5)
        if stable and alpha<0.29:
            if point[0]<0:
                ax.scatter(point[0], point[1], s=18, color='w', marker='x',  zorder=5)
            else:
                ax.scatter(point[0], point[1], s=3, facecolor='w', marker='.',  zorder=5)
    cc_fixed = '#9F2D69'
    ax.set_xlabel(r"$\kappa_0$", color='w')
    ax.set_xlim(kappa0_grid[0], kappa0_grid[-1])
    ax.set_ylim(kappa2_grid[0], kappa2_grid[-1])
    for spine in ax.spines.values():
        spine.set_visible(True)

axs[0].set_ylabel(r"$\kappa_2$", color='w')
savefig('multitask_supfig1_a_nocolorbar.pdf')
fig.colorbar(images[-1], ax=np.ravel(axs), label=r"log(speed)", fraction=0.035, pad=0.02)
savefig('multitask_supfig1_a.pdf')

def optimize_projected_fixed_point(initial_kappa, alpha=0.0, kappa_bounds=(-3.5, 3.5)):
    lower = np.full(3, kappa_bounds[0])
    upper = np.full(3, kappa_bounds[1])
    sol = least_squares(lambda z: projected_flow(z, alpha=alpha),
                        np.clip(initial_kappa, lower, upper),
                        bounds=(lower, upper),
                        xtol=1e-11, ftol=1e-11, gtol=1e-11,
                        max_nfev=100)
    point = sol.x
    residual = np.linalg.norm(projected_flow(point, alpha=alpha))
    return point, residual

def axis_kappa2_roots(alpha=0.0, kappa2_min=-3.0, kappa2_max=3.0):
    def flow_on_axis(kappa2):
        return projected_flow(np.array([0.0, 0.0, kappa2]), alpha=alpha)[2]

    grid = np.linspace(kappa2_min, kappa2_max, 700)
    vals = np.array([flow_on_axis(kappa2) for kappa2 in grid])
    roots = []
    for ii in range(len(grid)-1):
        if vals[ii] == 0:
            roots.append(grid[ii])
        elif vals[ii]*vals[ii+1] < 0:
            roots.append(brentq(flow_on_axis, grid[ii], grid[ii+1]))
    return roots

alpha_plot = 0.0
kappa2_roots = axis_kappa2_roots(alpha=alpha_plot)
center_fp, center_residual = optimize_projected_fixed_point(np.zeros(3), alpha=alpha_plot)
positive_fp, positive_residual = optimize_projected_fixed_point(
    np.array([0.0, 0.0, max(kappa2_roots)]), alpha=alpha_plot)
negative_fp, negative_residual = optimize_projected_fixed_point(
    np.array([0.0, 0.0, min(kappa2_roots)]), alpha=alpha_plot)

lc_dt = 0.04
lc_burn_time = 120.0
lc_plot_time = 20.0
lc_burn_steps = int(lc_burn_time/lc_dt)
lc_plot_steps = int(lc_plot_time/lc_dt)
kappa = np.array([1.5, 0.0, 0.0])

for _ in range(lc_burn_steps):
    kappa = kappa + lc_dt*projected_flow(kappa, alpha=alpha_plot)

limit_cycle = np.zeros((lc_plot_steps, 3))
for it in range(lc_plot_steps):
    limit_cycle[it, :] = kappa
    kappa = kappa + lc_dt*projected_flow(kappa, alpha=alpha_plot)

fig = plt.figure(figsize=[1.5*2.2, 1.5*2])
ax = fig.add_subplot(111, projection="3d")

ax.plot(limit_cycle[:, 0], limit_cycle[:, 1], limit_cycle[:, 2],
        c=cc_lc, ls="-", lw=1.2, zorder=10)
ax.scatter(center_fp[0], center_fp[1], center_fp[2],
           edgecolor='k', color='w', s=35, zorder=30)
ax.scatter([positive_fp[0], negative_fp[0]],
           [positive_fp[1], negative_fp[1]],
           [positive_fp[2], negative_fp[2]],
           edgecolor=cc_fixed, color='w', s=35, zorder=40)

axis_len = 2.0
ax.plot([0, 2.*axis_len], [0, 0], [0, 0], c=cc_lc, lw=1.5, zorder=0)
ax.plot([0, 0], [0, 1.8*axis_len], [0, 0], c=cc_lc, lw=1.5, zorder=0)
ax.plot([0, 0], [0, 0], [0, 0.9*axis_len], c=cc_fixed, lw=1.5, zorder=0)

ax.axis('off')
ax.view_init(elev=24.01, azim=29.38)
ax.set_box_aspect(aspect=(4, 4, 3), zoom=1.)
ax.set_xlim(-2.9, 2.9)
ax.set_ylim(-2.9, 2.9)
ax.set_zlim(-0.9, 0.9)

savefig('multitask_supfig1_a_3d_cycle_fixed_points.pdf', transparent=True)

plt.show()
