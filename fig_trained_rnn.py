import torch
import numpy as np
import matplotlib.pyplot as plt

#parameters
N_in  = 4 #[task A, task B, stimulus 1, stimulus 2]
N = 500
N_out  = 2
g = 1.5

dt = 0.1
N_steps = 600

T_stimon  = 10
T_stimoff = 20
T_resp  = 50

freq = 0.25

N_batch = 3
N_epochs = 1000
lr_start = 1e-3
lr_end   = 1e-4

def init_params():
    p = {
        "W_in": torch.randn(N, N_in)/np.sqrt(N_in),
        "J": (g / np.sqrt(N)) * torch.randn(N, N),
        "theta": -.2 * torch.ones(N),
        "W_out": 0.1 * torch.randn(N_out, N) / np.sqrt(N),
    }
    for v in p.values():
        v.requires_grad_(True)
    return p


def rnn_forward(p, h):
    N_steps, B, _ = h.shape
    x = torch.zeros(B, N)
    outs = []
    xs = []
    for t in range(N_steps):
        phi = torch.relu(x - p["theta"])
        x = (1 - dt) * x + dt * (h[t] @ p["W_in"].T + phi @ p["J"].T)
        outs.append(phi @ p["W_out"].T)
        xs.append(x)
    return torch.stack(outs), torch.stack(xs)

def make_batch(N_batch):
    h = torch.zeros(N_steps, N_batch, N_in) #input channel activity
    y = torch.zeros(N_steps, N_batch, N_out) #readouts
    m = torch.zeros(N_steps, N_batch, N_out) #loss mask
    t = torch.arange(N_steps) * dt

    task = np.arange(N_batch) % 3
    for i in range(N_batch):
        if task[i] == 0:
            h[:, i, 0] = 1.
            y[:, i, 0] = torch.sin(2 * np.pi * freq * t)
            m[T_resp:, i, :] = 1.
        else:
            h[:, i, 1] = 1.
            h[T_stimon:T_stimoff, i, task[i] + 1] = 1.
            y[T_resp:, i, 1] = 1. if task[i] == 1 else -1.
            m[T_resp:, i, :] = 1.
    return h, y, m

def train():
    p = init_params()
    opt = torch.optim.Adam(list(p.values()), lr=lr_start)
    gamma = (lr_end / lr_start) ** (1.0 / (N_epochs - 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    for step in range(N_epochs):
        h, y, m = make_batch(N_batch)
        out, xs = rnn_forward(p, h)
        loss = (m * (out - y) ** 2).sum() / m.sum()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(p.values(), 1.0)
        opt.step()
        sched.step()

        if step % 10 == 0:
            print(f"step {step:5d}   loss {loss.item():.4f}   lr {sched.get_last_lr()[0]:.2e}")
    return p

def save_params(p, path):
    torch.save({k: v.detach() for k, v in p.items()}, path)

def load_params(path):
    state = torch.load(path)
    p = {k: v.clone().requires_grad_(True) for k, v in state.items()}
    return p

#simulate trial types
def run_conditions(p):
    he = torch.zeros(N_steps, 3, N_in)
    he[:, 0, 0] = 1.0
    he[:, 1, 1] = 1.0
    he[T_stimon:T_stimoff, 1, 2] = 1.0
    he[:, 2, 1] = 1.0
    he[T_stimon:T_stimoff, 2, 3] = 1.0
    with torch.no_grad():
        out, xs = rnn_forward(p, he)
    return out, xs

#compute task subspaces using PCA
def task_subspaces(xs):
    XA = xs[T_resp:, 0, :]
    XA = XA - XA.mean(dim=0, keepdim=True)
    _, SA, VA = torch.linalg.svd(XA, full_matrices=False)

    XB = xs[T_resp:, 1:, :].reshape(-1, N)
    XB = XB - XB.mean(dim=0, keepdim=True)
    _, SB, VB = torch.linalg.svd(XB, full_matrices=False)

    return VA[:2], VB[:1], SA, SB

#compute summary statistics
def compute_stats(path):
    p = load_params(path)
    _, xs = run_conditions(p)

    A_axes, B_axis, SA, SB = task_subspaces(xs)

    #variance explained
    A_var_explained = (SA ** 2 / (SA ** 2).sum())[:2]
    B_var_explained = (SB ** 2 / (SB ** 2).sum())[:1]

    #subspace alignment
    cos_pa = torch.linalg.svdvals(A_axes @ B_axis.T)
    subspace_angle = torch.rad2deg(torch.arccos(cos_pa.clamp(-1.0, 1.0)))

    #per-neuron task selectivity
    phis = torch.relu(xs - p["theta"])
    var_A = phis[:, 0, :].var(dim=0, unbiased=False)
    var_B = phis[:, 1:, :].reshape(-1, N).var(dim=0, unbiased=False)
    selectivity = (torch.sqrt(var_A) - torch.sqrt(var_B)) / (torch.sqrt(var_A) + torch.sqrt(var_B))

    return {
        "selectivity": selectivity,
        "A_var_explained": A_var_explained,
        "B_var_explained": B_var_explained,
        "subspace_angle": subspace_angle,
    }

Nnets = 10
newparams = False
if newparams:
    for n in range(Nnets):
        print(f"training network {n+1}")
        p = train()
        save_params(p, f"data/rnn_two_tasks_params_{n}.pt")

iload = 0
p = load_params(f"data/rnn_two_tasks_params_{iload}.pt")

N_steps = 1000 #test time
t = torch.arange(N_steps) * dt
ye = torch.zeros(N_steps, 3, N_out)
ye[:, 0, 0] = torch.sin(2*np.pi*freq*t)
ye[T_resp:, 1, 1] = 1.
ye[T_resp:, 2, 1] = -1.

out, xs = run_conditions(p)

taskA_axes, taskB_axis, SA, SB = task_subspaces(xs)
axes3 = torch.cat([taskA_axes, taskB_axis], dim=0)

#project trajectories onto axes
Xfull = xs.reshape(-1, N)
Xcfull = Xfull - Xfull.mean(dim=0, keepdim=True)
pcs = Xcfull @ axes3.T
pcs = pcs.reshape(N_steps, 3, 3)

#summary statistics
all_stats = [compute_stats(f"data/rnn_two_tasks_params_{n}.pt")
             for n in range(Nnets)]
stats = all_stats[0]
subspace_angle = stats["subspace_angle"]

selectivity = torch.cat([s["selectivity"] for s in all_stats])

angles = torch.stack([s["subspace_angle"][0] for s in all_stats])
print(f"subspace angle: {angles.mean():.1f} +/- {angles.std():.1f} deg  "
      f"(across {Nnets} nets)")

P = pcs.numpy()
all_x = P[:, :, 0].ravel()
all_y = P[:, :, 1].ravel()
all_z = P[:, :, 2].ravel()

c_A = "#3332CC"
c_B = "#A3145F"
c_B2 = "#C995B4"

cols = ["k","#b3b3b3","#808080"]

fig3 = plt.figure(figsize=(2.5,2.5))
ax3 = fig3.add_subplot(1, 1, 1, projection="3d")
markersize=15
markerlw=0.5

for bi in range(3):
    traj = P[:, bi, :]
    ax3.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=cols[bi])
    ax3.scatter( traj[-1,0], traj[-1,1], traj[-1,2], color=cols[bi],marker='o',s=markersize,lw=markerlw) #end marker
    ax3.scatter( traj[0,0], traj[0,1], traj[0,2],marker='o',s=markersize,lw=markerlw, facecolors='white', edgecolors=cols[bi]) #start marker
 

fac = 1.2
Lx, Ly, Lz = fac*np.abs(all_x).max(), fac*np.abs(all_y).max(), fac*np.abs(all_z).max()

ax3.set_xlim(fac*all_x.min(), fac*all_x.max())
ax3.set_ylim(fac*all_y.min(), fac*all_y.max())
ax3.set_zlim(fac*all_z.min(), fac*all_z.max())
ax3.set_box_aspect((1, 1, 1))
ax3.grid(False)
ax3.set_axis_off()
ax3.plot([0, 0], [0, -Ly], [0, 0], color=c_A)
ax3.plot([0, Lx], [0, 0], [0, 0], color=c_A)
ax3.plot([0, 0], [0, 0], [0, Lz], color=c_B)
ax3.set_facecolor('none')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_zticks([])

fig3.tight_layout()
fig3.savefig("figs/rule_trajs.svg")

sel_conn = np.load("data/connectivity_sel.npz")['sel']
sel_conn = sel_conn[:(Nnets*N)]

figs, axsel = plt.subplots(figsize=(1.5, 1.5))
axsel.hist(sel_conn, bins=10, range=(-1, 1),
           color="#ff6600",lw=0.5,histtype='step')
axsel.hist(selectivity.detach().numpy(), bins=10, range=(-1, 1),
           color="k",lw=0.5,histtype='step')


axsel.set_xlabel("Task selectivity")
axsel.set_ylabel("Count")
figs.tight_layout()
figs.savefig("figs/neuron_sel.svg")

plt.show()

A_ve = torch.stack([s["A_var_explained"].sum() for s in all_stats])
B_ve = torch.stack([s["B_var_explained"][0] for s in all_stats])
print(f"task A variance explained: "
      f"{A_ve.mean():.3f} +/- {A_ve.std():.3f}  (across {Nnets} nets)")
print(f"task B variance explained: "
      f"{B_ve.mean():.3f} +/- {B_ve.std():.3f}  (across {Nnets} nets)")

###plot task outputs
fig = plt.figure(figsize=[1.3, 1.6],dpi=120)
ax = fig.add_subplot(211)

tinds = np.arange(300)
t = dt*tinds

readouts = out.detach().numpy()
plt.plot(t,out[tinds,0,0],color=c_A)
plt.plot(t,out[tinds,1,0],color=c_B)
plt.plot(t,out[tinds,2,0],color=c_B2)
plt.axvspan(t[T_stimon],t[T_stimoff],color="#bbbbbb",linewidth=0)

ax.set_ylabel('task A readout',fontsize=6)
plt.xticks((0,10,20,30),fontsize=5)
plt.tight_layout()


ax = fig.add_subplot(212)
plt.plot(t,out[tinds,0,1],color=c_A)
plt.plot(t,out[tinds,1,1],color=c_B)
plt.plot(t,out[tinds,2,1],color=c_B2)
plt.axvspan(t[T_stimon],t[T_stimoff],color="#bbbbbb",linewidth=0)

ax.set_xlabel('time',fontsize=6)
ax.set_ylabel('task B readout',fontsize=6)
plt.xticks((0,10,20,30),fontsize=5)
plt.tight_layout()

plt.savefig("figs/readouts_rule.svg")


