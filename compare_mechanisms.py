import numpy as np
import matplotlib.pyplot as plt
from utils import *

mechanism = "barcode" #"transient", "connectivity", or "barcode"

#parameters
N = 10000 #number of neurons
Ra = [2,1] #rank of each task
R = np.max(Ra)
P = 2 #number of tasks
N_batch = 2 #number of initial conditions/trajectories

f_barcode = 0.7 #barcode sparsity (when mechanism == "barcode")
theta = -1. #threshold
g_I = 2. #inhibition strength
s = 2. #rule input strength (when mechanism == "barcode")

dt = 0.05
T_sim = 60.

sigma = np.zeros((R, R, P)) #RxR connectivity overlap matrix for each task
sigma[:, :, 0] = np.array([[1.6, 0.8], [-0.8, 1.6]]) #task A: limit cycle 
sigma[:, :, 1] = np.array([[1.8, 0.], [0., 0.]]) #task B: bistable

conn_seed = 0

if mechanism == "transient":
    batches_to_plot = [2,1,0]
    alpha = np.array([1., 1.])

    m, n = connectivity_vectors(sigma,Ra,N,seed=conn_seed)
    m1 = m2 = m3 = m
    n1 = n2 = n3 = n
    pulse1 = pulse2 = pulse3 = 0
    taskA_vecs = m[0,:,:]
    taskB_vec = m[1,:,0]

    x1_0 = np.zeros([N_batch,N])
    x1_0[0,:] = 0.25*taskB_vec
    x1_0[1,:] = -0.25*taskB_vec

    x2_0 = np.zeros([N_batch,N])
    x2_0[0,:] = (taskA_vecs[:,0] + taskA_vecs[:,1])/2
    x3_0 = None

    mx1 = theta*np.ones([2,1])
    mx2 = theta*np.ones([2,1])
    Niter = 100
    eta = 0.05
    for ii in range(Niter):
        mx1 += eta*(-mx1 - g_I*np.mean(phi(x1_0 + mx1,theta),axis=1)[:,np.newaxis])
        mx2 += eta*(-mx2 - g_I*np.mean(phi(x2_0 + mx2,theta),axis=1)[:,np.newaxis])
    x1_0 += mx1
    x2_0 += mx2

elif mechanism == "connectivity":
    batches_to_plot = [2,2,2]
    m, n = connectivity_vectors(sigma,Ra,N,seed=conn_seed)
    alpha_sel = 1.
    alpha_unsel = 0.5
    m1 = alpha_sel*np.copy(m)
    m2 = alpha_sel*np.copy(m)
    m3 = alpha_sel*np.copy(m)
    n1 = n2 = n3 = n
    m1[1,:,:] = alpha_unsel*m[1,:,:]
    m2[0,:,:] = alpha_unsel*m[0,:,:]
    m3[1,:,:] = alpha_unsel*m[1,:,:]
    pulse1,pulse2,pulse3 = [0,1,0]
    x1_0 = np.random.randn(N_batch,N)
    x2_0 = x3_0 = None #use end state of previous epoch

elif mechanism == "barcode":
    batches_to_plot = [2,2,2]
    sigma[:, :, 0] = np.array([[1.75, 1.0], [-1.0, 1.75]])
    sigma[:, :, 1] = np.array([[1.7, 0.], [0., 0.]])
    m, n, barcodes = connectivity_vectors(sigma,Ra,N,f_barcode,seed=conn_seed)
    theta = 1.
    rule1 = s * barcodes[0]
    rule2 = s * barcodes[1]
    rule3 = np.maximum(rule1, rule2)
    m1 = m2 = m3 = m/f_barcode
    n1 = n2 = n3 = n
    pulse1,pulse2,pulse3 = [0,1,0]
    x1_0 = np.random.randn(N_batch,N)
    x2_0 = x3_0 = None #use end state of previous epoch
else:
    raise ValueError("invalid mechanism")

taskA_vecs = m[0,:,:]
taskB_vecs = m[1,:,:]

#pulse perturbation
N_steps = int(T_sim/dt)
pulse_pattern = np.zeros(N_steps)
pulse_pattern[0:100] = np.exp(-np.arange(100)/10)
pulse_dirs = np.vstack([taskB_vecs[:,0],-taskB_vecs[:,0]])
I = pulse_pattern[None, None, :] * pulse_dirs[:, :, None]
if mechanism == "barcode":
    I1 = rule1[np.newaxis,:,np.newaxis] + pulse1 * I
    I2 = rule2[np.newaxis,:,np.newaxis] + pulse2 * I
    I3 = rule3[np.newaxis,:,np.newaxis] + pulse3 * I
else:
    I1 = pulse1 * I
    I2 = pulse2 * I
    I3 = pulse3 * I


###simulate
x1, phi1,_,_ = simulate_rnn(m=m1, n=n1, T_sim=T_sim, dt=dt, theta=theta, g_I=g_I, x0=x1_0, I=I1)

if batches_to_plot[1] > 0:
    if x2_0 is None:
        x2_0 = x1[-1]
    x2,phi2,sel_task_len,dotprods = simulate_rnn(m=m2, n=n2, T_sim=T_sim, dt=dt, theta=theta, g_I=g_I, x0=x2_0, I=I2,decay_thresh=0.25)
else:
    phi2 = np.zeros([0,N_batch,N])

if batches_to_plot[2] > 0:
    if x3_0 is None:
        x3_0 = x2[-1]
    x3,phi3,_,_ = simulate_rnn(m=m3, n=n3, T_sim=T_sim, dt=dt, theta=theta, g_I=g_I, x0=x3_0, I=I3)
else:
    x3 = np.zeros([0,N_batch,N])
    phi3 = np.zeros([0,N_batch,N])

x = np.concatenate([x1, x2, x3], axis=0)
K = np.einsum("tnr,sbn->tbsr", m, x) #low-dimensional dynamics array, with indices [task,batch,timepoint,dimension within task]

###3d plots
epoch_edges = [0, N_steps, 2 * N_steps, 3 * N_steps]

batch_colors = ['black', 'grey'] 
taskAcol = '#3232CC'
taskBcol = '#A3145F' 
markersize=15
markerlw=0.5
fig = plt.figure(figsize=(16/2.54,5.5/2.54))
axes = []
for ei in range(3):
    if batches_to_plot[ei] > 0:
        ax3 = fig.add_subplot(1, 3, ei+1, projection="3d")
    for bi in range(batches_to_plot[ei]):
        s0, s1 = epoch_edges[ei], epoch_edges[ei + 1]
        if mechanism == "transient" and ei == 1: #separate part of trajectory corresponding to transient interval and plot in solid vs. dashed
            sdecay = s0+int(sel_task_len[0]/dt)
            ax3.plot( K[0, bi, s0:sdecay, 0], K[0, bi, s0:sdecay, 1], K[1,bi, s0:sdecay,0], color=batch_colors[bi])
            ax3.plot( K[0, bi, sdecay:s1, 0], K[0, bi, sdecay:s1, 1], K[1,bi, sdecay:s1,0], color=batch_colors[bi],linestyle="dashed")
            ax3.scatter( K[0, bi, sdecay, 0], K[0, bi, sdecay, 1], K[1,bi, sdecay,0], color=taskAcol,marker='o',s=markersize/2,lw=markerlw) #task decay marker
 
        else:
            ax3.plot( K[0, bi, s0:s1, 0], K[0, bi, s0:s1, 1], K[1,bi, s0:s1,0], color=batch_colors[bi])

        ax3.scatter( K[0, bi, s1-1, 0], K[0, bi, s1-1, 1], K[1,bi, s1-1,0], color=batch_colors[bi],marker='o',s=markersize,lw=markerlw) #end marker
        ax3.scatter( K[0, bi, s0, 0], K[0, bi, s0, 1], K[1,bi, s0,0], marker='o', s=markersize, facecolors='white', edgecolors=batch_colors[bi],lw=markerlw) #start marker
        
    axes.append(ax3)

maxx = np.max(np.abs(K[0,:,:,0]))
maxy = np.max(np.abs(K[0,:,:,1]))
maxz = np.max(np.abs(K[1,:,:,0]))
for ax3 in axes:
    ax3.set_xlim(-maxx,maxx)
    ax3.set_ylim(-maxy,maxy)
    ax3.set_zlim(-maxz,maxz)
    ax3.set_box_aspect((1, 1, 1))
    ax3.grid(False) 
    ax3.set_axis_off()
    ax3.plot([0,0], [0,0], [0,1.5*maxz], color=taskBcol)
    ax3.plot([0,0], [0,-1.5*maxy], [-0,-0], color=taskAcol)
    ax3.plot([0,1.5*maxx], [0,0], [-0,-0], color=taskAcol)
    ax3.set_facecolor('none')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])

fig.tight_layout()
plt.show()

plt.savefig("figs/"+mechanism+".svg")

###selectivity calculation
Tstart = int(N_steps*0.5)
phiA = phi1[Tstart:,:,:]
phiB = phi2[Tstart:,:,:]
varA = np.std(phiA,axis=(0,1))
varB = np.std(phiB,axis=(0,1))
eps = 1e-12
sel = (varA-varB)/(varA+varB+eps)
np.savez_compressed("data/"+mechanism+"_sel.npz",sel=sel)
