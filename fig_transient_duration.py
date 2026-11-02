import numpy as np
import matplotlib.pyplot as plt
from utils import * 

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
T_sim = 120.
N_steps = int(T_sim/dt)

sigma = np.zeros((R, R, P)) #RxR connectivity overlap matrix for each task
sigma[:, :, 0] = np.array([[1.6, 0.8], [-0.8, 1.6]]) #task A: limit cycle
sigma[:, :, 1] = np.array([[1.7, 0.], [0., 0.]]) #task B: bistable

#generate connectivity
Nnet = 20

taskB_deltas = np.logspace(-2.,.5,20)
Nsim = len(taskB_deltas)
tasklen = np.zeros([Nnet,Nsim])

###
for ni in range(Nnet):
    m1, n1 = connectivity_vectors(sigma,Ra,N,seed=ni)
    I1 = np.zeros([N_batch,N,N_steps])

    taskA_vecs = m1[0,:,:]
    taskB_vec = m1[1,:,0]


    for si in range(Nsim):
        x1_0 = np.zeros([N_batch,N])
        x1_0[0,:] = (taskA_vecs[:,0] + taskA_vecs[:,1] + taskB_deltas[si]*taskB_vec)/2
        x1_0[1,:] = (taskA_vecs[:,0] + taskA_vecs[:,1] - taskB_deltas[si]*taskB_vec)/2
        mx1 = theta*np.ones([2,1])
        Niter = 100
        eta = 0.05
        for ii in range(Niter):
            mx1 += eta*(-mx1 - g_I*np.mean(phi(x1_0 + mx1,theta),axis=1)[:,np.newaxis])
        x1_0 += mx1

        x1, phi1,sel_task_len,dotprods = simulate_rnn(m=m1, n=n1, T_sim=T_sim, dt=dt, theta=theta, g_I=g_I, x0=x1_0, I=I1, sel_task_vecs=taskA_vecs)

        tasklen[ni,si] = np.min(sel_task_len)
        print("net",ni,"sim",si,tasklen[ni,si])

###

plt.figure(figsize=(4.5/2.54,4./2.54))
mtasklen = np.nanmean(tasklen,axis=0)
stdtasklen = np.nanstd(tasklen,axis=0)
errorscatter2(taskB_deltas,mtasklen,stdtasklen/np.sqrt(Nnet),connect=False)
plt.semilogx()
plt.xlim(0.008,5)
plt.ylim(0,80)
plt.yticks((0,25,50,75))
plt.ylabel("T",style="italic")
plt.xlabel("Task B projection")
plt.tight_layout()

plt.savefig("figs/task_duration.svg")
