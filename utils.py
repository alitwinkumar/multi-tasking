import numpy as np

def phi(x,theta):
    return np.maximum(0.0, x-theta)

def connectivity_vectors(sigma, Ra, N, f=1., seed=0):
    #returns low-rank connectivity vectors m and n, and barcodes if f<1
    rng = np.random.default_rng(seed)

    R = sigma.shape[0]
    P = sigma.shape[2]
    sigma_mm = np.eye(R)
    sigma_nn = np.eye(R)

    loadings = np.zeros((P, N, 2 * R), dtype=float)
    for mu in range(P):
        sigma_mn = sigma[:, :, mu]
        cov = np.block([[sigma_mm, sigma_mn], [sigma_mn.T, sigma_nn]])
        while np.min(np.linalg.eigvalsh(cov)) < 0:
            sigma_nn = sigma_nn * 1.1
            cov = np.block([[sigma_mm, sigma_mn], [sigma_mn.T, sigma_nn]])
        loadings[mu] = rng.multivariate_normal(mean=np.zeros(2 * R), cov=cov, size=N)
        for ri in range(Ra[mu],R): #zero out vectors when task dimension is smaller than max dimension R
            loadings[mu,:,ri] = 0
            loadings[mu,:,R+ri] = 0

    if f < 1:
        barcodes = (rng.random((P, N)) < f).astype(float)
        loadings[:, :, :R] *= barcodes[:, :, None]
        loadings[:, :, R:] *= barcodes[:, :, None]
    m = loadings[:, :, :R]
    n = loadings[:, :, R:]

    if f < 1:
        return m, n, barcodes
    else:
        return m, n

def simulate_rnn(m, n, T_sim, dt, theta, g_I, x0, I, sel_task_vecs=None, stop_when_decayed=False,decay_thresh=0.25):
    #simulates low-rank RNN
    rng = np.random.default_rng()
    N = m.shape[1]
    N_batch = x0.shape[0]

    if sel_task_vecs is None:
        sel_task_vecs = m[0,:,:]

    m_flat = m.transpose(1, 0, 2).reshape(N, -1)
    n_flat = n.transpose(1, 0, 2).reshape(N, -1)

    N_steps = int(T_sim / dt)

    x = np.copy(x0)

    x_save = np.zeros((N_steps, x.shape[0], N), dtype=float)
    phi_save = np.zeros_like(x_save)
    dotprod_save = np.zeros([N_steps,N_batch])

    def drift(x_state):
        phi_state = phi(x_state,theta)
        return - x_state + (phi_state @ n_flat) @ m_flat.T/N - g_I * np.mean(phi_state, axis=1, keepdims=True)

    sel_task_len = np.zeros(N_batch)
    sel_dotprod_0 = np.zeros(N_batch)
    for bi in range(N_batch):
        sel_dotprod_0[bi] = np.linalg.norm(x[bi,:] @ sel_task_vecs)

    for ti in range(N_steps):
        k1 = drift(x)
        k2 = drift(x + 0.5 * dt * k1)
        k3 = drift(x + 0.5 * dt * k2)
        k4 = drift(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        x = x + dt * I[:, :, ti]

        #track dot product of state with selected task vectors
        sel_dotprod = np.zeros(N_batch)
        decayed = np.zeros(N_batch,dtype=np.bool)
        for bi in range(N_batch):
            sel_dotprod[bi] = np.linalg.norm(x[bi,:] @ sel_task_vecs)
            if sel_dotprod[bi] > decay_thresh*sel_dotprod_0[bi]:
                sel_task_len[bi] = ti*dt
            else:
                decayed[bi] = True
        if np.any(decayed) and stop_when_decayed: #stop sim
            break

        x_save[ti] = x
        phi_save[ti] = phi(x,theta)
        dotprod_save[ti,:] = sel_dotprod 

    return x_save, phi_save, sel_task_len, dotprod_save
