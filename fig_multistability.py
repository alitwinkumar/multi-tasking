"""Ring-attractor projections and switching between a limit cycle and fixed points."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

output_dir = Path(__file__).resolve().parent / 'plots' / Path(__file__).stem
output_dir.mkdir(parents=True, exist_ok=True)


def savefig(name, **kwargs):
    fig = plt.gcf()
    stem = Path(name).stem
    fig.savefig(output_dir / (stem + '.pdf'), **kwargs)

def plot_rings():
    N = 10000

    np.random.seed(100)

    sigma1 = 2.5
    sigma2= 2.3

    thetas = 2*np.pi*np.arange(N)/N
    thetas1=  np.random.permutation(thetas)[:,np.newaxis]
    thetas2= np.random.permutation(thetas)[:,np.newaxis]

    def phi(x):
        return(np.maximum(x+1,0))

    def dyns_step(x):
        c1 = np.cos(thetas1[:,0])
        s1 = np.sin(thetas1[:,0])
        c2 = np.cos(thetas2[:,0])
        s2 = np.sin(thetas2[:,0])
        dx = -x
        r = phi(x)
        for vec, sigma in zip([c1, s1, c2, s2], [sigma1, sigma1, sigma2, sigma2]):
            dx += sigma*vec*np.mean(vec*r)
        return dx

    def cart_to_polar(Ks):
        Nt = np.shape(Ks)[1]
        Rhos = np.zeros((2,Nt))
        Thetas = np.zeros((2,Nt))
        Rhos[0,:] = np.sqrt(Ks[0,:]**2+Ks[1,:]**2)
        Rhos[1,:] = np.sqrt(Ks[2,:]**2+Ks[3,:]**2)
        Thetas[0,:]= np.arctan2(Ks[1,:], Ks[0,:])
        Thetas[1,:]= np.arctan2(Ks[3,:], Ks[2,:])
        return(Rhos, Thetas)

    T = 60
    dt = 0.1
    time = np.arange(0, T, step=dt)

    m1 = np.sqrt(2)*np.cos(thetas1)
    m2 = np.sqrt(2)*np.sin(thetas1)
    m3 = np.sqrt(2)*np.cos(thetas2)
    m3 = m3-np.mean(m1*m3)*m1-np.mean(m2*m3)*m2
    m4 = np.sqrt(2)*np.sin(thetas2)
    m4 = m4-np.mean(m1*m4)*m1-np.mean(m2*m4)*m2
    Ms = np.hstack((m1, m2, m3, m4))

    Ks = np.zeros((4, len(time)))

    Trials = 30

    KKs = np.zeros((4, len(time), Trials))
    RRhos   = np.zeros((2, len(time), Trials))
    TThetas = np.zeros((2, len(time), Trials))
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for trials in range(Trials):
        x= 2*np.random.randn(N)

        thet1 = 2*np.pi*trials/(Trials//2)+0.14*np.random.rand()*2*np.pi
        if trials<Trials//2:
            rho1 = (0.6+0.4*np.random.randn())+(np.random.rand()>0.8)*(1+0.3*np.random.randn())

            x += rho1*np.cos(thet1)*m1[:,0]+rho1*np.sin(thet1)*m2[:,0]+0.1*(np.random.randn()*m3[:,0]+np.random.randn()*m4[:,0])
        else:
            rho1 = (0.6+0.15*np.random.randn())+(np.random.rand()>0.8)*(0.95+0.15*np.random.randn())
            x += rho1*np.cos(thet1)*m3[:,0]+rho1*np.sin(thet1)*m4[:,0]+0.2*(np.random.randn()*m1[:,0]+np.random.randn()*m2[:,0])

        for it, t in enumerate(time):
            Ks[:,it] = Ms.T.dot(x)*(1/N)
            x = x+dt*dyns_step(x)
        Rhos, Thetas = cart_to_polar(Ks)

        ax1.plot(Rhos[0,:], c='C0')
        ax1.plot(Rhos[1,:], c='C1')

        ax2.plot(Thetas[0,:], c='C0')
        ax2.plot(Thetas[1,:], c='C1')

        KKs[:,:,trials] = Ks
        RRhos[:,:,trials]= Rhos
        TThetas[:,:,trials]=Thetas

    plt.style.use('ggplot')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig_width = 1.5*2.2 # width in inches
    fig_height = 1.5*2  # height in inches
    fig_size =  [fig_width,fig_height]
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.autolayout'] = True

    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['lines.markeredgewidth'] = 0.003
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['font.size'] = 14#9
    plt.rcParams['legend.fontsize'] = 11#7.
    plt.rcParams['axes.facecolor'] = '1'
    plt.rcParams['axes.edgecolor'] = '0'
    plt.rcParams['axes.linewidth'] = '0.7'

    plt.rcParams['axes.labelcolor'] = '0'
    plt.rcParams['axes.labelsize'] = 14#9
    plt.rcParams['xtick.labelsize'] = 11#7
    plt.rcParams['ytick.labelsize'] = 11#7
    plt.rcParams['xtick.color'] = '0'
    plt.rcParams['ytick.color'] = '0'
    plt.rcParams['xtick.major.size'] = 2
    plt.rcParams['ytick.major.size'] = 2

    plt.rcParams['font.sans-serif'] = 'Arial'

    def clean_axis(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        return()

    c1 = '#9F2D69'
    c2 = '#3332CB'

    fig = plt.figure(figsize=[1.5*2.2, 1.5*2])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    clean_axis(ax1)

    first = True
    for trials in range(Trials):
        if RRhos[0,-1,trials]>RRhos[1,-1,trials]:
            c=c1

            ax1.plot(KKs[0,:,trials], KKs[1,:,trials], c=c1)
            ax1.scatter(KKs[0,0,trials], KKs[1,0,trials], edgecolor=c1, color='w' , zorder=5, s=20)
            ax1.scatter(KKs[0,-1,trials], KKs[1,-1,trials], color='k' , zorder=7, s=20)

            ax2.plot(KKs[2,:,trials], KKs[3,:,trials], c=c2)
            ax2.scatter(KKs[2,0,trials], KKs[3,0,trials], edgecolor=c2, color='w' , zorder=5, s=20)
            ax2.scatter(KKs[2,-1,trials], KKs[3,-1,trials], color='k' , zorder=7, s=20)

    LS = 2

    ax1.set_xlim([-LS, LS])
    ax2.set_xlim([-LS, LS])
    ax1.set_ylim([-LS, LS])
    ax2.set_ylim([-LS, LS])

    ax1.axis('off')
    ax2.axis('off')
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_box_aspect(1)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_box_aspect(1)

    savefig('multistable_rings_rings1.pdf', transparent=True,)

    c1 = '#9F2D69'
    c2 = '#3332CB'

    fig = plt.figure(figsize=[1.5*2.2, 1.5*2])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    clean_axis(ax1)

    first = True
    for trials in range(Trials):
        if RRhos[0,-1,trials]<RRhos[1,-1,trials]:
            c=c1

            ax1.plot(KKs[0,:,trials], KKs[1,:,trials], c=c1)
            ax1.scatter(KKs[0,0,trials], KKs[1,0,trials], edgecolor=c1, color='w' , zorder=5, s=20)
            ax1.scatter(KKs[0,-1,trials], KKs[1,-1,trials], color='k' , zorder=7, s=20)

            ax2.plot(KKs[2,:,trials], KKs[3,:,trials], c=c2)
            ax2.scatter(KKs[2,0,trials], KKs[3,0,trials], edgecolor=c2, color='w' , zorder=5, s=20)
            ax2.scatter(KKs[2,-1,trials], KKs[3,-1,trials], color='k' , zorder=7, s=20)

    LS = 2

    ax1.set_xlim([-LS, LS])
    ax2.set_xlim([-LS, LS])
    ax1.set_ylim([-LS, LS])
    ax2.set_ylim([-LS, LS])
    ax1.axis('off')
    ax2.axis('off')
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_box_aspect(1)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_box_aspect(1)
    savefig('multistable_rings_rings2.pdf', transparent=True,)



def plot_switch():
    N = 1000

    np.random.seed(20)

    phase_shift = 0.35*np.pi
    sigma1 = 2.2/np.abs(np.cos(phase_shift))#5.9
    sigma2= 2.6#3.0

    thetas = 2*np.pi*np.arange(N)/N
    thetas1=  np.random.permutation(thetas)[:,np.newaxis]
    thetas2=  np.random.permutation(thetas)[:,np.newaxis]

    def phi(x):
        return(np.maximum(x+1,0))

    def dyns_step(x):
        c1 = np.cos(thetas1[:,0])
        s1 = np.sin(thetas1[:,0])
        c2 = np.cos(thetas2[:,0])
        cp = np.cos(phase_shift)
        sp = np.sin(phase_shift)

        r = phi(x)

        dx = -x
        dx += sigma1*c1*np.mean((cp*c1-s1*sp)*r)
        dx += sigma1*s1*np.mean((cp*s1+c1*sp)*r)
        dx += sigma2*c2*np.mean(c2*r)
        return dx

    def cart_to_polar(Ks):
        Nt = np.shape(Ks)[1]
        Rhos = np.zeros((2,Nt))
        Thetas = np.zeros((1,Nt))
        Rhos[0,:] = np.sqrt(Ks[0,:]**2+Ks[1,:]**2)
        Rhos[1,:] = np.sqrt(Ks[2,:]**2)
        Thetas[0,:]= np.arctan2(Ks[1,:], Ks[0,:])
        return(Rhos, Thetas)

    T = 60
    dt = 0.1
    time = np.arange(0, T, step=dt)

    m1 = np.sqrt(2)*np.cos(thetas1)
    m2 = np.sqrt(2)*np.sin(thetas1)
    m3 = np.sqrt(2)*np.cos(thetas2)
    m3 = m3-np.mean(m1*m3)*m1-np.mean(m2*m3)*m2

    Ms = np.hstack((m1, m2, m3))

    plt.style.use('ggplot')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig_width = 1.5*2.2 # width in inches
    fig_height = 1.5*2  # height in inches
    fig_size =  [fig_width,fig_height]
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.autolayout'] = True

    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['lines.markeredgewidth'] = 0.003
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['font.size'] = 14#9
    plt.rcParams['legend.fontsize'] = 11#7.
    plt.rcParams['axes.facecolor'] = '1'
    plt.rcParams['axes.edgecolor'] = '0'
    plt.rcParams['axes.linewidth'] = '0.7'

    plt.rcParams['axes.labelcolor'] = '0'
    plt.rcParams['axes.labelsize'] = 14#9
    plt.rcParams['xtick.labelsize'] = 11#7
    plt.rcParams['ytick.labelsize'] = 11#7
    plt.rcParams['xtick.color'] = '0'
    plt.rcParams['ytick.color'] = '0'
    plt.rcParams['xtick.major.size'] = 2
    plt.rcParams['ytick.major.size'] = 2

    plt.rcParams['font.sans-serif'] = 'Arial'

    def clean_axis(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        return()

    cc1 = '#9F2D69'
    cc2 = '#3332CB'
    T = 45
    dt = 0.05
    time = np.arange(0, T, step=dt)

    Trials = 2
    np.random.seed(10)
    KKs = np.zeros((3, len(time), Trials))
    RRhos   = np.zeros((2, len(time), Trials))
    TThetas = np.zeros((1, len(time), Trials))

    val = 1.2
    x0= np.random.randn(N)

    x0 += 0.5*np.random.randn()*m1[:,0]+0.5*np.random.randn()*m2[:,0]

    Ks = np.zeros((3, len(time)))
    inpIND = 395
    for trials in range(Trials):
        x = x0
        if trials==0:
            Inp = val*m3[:,0]
        else:
            Inp = -val*m3[:,0]

        for it, t in enumerate(time):
            Ks[:,it] = Ms.T.dot(x)*(1/N)

            x = x+dt*dyns_step(x)
            if it==inpIND:
                x+=Inp
        Rhos, Thetas = cart_to_polar(Ks)

        KKs[:,:,trials] = Ks
        RRhos[:,:,trials]= Rhos
        TThetas[:,:,trials]=Thetas

    time_mask = time<60

    fig = plt.figure(figsize=[1.5*2.2, 1.5*2])
    ax = fig.add_subplot(111, projection="3d")

    for itera in range(3):
        KX = np.zeros((3, 2))
        KX[itera,:] = [0, 2.2]
        if itera<2:
            c=cc2
        else:
            c=cc1
        ax.plot(KX[0,:], KX[1,:], KX[2,:], c=c, zorder=0)

    for trials in [1,0]:
        if trials==0:
            col = 0.0*np.ones(3)
        else:
            col = 0.4*np.ones(3)
        ax.plot(KKs[0,time_mask,trials], KKs[1,time_mask,trials], KKs[2,time_mask,trials], c=col, lw=1, zorder=1)

        ax.scatter(KKs[0,time_mask,trials][-1], KKs[1,time_mask,trials][-1], KKs[2, time_mask, trials][-1], edgecolor=col, color=col , zorder=50, s=30)
        ax.scatter(KKs[0,0,trials], KKs[1,0,trials], KKs[2, 0, trials], edgecolor=col, color='w' , zorder=60, s=30)
        ax.scatter(KKs[0,inpIND,trials], KKs[1,inpIND,trials], KKs[2, inpIND, trials], edgecolor=col, color='C1' , zorder=60, s=30)
        ax.scatter(KKs[0,inpIND+1,trials], KKs[1,inpIND+1,trials], KKs[2, inpIND+1, trials], edgecolor=col, color='C1' , zorder=60, s=30)

    theta = np.linspace(0, 2*np.pi, 100)

    ax.axis('off')
    ax.view_init(elev=11.6, azim=31.42)
    ax.set_box_aspect(aspect=(4, 4, 3), zoom=1.9)

    savefig('multistable_3d_input.pdf')

    fig = plt.figure(figsize=[1.3*2.5, 2])
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    for trials in [1,0]:
        if trials==0:
            col = cc2
            alp=1
        else:
            col = "#818BD3"
            alp=1

        ax.plot(time[time_mask], KKs[0,time_mask,trials], c=col, lw=1., zorder=1, alpha=alp)
        ax.scatter(time[time_mask][-1], KKs[0,time_mask,trials][-1], edgecolor=col, color=col , zorder=50, s=30, alpha=alp)
        ax.scatter(time[time_mask][0], KKs[0,time_mask,trials][0], edgecolor=col, color='w' , zorder=50, s=30)
        ax.scatter(time[inpIND], KKs[0,inpIND,trials], edgecolor=col, color='C1' , zorder=50, s=20)
        ax.scatter(time[inpIND+1], KKs[0,inpIND+1,trials], edgecolor=col, color='C1' , zorder=50, s=20)

    YL = ax.get_ylim()
    ax.plot([time[inpIND], time[inpIND]], YL, lw=6, alpha=0.4, c='C1')
    ax.set_xlabel('time')
    ax.set_ylabel('output task A')

    savefig('multistable_3d_input2_A.pdf', transparent=True)

    fig = plt.figure(figsize=[1.3*2.5, 2])
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    for trials in [1,0]:
        if trials==0:
            col = cc1#0.0*np.ones(3)
            alp = 1
        else:
            col = "#C995B4"
            alp = 1
        ax.plot(time[time_mask], KKs[2,time_mask,trials], c=col, lw=1., zorder=1, alpha=alp)
        ax.scatter(time[time_mask][-1], KKs[2,time_mask,trials][-1], edgecolor=col, color=col , zorder=50, s=30, alpha=alp)
        ax.scatter(time[time_mask][0], KKs[2,time_mask,trials][0], edgecolor=col, color='w' , zorder=50, s=30)
        ax.scatter(time[inpIND], KKs[2,inpIND,trials], edgecolor=col, color='C1' , zorder=50, s=20)
        ax.scatter(time[inpIND+1], KKs[2,inpIND+1,trials], edgecolor=col, color='C1' , zorder=50, s=20)

    ax.set_xlabel('time')
    ax.set_ylabel('output task B')

    ax.plot([time[inpIND], time[inpIND]], YL, lw=6, alpha=0.4, c='C1')

    savefig('multistable_3d_input2_B.pdf', transparent=True)



if __name__ == '__main__':
    plot_rings()
    plot_switch()

plt.show()
