"""Figure 7A and E: barcode blocks, twelve-task readouts, and neuron activity."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

output_dir = Path(__file__).resolve().parent / 'plots' / Path(__file__).stem
output_dir.mkdir(parents=True, exist_ok=True)


def savefig(name, **kwargs):
    fig = plt.gcf()
    stem = Path(name).stem
    fig.savefig(output_dir / (stem + '.pdf'), **kwargs)


seed = 11
rng = np.random.default_rng(seed)

N = 6000
f = 0.3
delta_m = 0.1
theta = -0.3
selection_strength = -0.6
sequence_start_impulse = 1.0
dt = 0.04
tau = 1.0
epoch_T = 21.0
gI = 1.6
x0_scale = 0.03

lc_params = [
    {'sigma': 1.5, 'omega': 0.6},
    {'sigma': 1.55, 'omega': 1.},
    {'sigma': 1.6, 'omega': 1.5},
    {'sigma': 1.65, 'omega': 1.8},
]

seq_params = [
    {'K': 8,  'gain': 1.4},
    {'K': 10,  'gain': 1.45},
    {'K': 12, 'gain': 1.5},
    {'K': 14, 'gain': 1.55},
]

fp_params = [
    {'sigma': 1.5},
    {'sigma': 1.6},
    {'sigma': 1.65},
    {'sigma': 1.7},
]

def phi(x):
    return np.maximum(0.0, x)

def gaussian_patterns(rng, N, K):
    return rng.normal(size=(N, K))

def orthogonalize_pair(patterns):
    q, _ = np.linalg.qr(patterns, mode='reduced')
    return np.sqrt(patterns.shape[0]) * q

def make_exact_mask(rng, N, f):
    mask = np.zeros(N, dtype=bool)
    active = rng.choice(N, size=int(round(f * N)), replace=False)
    mask[active] = True
    return mask

def apply_barcode_to_patterns(raw, mask, delta_m):
    m_barcode = np.where(mask, 1.0, delta_m)
    n_barcode = mask.astype(float)
    m_patterns = raw * m_barcode[:, None]
    n_patterns = raw * n_barcode[:, None]
    return m_patterns, n_patterns

def pattern_readout(patterns, y):
    norm = np.sum(patterns * patterns, axis=0)
    norm = np.maximum(norm, 1e-12)
    return patterns.T @ y / norm

def n_readout(task, y):
    return task['n_patterns'].T @ y / (f * N)

def global_inhibition(r, gI):
    return gI * np.mean(r)

def make_limit_cycle_task(rng, N, f, delta_m, sigma, omega):
    mask = make_exact_mask(rng, N, f)
    raw = orthogonalize_pair(gaussian_patterns(rng, N, 2))
    m_patterns, n_patterns = apply_barcode_to_patterns(raw, mask, delta_m)
    return {
        'kind': 'lc',
        'mask': mask,
        'm_patterns': m_patterns,
        'n_patterns': n_patterns,
        'readout_patterns': m_patterns,
        'sigma': sigma,
        'omega': omega,
    }

def make_sequence_task(rng, N, f, delta_m, K, gain):
    mask = make_exact_mask(rng, N, f)
    raw = gaussian_patterns(rng, N, K)
    m_patterns, n_patterns = apply_barcode_to_patterns(raw, mask, delta_m)
    return {
        'kind': 'seq',
        'mask': mask,
        'm_patterns': m_patterns,
        'n_patterns': n_patterns,
        'readout_patterns': m_patterns,
        'gain': gain,
    }

def make_fixed_point_task(rng, N, f, delta_m, sigma):
    mask = make_exact_mask(rng, N, f)
    raw = gaussian_patterns(rng, N, 1)
    m_patterns, n_patterns = apply_barcode_to_patterns(raw, mask, delta_m)
    return {
        'kind': 'fp',
        'mask': mask,
        'm_patterns': m_patterns,
        'n_patterns': sigma * n_patterns,
        'readout_patterns': m_patterns,
        'sigma': sigma,
    }

def build_tasks(rng, N, f, delta_m):
    tasks = []
    labels = []

    for i, par in enumerate(lc_params):
        tasks.append(make_limit_cycle_task(rng, N, f, delta_m, **par))
        labels.append(f'LC {i + 1}')

    for i, par in enumerate(seq_params):
        tasks.append(make_sequence_task(rng, N, f, delta_m, **par))
        labels.append(f'Seq {i + 1}') #(K={par["K"]})')

    for i, par in enumerate(fp_params):
        tasks.append(make_fixed_point_task(rng, N, f, delta_m, **par))
        labels.append(f'FP {i + 1}')

    return tasks, labels

def limit_cycle_recurrent_input(task, r):
    z = n_readout(task, r)
    A = np.array([
        [task['sigma'], -task['omega']],
        [task['omega'],  task['sigma']],
    ])
    return task['m_patterns'] @ (A @ z)

def sequence_recurrent_input(task, r):
    q = n_readout(task, r)
    return task['gain'] * (task['m_patterns'][:, 1:] @ q[:-1])

def fixed_point_recurrent_input(task, r):
    amp = task['n_patterns'][:, 0] @ r / (f * N)
    return task['m_patterns'][:, 0] * amp

def recurrent_input(tasks, r):
    rec = np.zeros_like(r)
    for task in tasks:
        if task['kind'] == 'lc':
            rec += limit_cycle_recurrent_input(task, r)
        elif task['kind'] == 'seq':
            rec += sequence_recurrent_input(task, r)
        elif task['kind'] == 'fp':
            rec += fixed_point_recurrent_input(task, r)
        else:
            raise ValueError(task['kind'])
    return rec

def selection_schedule(n_tasks):
    singles = [(i,) for i in range(n_tasks)]

    pairs = [
        (0, 3),   # LC1 with LC4
        (1, 2),   # LC2 with LC3
        (0, 4),   # LC1 with SEQ1
        (3, 7),   # LC4 with SEQ4
        (0, 8),   # LC1 with FP1
        (3, 11),  # LC4 with FP4
        (4, 7),   # SEQ1 with SEQ4
        (5, 6),   # SEQ2 with SEQ3
        (4, 8),   # SEQ1 with FP1
        (7, 11),  # SEQ4 with FP4
        (8, 11),  # FP1 with FP4
        (9, 10),  # FP2 with FP3
    ]
    return singles + pairs

def selected_union_mask(tasks, selected):
    mask = np.zeros(tasks[0]['mask'].shape, dtype=bool)
    for task_idx in selected:
        mask |= tasks[task_idx]['mask']
    return mask

def selection_threshold_input(tasks, selected, selection_strength):
    return -selection_strength * selected_union_mask(tasks, selected).astype(float)

def selection_counts(tasks, schedule):
    counts = np.zeros(tasks[0]['mask'].shape, dtype=int)
    for selected in schedule:
        counts += selected_union_mask(tasks, selected).astype(int)
    return counts

def selected_neurons(tasks, schedule):
    # Original displayed cells, preserving the NumPy 1.26 sort order of tied counts.
    neuron_ids = np.array([3233, 4803, 5868, 1099, 27, 1265, 3221, 4766,
                           86, 2981, 2227, 3085, 1445, 3165, 1235, 606])
    counts = selection_counts(tasks, schedule)
    if len(counts) <= neuron_ids.max():
        raise ValueError('Choose new neuron_ids when changing the network size.')
    groups = [f'neuron {i}' for i in neuron_ids]
    return neuron_ids, groups, counts[neuron_ids]

def sequence_start_input(tasks, selected, sequence_start_impulse):
    start = np.zeros(tasks[0]['mask'].shape)
    n_seq = 0
    for task_idx in selected:
        task = tasks[task_idx]
        if task['kind'] != 'seq':
            continue
        start += sequence_start_impulse * task['m_patterns'][:, 0] / dt
        n_seq += 1
    if n_seq > 1:
        start /= np.sqrt(n_seq)
    return start

def rates(x, theta):
    return phi(x - theta)

def task_readouts(tasks, x, r):
    out = []
    details = []
    for task in tasks:
        source = r if task['kind'] in ('seq', 'fp') else x
        z = pattern_readout(task['readout_patterns'], source)
        details.append(z)
        if task['kind'] == 'seq':
            out.append(np.max(z))
        else:
            out.append(z[0])
    return np.array(out), details

def detail_layout(tasks):
    layout = []
    cursor = 0
    for task in tasks:
        width = task['readout_patterns'].shape[1]
        layout.append(slice(cursor, cursor + width))
        cursor += width
    return layout, cursor

def simulate_combined_network(tasks, schedule, seed=100):
    rng = np.random.default_rng(seed)
    steps_per_epoch = int(round(epoch_T / dt))
    total_steps = steps_per_epoch * len(schedule) + 1
    t = np.arange(total_steps) * dt

    x = x0_scale * rng.normal(size=N)
    layout, n_detail = detail_layout(tasks)
    readouts = np.zeros((len(tasks), total_steps))
    detail_readouts = np.zeros((n_detail, total_steps))
    traced_neuron_ids, neuron_trace_groups, neuron_selection_counts = selected_neurons(tasks, schedule)
    neuron_rate_traces = np.zeros((len(traced_neuron_ids), total_steps), dtype=np.float32)
    neuron_selected_trace = np.zeros((len(traced_neuron_ids), total_steps), dtype=bool)
    epoch_edges = [0]

    sel_input = np.zeros(N)
    r = rates(x, theta)
    readouts[:, 0], details = task_readouts(tasks, x, r)
    detail_readouts[:, 0] = np.concatenate(details)
    neuron_rate_traces[:, 0] = r[traced_neuron_ids]

    write_idx = 1
    for selected in schedule:
        sel_input = selection_threshold_input(tasks, selected, selection_strength)
        seq_start = sequence_start_input(tasks, selected, sequence_start_impulse)
        for epoch_step in range(steps_per_epoch):
            r = rates(x, theta)
            rec = recurrent_input(tasks, r)
            inh = global_inhibition(r, gI)
            external = seq_start if epoch_step == 0 else 0.0
            dx = -x + rec - inh + external + sel_input
            x = x + (dt / tau) * dx

            r = rates(x, theta)
            readouts[:, write_idx], details = task_readouts(tasks, x, r)
            detail_readouts[:, write_idx] = np.concatenate(details)
            neuron_rate_traces[:, write_idx] = r[traced_neuron_ids]
            neuron_selected_trace[:, write_idx] = sel_input[traced_neuron_ids] > 0.0
            write_idx += 1
        epoch_edges.append(write_idx - 1)

    return (t, readouts, detail_readouts, layout, traced_neuron_ids, neuron_trace_groups,
            neuron_selection_counts, neuron_rate_traces, neuron_selected_trace, epoch_edges)

def task_highlight_color(task):
    return {'lc': '#9ecae1', 'seq': '#f6b26b', 'fp': "#D05C4B"}[task['kind']]

def selected_time_mask(schedule, epoch_edges, item_idx, total_steps):
    mask = np.zeros(total_steps, dtype=bool)
    for epoch_idx, selected in enumerate(schedule):
        if item_idx in selected:
            i0 = epoch_edges[epoch_idx]
            i1 = epoch_edges[epoch_idx + 1]
            mask[i0:i1 + 1] = True
    return mask

def plot_selected_curve_segments(ax, t, y, selected_mask, color, lw=5.0, alpha=0.28):
    y_selected = np.where(selected_mask, y, np.nan)
    ax.plot(t, y_selected, color=color, lw=lw, alpha=alpha, solid_capstyle='round', zorder=1)

def plot_offset_task_readouts(t, tasks, labels, detail_readouts, layout, schedule, epoch_edges):
    fig, ax = plt.subplots()
    offsets = -4.5 * np.arange(len(tasks))
    cmaps = {'lc': 'Blues', 'seq': 'magma', 'fp': 'autumn'}
    for task_idx, task in enumerate(tasks):
        Y = detail_readouts[layout[task_idx]]
        if task['kind'] == 'seq' and Y.shape[0] > 2:
            Y = Y[2:]
        spread = np.percentile(Y, 95) - np.percentile(Y, 5)
        if spread <= 1e-12:
            spread = np.std(Y)
        if spread <= 1e-12:
            spread = 1.0
        cmap = plt.get_cmap(cmaps[task['kind']])
        selected_mask = selected_time_mask(schedule, epoch_edges, task_idx, len(t))
        for k in range(Y.shape[0]):
            color = cmap(0.25 + 0.65 * k / max(Y.shape[0] - 1, 1))

            if task['kind']=='fp':
                y = 2*Y[k] / spread + offsets[task_idx]

            else:
                y = Y[k] / spread + offsets[task_idx]
            ax.plot(t, y, color=color, lw=0.8, zorder=2)
            plot_selected_curve_segments(ax, t, y, selected_mask, task_highlight_color(task), lw=4.0, alpha=0.35)
        ax.text(t[0]-10, offsets[task_idx], labels[task_idx], ha='right', va='center', fontsize=7, color=task_highlight_color2(task))

    for ische, task_idcs in enumerate(schedule):
        if np.mod(ische, 4)==0 and ische<13:
            label = ''
            for itt, task_idx in enumerate(task_idcs):
                task = tasks[task_idx]
                ax.text(ische*epoch_T +epoch_T//2, 5+3*itt, labels[task_idx], ha='center', va='center', fontsize=7, color=task_highlight_color2(task))
        elif np.mod(ische, 3)==0 and ische>13:
            label = ''
            for itt, task_idx in enumerate(task_idcs):
                task = tasks[task_idx]
                ax.text(ische*epoch_T +epoch_T//2, 5+3*itt, labels[task_idx], ha='center', va='center', fontsize=7, color=task_highlight_color2(task))
    for edge in epoch_edges:
        ax.axvline(t[edge], color='k', lw=0.25, alpha=0.2)

    ax.set_yticks(offsets)
    ax.set_yticklabels(labels)
    ax.set_xlabel('time')
    plt.axis('off')
    fig.tight_layout()

    return fig, ax


def plot_representative_neuron_traces(t, neuron_rate_traces, traced_neuron_ids, neuron_trace_groups,
                                      neuron_selection_counts, neuron_selected_trace, epoch_edges):
    n_traces = len(traced_neuron_ids)
    fig, ax = plt.subplots()
    offsets = 2.5 * np.arange(n_traces)
    for row, neuron_id in enumerate(traced_neuron_ids):
        y = neuron_rate_traces[row]
        spread = np.percentile(y, 95) - np.percentile(y, 5)
        if spread <= 1e-12:
            spread = np.std(y)
        if spread <= 1e-12:
            spread = 1.0
        y_plot = y / spread + offsets[row]
        ax.plot(t, y_plot, color='k', lw=0.75, zorder=2)

    tick_labels = [
        f'{group}: n {neuron_id} ({count})'
        for group, neuron_id, count in zip(neuron_trace_groups, traced_neuron_ids, neuron_selection_counts)
    ]
    for edge in epoch_edges:
        ax.axvline(t[edge], color='k', lw=0.25, alpha=0.2)

    ax.text(t[0]-10, 0, 'Seq 1', ha='right', va='center', fontsize=7, color='w')
    ax.set_yticks(offsets)
    ax.set_yticklabels([])#tick_labels)
    ax.set_ylabel('activitiy of single neurons')
    ax.set_xlabel('time')
    plt.axis('off')
    fig.tight_layout()
    return fig, ax








def task_highlight_color2(task):
    return {'lc': "#0A6A9D", 'seq': "#d2720c", 'fp': "#D10C3D"}[task['kind']]


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
plt.rcParams['figure.figsize'] = [6.5*2.5/2.54, 5/2.54]
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

# The plotted session used the third consecutive network draw after seed 11.
for realization in range(3):
    tasks, labels = build_tasks(rng, N, f, delta_m)
schedule = selection_schedule(len(tasks))
(t, readouts, detail_readouts, layout, traced_neuron_ids, neuron_trace_groups,
 neuron_selection_counts, neuron_rate_traces, neuron_selected_trace,
 epoch_edges) = simulate_combined_network(tasks, schedule, seed=seed + 100)
plot_offset_task_readouts(t, tasks, labels, detail_readouts, layout, schedule, epoch_edges)
savefig('barcodes_task_readouts')
plot_representative_neuron_traces(t, neuron_rate_traces, traced_neuron_ids,
                                neuron_trace_groups, neuron_selection_counts,
                                neuron_selected_trace, epoch_edges)
savefig('barcodes_neurons')

def plot_barcode_blocks():
    # Panel A: illustrative barcode masks and connectivity blocks.
    N = 5000
    f = 0.7
    conn_seed = 1
    rng = np.random.default_rng(conn_seed)
    covs = [np.array([[0.8, 0.4], [-0.4, 0.8]]), 
            np.array([[0.5, 0.3], [0.3, 0.5]])]
    loadings = np.zeros((2, N, 4))
    for task, C in enumerate(covs):
        cov = np.block([[np.eye(2), C], [C.T, np.eye(2)]]) / N
        loadings[task] = rng.multivariate_normal(np.zeros(4), cov, size=N)
    barcodes = (rng.random((2, N)) < f).astype(float)
    loadings *= barcodes[:, :, None]
    alpha = np.array([2.2, 2.0]) / f
    for task, label in enumerate(['A', 'B']):
        fig, ax = plt.subplots(figsize=(2, 2))
        J = alpha[task] * loadings[task, :20, :2] @ loadings[task, :20, 2:].T
        ax.pcolor(J, cmap='bwr', vmin=-19/N, vmax=19/N, rasterized=True)
        ax.axis('off')
        savefig('barcode_connectivity_' + label, transparent=True)
        fig, ax = plt.subplots(figsize=(0.35, 2))
        ax.imshow(barcodes[task, :20, None], cmap='binary', interpolation='nearest', aspect='auto')
        ax.axis('off')
        savefig('barcode_' + label, transparent=True)

plot_barcode_blocks()

plt.show()
