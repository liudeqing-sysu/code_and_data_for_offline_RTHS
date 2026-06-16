import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from adaptive_relaxation_coupling import run_adaptive_relaxation_coupling

# Offline RTHS numerical workflow step.
if __name__ == "__main__":
    # Offline RTHS numerical workflow step.
    TARGET_ITER_STEPS = [1, 16, 30, 45] #[1, 3, 5, 7] [1, 5, 9, 14] [1, 9, 17, 25] [1, 16, 30, 45] 
    PLOT_DPI = 600
    SAVE_FORMAT = "svg"
    PLOT_FIGSIZE = (10, 5)

    # Offline RTHS numerical workflow step.
    L_total = 10.0
    L_left = 0.4
    L_right = L_total - L_left
    E = 5e10
    rho = 1160
    width = 0.5
    thickness = 0.005
    dt = 0.05
    T = 10
    max_iter = 50
    tol = 1e-4

    # Offline RTHS numerical workflow step.
    impulse_amplitude = 30.0
    impulse_duration = 0.1
    t_peak = 0.04
    nsteps = int(np.floor(T / dt)) + 1
    t = np.linspace(0, T, nsteps)
    ext_force_right = np.zeros_like(t)
    mask = (t >= t_peak) & (t <= t_peak + impulse_duration)
    ext_force_right[mask] = impulse_amplitude * np.sin(np.pi * (t[mask] - t_peak) / impulse_duration)

    n_elems_right = max(int(L_right / 0.01), 30)
    n_elems_left = max(int(L_left / 0.01), 5)

    # Offline RTHS numerical workflow step.
    PLOT_FIGSIZE_NEW = (4.65 / 2.54, 3.5 / 2.54)  # Offline RTHS numerical workflow step.
    
    # Offline RTHS numerical workflow step.
    relaxation_ini = 0.7
    relaxation_min = 0.01
    relaxation_max = 1.0
    adaptive_relaxation = True
    ext_force_dirichlet_time = None
    ext_pos_dirichlet = 0.5

    # Offline RTHS numerical workflow step.
    print(f"Start computation: {'adaptive relaxation factor' if adaptive_relaxation else f'fixed relaxation factor {relaxation_ini}'}")
    Fb, Mb, history, U_neumann, U_dirichlet_final, t, relax_hist, delta_y_hist, delta_rz_hist = run_adaptive_relaxation_coupling(
        L_neumann=L_right, 
        L_dirichlet=L_left,
        n_elems_neumann=n_elems_right, 
        n_elems_dirichlet=n_elems_left,
        E=E, 
        rho=rho, 
        width_neumann=width, 
        thickness_neumann=thickness,
        width_dirichlet=width, 
        thickness_dirichlet=thickness,
        ext_force_neumann_time=ext_force_right, 
        ext_pos_neumann=5.0,
        ext_force_dirichlet_time=ext_force_dirichlet_time,
        ext_pos_dirichlet=ext_pos_dirichlet,
        dt=dt, 
        total_time=T, 
        tol=tol, 
        max_iter=max_iter, 
        relaxation_ini=relaxation_ini,
        adaptive_relaxation=adaptive_relaxation,
        relaxation_min=relaxation_min,
        relaxation_max=relaxation_max
    )
# Offline RTHS numerical workflow step.
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 9,              # Offline RTHS numerical workflow step.
    'axes.linewidth': 1,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 0.8,
    'figure.dpi': PLOT_DPI
})

save_dir = "iter_error_plots"
os.makedirs(save_dir, exist_ok=True)
plot_color = "#1f76b4"

# Offline RTHS numerical workflow step.
valid_iters = [it for it in TARGET_ITER_STEPS if it <= len(delta_y_hist)]
if len(valid_iters) == 0:
    raise ValueError('Figure-generation status message.')

# Offline RTHS numerical workflow step.
def ceil_to_2_decimals(val):
    return np.ceil(val * 100) / 100

if len(delta_y_hist) > 0:
    y1 = delta_y_hist[0]
    rz1 = delta_rz_hist[0]

    y_abs_max = max(abs(y1.min()), abs(y1.max()))
    y_abs_max_ceil = ceil_to_2_decimals(y_abs_max * 1.1)
    ylim_disp = [-y_abs_max_ceil, y_abs_max_ceil]

    rz_abs_max = max(abs(rz1.min()), abs(rz1.max()))
    rz_abs_max_ceil = ceil_to_2_decimals(rz_abs_max * 1.1)
    ylim_rot = [-rz_abs_max_ceil, rz_abs_max_ceil]
else:
    ylim_disp = [-1e-5, 1e-5]
    ylim_rot = [-1e-5, 1e-5]

# Offline RTHS numerical workflow step.
from matplotlib.ticker import MultipleLocator

for it in valid_iters:
    delta_y = delta_y_hist[it - 1]
    delta_rz = delta_rz_hist[it - 1]

    # Offline RTHS numerical workflow step.
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_NEW)
    ax.plot(t, delta_y, color=plot_color, linewidth=0.8)

    ax.set_title(f'Iteration {it}', fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Error (m)', fontsize=9)

    ax.set_ylim(ylim_disp)
    ax.set_xlim(0, 10)
    ax.set_xticks([0, 5, 10])

    y_tick_max = max(abs(ylim_disp[0]), abs(ylim_disp[1]))
    y_ticks = [-y_tick_max, 0, y_tick_max]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{v:.2f}' for v in y_ticks], fontsize=9)

    ax.yaxis.set_minor_locator(MultipleLocator(y_tick_max / 2))
    ax.tick_params(axis='both', labelsize=9)

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/left_{L_left}_iter_{it}_displacement_error.{SAVE_FORMAT}', bbox_inches='tight')
    plt.close()

    # Offline RTHS numerical workflow step.
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_NEW)
    ax.plot(t, delta_rz, color=plot_color, linewidth=0.8)

    ax.set_title(f'Iteration {it}', fontsize=9)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Error (rad)', fontsize=9)

    ax.set_ylim(ylim_rot)
    ax.set_xlim(0, 10)
    ax.set_xticks([0, 5, 10])

    y_tick_max = max(abs(ylim_rot[0]), abs(ylim_rot[1]))
    y_ticks = [-y_tick_max, 0, y_tick_max]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{v:.2f}' for v in y_ticks], fontsize=9)

    ax.yaxis.set_minor_locator(MultipleLocator(y_tick_max / 2))
    ax.tick_params(axis='both', labelsize=9)

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/left_{L_left}_iter_{it}_rotation_error.{SAVE_FORMAT}', bbox_inches='tight')
    plt.close()

# CSV input/output step for post-processing.
csv_filename = f"iter_error_all_steps_left_{L_left}.csv"
csv_path = os.path.join(save_dir, csv_filename)

header = ["Time (s)"]
for it in range(1, len(delta_y_hist) + 1):
    header.append(f"Iter_{it}_disp_error (m)")
    header.append(f"Iter_{it}_rot_error (rad)")

csv_data = []
for i in range(len(t)):
    row = [t[i]]
    for it_idx in range(len(delta_y_hist)):
        row.append(delta_y_hist[it_idx][i])
        row.append(delta_rz_hist[it_idx][i])
    csv_data.append(row)

with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(csv_data)

print(f"All iteration-step residuals saved to: {csv_path}")
print('Figure-generation status message.')
print(f"Total number of iterations aligned with RMSE: {len(delta_y_hist)}")
print(f"RMSE array length: {len(history)}")
print(f"Plotted iteration steps: {valid_iters}")
