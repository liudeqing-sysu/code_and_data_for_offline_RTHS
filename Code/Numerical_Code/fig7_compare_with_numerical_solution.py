import numpy as np
import matplotlib.pyplot as plt
from force_driven_beam import solve_force_driven_beam
import os
import csv
from adaptive_relaxation_coupling import run_adaptive_relaxation_coupling

# Offline RTHS numerical workflow step.
if __name__ == "__main__":
    # Offline RTHS numerical workflow step.
    PLOT_DPI = 600
    SAVE_FORMAT = "svg"
    PLOT_FIGSIZE_NEW = (4.9 / 2.54, 4 / 2.54)  # Offline RTHS numerical workflow step.

    # Offline RTHS numerical workflow step.
    L_total = 10.0
    L_left = 0.1
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
    n_elems_total = max(int(L_total / 0.01), 100)
    
    # Offline RTHS numerical workflow step.
    relaxation_ini = 0.7
    relaxation_min = 0.01
    relaxation_max = 1.0
    adaptive_relaxation = True
    ext_force_dirichlet_time = None
    ext_pos_dirichlet = 0.5

    # Comparison between coupled and fully numerical reference responses.
    print('Figure-generation status message.')
    boundary_y_numerical, boundary_rz_numerical, U_numerical = solve_force_driven_beam(
        L=L_total,
        n_elems=n_elems_total,
        E=E,
        rho=rho,
        width=width,
        thickness=thickness,
        F_boundary=np.zeros(nsteps),
        M_boundary=np.zeros(nsteps),
        ext_force=ext_force_right,
        ext_pos=5.0,
        dt=dt,
        total_time=T
    )
    
    # Comparison between coupled and fully numerical reference responses.
    interface_node_idx = int(np.round(L_left / (L_total / n_elems_total)))
    interface_node_idx = max(0, min(int(L_total / (L_total / n_elems_total)), interface_node_idx))
    y_numerical_interface = U_numerical[2 * interface_node_idx, :]
    rz_numerical_interface = U_numerical[2 * interface_node_idx + 1, :]
    
    print(f"Fully numerical solution completed; interface node index: {interface_node_idx}")

    # 2. Aitken.
    print('Figure-generation status message.')
    Fb, Mb, history, U_neumann, U_dirichlet_final, t_aitken, relax_hist, delta_y_hist, delta_rz_hist = run_adaptive_relaxation_coupling(
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
    
    print(f"Aitken-based decoupled solution completed; converged iteration count: {len(history)}")

    # Offline RTHS numerical workflow step.
    # Comparison between coupled and fully numerical reference responses.
    y_right_interface = U_neumann[0, :]
    rz_right_interface = U_neumann[1, :]
    
    # Comparison between coupled and fully numerical reference responses.
    n_nodes_left = n_elems_left + 1
    right_y_idx_left = 2 * (n_nodes_left - 1)
    right_rz_idx_left = right_y_idx_left + 1
    y_left_interface = U_dirichlet_final[right_y_idx_left, :]
    rz_left_interface = U_dirichlet_final[right_rz_idx_left, :]

    # Offline RTHS numerical workflow step.
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 9,
        'axes.linewidth': 0.8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'lines.linewidth': 1.0,
        'figure.dpi': PLOT_DPI
    })

    save_dir = "compare_plots"
    os.makedirs(save_dir, exist_ok=True)

    # Legend.
    LEGEND_LOC = 'upper right'
    LEGEND_POS = (0.9999, 0.9999)


    # Offline RTHS numerical workflow step.
    print('Figure-generation status message.')

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_NEW)

    ax.plot(t, y_numerical_interface,
            color='black', linewidth=1.2,
            label='Ref')

    ax.plot(t, y_right_interface,
            color='#d62728', linestyle='--', linewidth=1.0,
            label='Num')

    ax.plot(t, y_left_interface,
            color='#1f77b4', linestyle=':', linewidth=1.0,  # Comparison between coupled and fully numerical reference responses.
            label='Exp')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Displacement (m)')

    ax.set_xlim(0, 10)
    ax.set_xticks(np.linspace(0, 10, 6))

    y_max = max(
        np.max(np.abs(y_numerical_interface)),
        np.max(np.abs(y_right_interface)),
        np.max(np.abs(y_left_interface))
    )
    y_min_limit = -0.55 * y_max
    y_max_limit = 1.7 * y_max

    # Comparison between coupled and fully numerical reference responses.
    y_ticks = np.arange(np.floor(y_min_limit / 0.2) * 0.2,
                        np.ceil(y_max_limit / 0.2) * 0.2 + 0.01,
                        0.2)
    ax.set_ylim(y_min_limit, y_max_limit)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))  # Offline RTHS numerical workflow step.

    # Legend 8.
    ax.legend(
        loc=LEGEND_LOC,
        bbox_to_anchor=LEGEND_POS,
        frameon=False,
        fontsize=8,
        handlelength=0.8,
        handletextpad=0.5,
        labelspacing=0.05,
        borderpad=0.25
    )

    ax.grid(False)
    ax.tick_params(direction='in', width=0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/displacement_comparison.{SAVE_FORMAT}', bbox_inches='tight')
    plt.close()

    print('Figure-generation status message.')


    # Offline RTHS numerical workflow step.
    print('Figure-generation status message.')

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_NEW)

    ax.plot(t, rz_numerical_interface,
            color='black', linewidth=1,
            label='Ref')

    ax.plot(t, rz_right_interface,
            color='#d62728', linestyle='--', linewidth=0.8,
            label='Num')

    ax.plot(t, rz_left_interface,
            color='#1f77b4', linestyle=':', linewidth=0.8,  # Comparison between coupled and fully numerical reference responses.
            label='Exp')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rotation (rad)')

    ax.set_xlim(0, 10)
    ax.set_xticks(np.linspace(0, 10, 6))

    rz_max = max(
        np.max(np.abs(rz_numerical_interface)),
        np.max(np.abs(rz_right_interface)),
        np.max(np.abs(rz_left_interface))
    )
    rz_min_limit = -1.5 * rz_max
    rz_max_limit = 1.5 * rz_max

    # Comparison between coupled and fully numerical reference responses.
    rz_ticks = np.arange(np.floor(rz_min_limit / 0.1) * 0.1,
                        np.ceil(rz_max_limit / 0.1) * 0.1 + 0.01,
                        0.1)
    ax.set_ylim(rz_min_limit, rz_max_limit)
    ax.set_yticks(rz_ticks)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))  # Offline RTHS numerical workflow step.

    # Legend 8.
    ax.legend(
        loc=LEGEND_LOC,
        bbox_to_anchor=LEGEND_POS,
        frameon=False,
        fontsize=8,
        handlelength=0.8,
        handletextpad=0.5,
        labelspacing=0.05,
        borderpad=0.25
    )

    ax.grid(False)
    ax.tick_params(direction='in', width=0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/rotation_comparison.{SAVE_FORMAT}', bbox_inches='tight')
    plt.close()

    print('Figure-generation status message.')

    # CSV input/output step for post-processing.
    print('Figure-generation status message.')

    csv_path = os.path.join(save_dir, "comparison_data.csv")
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Comparison between coupled and fully numerical reference responses.
        writer.writerow(['Time (s)', 
                        'Ref_Disp (m)', 'Num_Disp (m)', 'Exp_Disp (m)',
                        'Ref_Rot (rad)', 'Num_Rot (rad)', 'Exp_Rot (rad)'])
        
        # Offline RTHS numerical workflow step.
        for i in range(len(t)):
            writer.writerow([
                t[i],
                y_numerical_interface[i],
                y_right_interface[i],
                y_left_interface[i],
                rz_numerical_interface[i],
                rz_right_interface[i],
                rz_left_interface[i]
            ])

    print(f"Data saved to: {csv_path}")

    # Offline RTHS numerical workflow step.
    print('Figure-generation status message.')
    error_y_right = np.sqrt(np.mean((y_numerical_interface - y_right_interface)**2))
    error_y_left = np.sqrt(np.mean((y_numerical_interface - y_left_interface)**2))
    error_rz_right = np.sqrt(np.mean((rz_numerical_interface - rz_right_interface)**2))
    error_rz_left = np.sqrt(np.mean((rz_numerical_interface - rz_left_interface)**2))

    print(f"Displacement RMSE (Num vs Ref): {error_y_right:.6e} m")
    print(f"Displacement RMSE (Exp vs Ref): {error_y_left:.6e} m")
    print(f"Rotation RMSE (Num vs Ref): {error_rz_right:.6e} rad")
    print(f"Rotation RMSE (Exp vs Ref): {error_rz_left:.6e} rad")

    max_y_numerical = np.max(np.abs(y_numerical_interface))
    max_rz_numerical = np.max(np.abs(rz_numerical_interface))

    if max_y_numerical > 0:
        error_y_right_rel = error_y_right / max_y_numerical * 100
        error_y_left_rel = error_y_left / max_y_numerical * 100
        print(f"Relative displacement error (Num): {error_y_right_rel:.4f}%")
        print(f"Relative displacement error (Exp): {error_y_left_rel:.4f}%")

    if max_rz_numerical > 0:
        error_rz_right_rel = error_rz_right / max_rz_numerical * 100
        error_rz_left_rel = error_rz_left / max_rz_numerical * 100
        print(f"Relative rotation error (Num): {error_rz_right_rel:.4f}%")
        print(f"Relative rotation error (Exp): {error_rz_left_rel:.4f}%")

    print('Figure-generation status message.')
    print(f"Files saved in: {save_dir}/")
