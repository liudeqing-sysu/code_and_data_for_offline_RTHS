import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from adaptive_relaxation_coupling import run_adaptive_relaxation_coupling

# Publication-quality convergence-comparison figure configuration.
if __name__ == "__main__":
    L_total = 10.0
    length_ratios = [0.1] #, 0.2, 0.3, 0.4  
    fixed_relax_list = [0.2, 0.4, 0.6, 0.8, 1.0]
    #################################################################
    ###################################################################
    #################################################################
    ###################################################################
    E = 5e10
    rho = 1160
    width = 0.5
    thickness = 0.005
    dt = 0.05
    T = 10
    max_iter = 10    # Publication and post-processing configuration.
    tol = 1e-4
    relaxation_ini = 0.7
    relaxation_min = 0.01
    relaxation_max = 1.0
    adaptive_relaxation = True
    ext_force_dirichlet_time = None
    ext_pos_dirichlet = 0.5

    impulse_amplitude = 30.0
    impulse_duration = 0.1
    t_peak = 0.04

    all_results = {}

    # Offline RTHS numerical workflow step.
    for L_left in length_ratios:
        L_right = L_total - L_left
        print(f"\n===== Left beam length = {L_left:.1f} m | right beam = {L_right:.1f} m =====")

        n_elems_right = max(int(L_right / 0.01), 30)
        n_elems_left = max(int(L_left / 0.01), 5)
        nsteps = int(np.floor(T / dt)) + 1
        t = np.linspace(0, T, nsteps)

        ext_force_right = np.zeros_like(t)
        mask = (t >= t_peak) & (t <= t_peak + impulse_duration)
        ext_force_right[mask] = impulse_amplitude * np.sin(np.pi * (t[mask] - t_peak) / impulse_duration)

        ratio_res = {}

        # Offline RTHS numerical workflow step.
        for r in fixed_relax_list:
            print(f"Computing fixed relaxation factor {r:.1f}...")
            _, _, history, *_ = run_adaptive_relaxation_coupling(
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
                ext_force_dirichlet_time=None,
                ext_pos_dirichlet=0.5,
                dt=dt, 
                total_time=T, 
                tol=tol, 
                max_iter=max_iter, 
                relaxation_ini=r,
                adaptive_relaxation=False,
                relaxation_min=relaxation_min,
                relaxation_max=relaxation_max
            )
            ratio_res[f"Fixed_{r:.1f}"] = history

        # Offline RTHS numerical workflow step.
        print("Computing adaptive relaxation factor...")
        _, _, history, *_ = run_adaptive_relaxation_coupling(
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
            ext_force_dirichlet_time=None,
            ext_pos_dirichlet=0.5,
            dt=dt, 
            total_time=T, 
            tol=tol, 
            max_iter=max_iter, 
            relaxation_ini=relaxation_ini,
            adaptive_relaxation=adaptive_relaxation,
            relaxation_min=relaxation_min,
            relaxation_max=relaxation_max
        )
        ratio_res["Adaptive"] = history
        all_results[f"Left_{L_left:.1f}"] = ratio_res

    # -------------------------------------------------------------------------
    # CSV input/output step for post-processing.
    # -------------------------------------------------------------------------
    os.makedirs("convergence_results", exist_ok=True)
    os.makedirs("convergence_comparison_figures", exist_ok=True)  # Offline RTHS numerical workflow step.

    for ratio_name, methods in all_results.items():
        csv_name = f"convergence_results/{ratio_name}.csv"
        max_iter_len = max(len(h) for h in methods.values())
        iter_col = list(range(1, max_iter_len+1))
        data_dict = {"Iteration": iter_col}
        for key, hist in methods.items():
            padded = hist + [np.nan] * (max_iter_len - len(hist))
            data_dict[key] = padded
        with open(csv_name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(data_dict.keys())
            for row in zip(*data_dict.values()):
                writer.writerow(row)
        print(f"{csv_name} saved successfully")

# -------------------------------------------------------------------------
# Publication-quality convergence-comparison figure configuration.
# -------------------------------------------------------------------------
plt.rcParams.update({
    # Publication-quality convergence-comparison figure configuration.
    'font.family': 'Times New Roman',
    'font.size': 10,          # Publication-quality convergence-comparison figure configuration.
    'axes.linewidth': 1,
    'axes.spines.top': True,   # Offline RTHS numerical workflow step.
    'axes.spines.right': True, # Offline RTHS numerical workflow step.
    'axes.labelsize': 10,      # Publication-quality convergence-comparison figure configuration.
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,      # Publication-quality convergence-comparison figure configuration.
    'lines.linewidth': 1,
    'lines.markersize': 2,
    'grid.alpha': 0.3,
    'figure.dpi': 600
})

# Offline RTHS numerical workflow step.
cm = 1 / 2.54

colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
markers = ['o', 'o', 'o', 'o', 'o']
fixed_values = [0.2, 0.4, 0.6, 0.8, 1.0]  # Offline RTHS numerical workflow step.

# Offline RTHS numerical workflow step.
for idx, (ratio_name, methods) in enumerate(all_results.items()):
    fig, ax = plt.subplots(figsize=(8*cm, 6*cm))

    # Publication-quality convergence-comparison figure configuration.
    fixed_keys = [k for k in methods if "Fixed" in k]
    for i, key in enumerate(fixed_keys):
        hist = methods[key]
        ax.semilogy(
            range(1, len(hist)+1), hist,
            color=colors[i], marker=markers[i], markevery=1,
            label=f'{fixed_values[i]}'
        )

    # Offline RTHS numerical workflow step.
    adaptive_hist = methods["Adaptive"]
    ax.semilogy(
        range(1, len(adaptive_hist)+1), adaptive_hist,
        color='black', 
        linestyle='-', 
        linewidth=1, 
        marker='o', 
        markevery=1,
        label='Adaptive'
    )

    ax.axhline(
        tol,
        color='#7A7A7A',          # Offline RTHS numerical workflow step.
        linestyle='--',       # Offline RTHS numerical workflow step.
        linewidth=1,
        label='Tolerance'
    )

    # Offline RTHS numerical workflow step.
    ax.set_xlabel('Iteration')
    ax.set_ylabel('RMSE (m)')

    # Publication-quality convergence-comparison figure configuration.
    ax.set_xticks(np.arange(0, max_iter + 1, 2))  #2  3  5   9 ###################################
    ax.set_ylim(1e-5, 0.3)       # Y (1e-5, 0.3) (1e-5, 0.3) (2e-5, 20) (2e-5, 20)########################.
    #################################################################
    ###################################################################
    #################################################################
    ###################################################################


    # Offline RTHS numerical workflow step.
    ax.legend(
        frameon=False,
        ncol=2,
        handlelength=0.8,
        handletextpad=0.4,
        columnspacing=0.3,
        loc='lower left',
        bbox_to_anchor=(0.00001, 0.00001) 
    )

    # Offline RTHS numerical workflow step.
    ax.grid(False)

    plt.tight_layout()

    # Offline RTHS numerical workflow step.
    save_path = f"convergence_comparison_figures/{ratio_name}"
    plt.savefig(f"{save_path}.svg", dpi=600, bbox_inches='tight')

    # -------------------------------------------------------------------------
    # Offline RTHS numerical workflow step.
    # -------------------------------------------------------------------------
    print('Figure-generation status message.')
    for ratio, methods in all_results.items():
        print(f"\n【{ratio}】")
        for name, hist in methods.items():
            conv = next((i+1 for i, v in enumerate(hist) if v < tol), "Not converged")
            print(f" {name:14s} | Final RMSE: {hist[-1]:.6f} | Iteration: {conv}")
