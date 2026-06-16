import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from force_driven_beam import solve_force_driven_beam
from displacement_driven_beam import solve_displacement_driven_beam

def run_adaptive_relaxation_coupling(
    L_neumann,L_dirichlet,n_elems_neumann,n_elems_dirichlet,E,rho,
    dt,total_time,tol,max_iter,relaxation_ini,
    width_neumann,  
    thickness_neumann,  
    width_dirichlet,  
    thickness_dirichlet,  
    ext_force_neumann_time,
    ext_pos_neumann,
    ext_force_dirichlet_time,
    ext_pos_dirichlet,
    adaptive_relaxation,  
    relaxation_min,  
    relaxation_max,
):
    """Couple two substructures by time-domain fixed-point iteration with relaxation.
    
        Aitken's acceleration method is used to adaptively update the relaxation factor for the offline real-time hybrid simulation coupling iterations.
    
        Returns the final interface force and moment histories, RMSE convergence history, substructure displacement fields, time vector, relaxation-factor history, and interface displacement/rotation residual histories.
    
    """
    nsteps = int(np.floor(total_time / dt)) + 1
    t = np.linspace(0.0, total_time, nsteps)

    relaxation_history = [relaxation_ini]
    residual_history = []  
    
    # Offline RTHS numerical workflow step.
    delta_y_history = []
    delta_rz_history = []  

    # Precompute node counts and index helpers
    interface_y_idx_neumann = 0
    interface_rz_idx_neumann = 1

    n_nodes_dirichlet = n_elems_dirichlet + 1
    interface_y_idx_dirichlet = 2*(n_nodes_dirichlet-1)
    interface_rz_idx_dirichlet = interface_y_idx_dirichlet + 1

    if ext_force_neumann_time is None:
        ext_force_neumann_time = np.zeros(nsteps)
    else:
        ext_force_neumann_time = np.asarray(ext_force_neumann_time, dtype=float).ravel()
        if ext_force_neumann_time.size != nsteps:
            raise ValueError('ext_force_neumann_time must have length nsteps')

    if ext_force_dirichlet_time is None:
        ext_force_dirichlet_time = np.zeros(nsteps)
    else:
        ext_force_dirichlet_time = np.asarray(ext_force_dirichlet_time, dtype=float).ravel()
        if ext_force_dirichlet_time.size != nsteps:
            raise ValueError('ext_force_dirichlet_time must have length nsteps')

    Fb = np.zeros(nsteps)
    Mb = np.zeros(nsteps)
    history = []

    # Initial neumann solve with zero boundary (iteration 0 reference)
    interface_y_neumann_prev, interface_rz_neumann_prev, U_neumann_prev = solve_force_driven_beam(
        L=L_neumann, n_elems=n_elems_neumann, E=E, rho=rho,
        width=width_neumann, thickness=thickness_neumann, 
        F_boundary=np.zeros(nsteps), M_boundary=np.zeros(nsteps),
        ext_force=ext_force_neumann_time, ext_pos=ext_pos_neumann,
        dt=dt, total_time=total_time,
    )

    R_prev = None
    omega_prev = relaxation_ini
    converged = False

    for it in range(1, max_iter + 1):
        
        if it == 1:
            y_drive = interface_y_neumann_prev.copy()
            rz_drive = interface_rz_neumann_prev.copy()
        else:
            y_drive = relaxation_ini * interface_y_neumann_prev + (1.0 - relaxation_ini) * y_drive_prev
            rz_drive = relaxation_ini * interface_rz_neumann_prev + (1.0 - relaxation_ini) * rz_drive_prev

        reaction_force, reaction_moment, U_dirichlet = solve_displacement_driven_beam(
            L=L_dirichlet, n_elems=n_elems_dirichlet, E=E, rho=rho,
            width=width_dirichlet, thickness=thickness_dirichlet,
            ext_force=ext_force_dirichlet_time, ext_pos=ext_pos_dirichlet,
            free_y=y_drive, free_rz=rz_drive,
            dt=dt, total_time=total_time,
        )
        
        # Offline RTHS numerical workflow step.
        Fb_new = reaction_force
        Mb_new = reaction_moment
        
        interface_y_neumann_curr, interface_rz_neumann_curr, U_neumann_curr = solve_force_driven_beam(
            L=L_neumann, n_elems=n_elems_neumann, E=E, rho=rho,
            width=width_neumann, thickness=thickness_neumann,
            F_boundary=Fb_new, M_boundary=Mb_new,
            ext_force=ext_force_neumann_time, ext_pos=ext_pos_neumann,
            dt=dt, total_time=total_time,
        )

        interface_y_dirichlet = U_dirichlet[interface_y_idx_dirichlet, :]
        interface_rz_dirichlet = U_dirichlet[interface_rz_idx_dirichlet, :]

        # Offline RTHS numerical workflow step.
        delta_y = interface_y_neumann_curr - interface_y_dirichlet
        delta_rz = interface_rz_neumann_curr - interface_rz_dirichlet
        
        # Offline RTHS numerical workflow step.
        delta_y_history.append(delta_y.copy())
        delta_rz_history.append(delta_rz.copy())

        # RMSE.
        diff_sq = delta_y**2 + (delta_rz * (L_neumann+L_dirichlet))**2
        rmse = np.sqrt(np.mean(diff_sq))
        
        history.append(rmse)

        # RMSE.
        check_rmse = rmse

        # Offline RTHS implementation note.
        # Aitken.
        # Offline RTHS implementation note.
        if adaptive_relaxation:
            L_char = L_neumann + L_dirichlet
            X_in = np.hstack([y_drive.ravel(), (rz_drive * L_char).ravel()])
            X_out = np.hstack([interface_y_neumann_curr.ravel(), (interface_rz_neumann_curr * L_char).ravel()])
            
            R_curr = X_out - X_in
            
            if it == 1:
                R_prev = R_curr.copy()
                omega_prev = relaxation_ini
            else:
                delta_R = R_curr - R_prev
                num = np.dot(R_prev, delta_R)
                den = np.dot(delta_R, delta_R)
                
                if den > 1e-14:
                    omega_new = -omega_prev * (num / den)
                    relaxation_ini = max(relaxation_min, min(relaxation_max, omega_new))
                    print(f"  [Aitken Update] omega: {omega_prev:.4f} -> {relaxation_ini:.4f}")
                    relaxation_history.append(relaxation_ini)
                
                R_prev = R_curr.copy()
                omega_prev = relaxation_ini

        if check_rmse < tol:
            print(f"\n{'='*60}")
            print(f"Converged in {it} iterations (RMSE={check_rmse:.3e})")
            print(f"Final relaxation factor: {relaxation_ini:.6f}")
            Fb = Fb_new
            Mb = Mb_new
            U_neumann = U_neumann_curr
            U_dirichlet_final = U_dirichlet
            converged = True
            break   

        # Prepare for next iteration
        y_drive_prev = y_drive.copy()
        rz_drive_prev = rz_drive.copy()
        interface_y_neumann_prev = interface_y_neumann_curr.copy()
        interface_rz_neumann_prev = interface_rz_neumann_curr.copy()
        Fb = Fb_new
        Mb = Mb_new

    if not converged:
        print(f"\n{'='*60}")
        print(f"Reached max_iter={max_iter}, final RMSE={check_rmse:.3e}")
        print(f"Final relaxation factor: {relaxation_ini:.6f}")
        try:
            U_neumann = U_neumann_curr
            U_dirichlet_final = U_dirichlet
        except NameError:
            U_neumann = U_neumann_prev
            U_dirichlet_final = np.zeros((2*(n_elems_dirichlet+1), nsteps))

    print('\nFixed-point coupling finished.')
    print(f'Convergence history: {history}')
    
    return Fb, Mb, history, U_neumann, U_dirichlet_final, t, relaxation_history, delta_y_history, delta_rz_history


if __name__ == "__main__":
    L_dirichlet = 0.2
    L_neumann = 10-L_dirichlet
    n_elems_neumann = max(int(L_neumann/0.1), 100)
    n_elems_dirichlet = max(int(L_dirichlet/0.1), 5)
    E = 5e10
    rho = 1160
    width_neumann = 0.5  
    thickness_neumann = 0.005  
    width_dirichlet = 0.5  
    thickness_dirichlet = 0.005 
    dt = 0.025
    T = 10
    tol = 1e-4
    max_iter = 40
    relaxation_min = 0.01
    relaxation_max = 1
    nsteps = int(np.floor(T/dt)) + 1
    t = np.linspace(0.0, T, nsteps)


    # Offline RTHS numerical workflow step.
    impulse_amplitude = 30.0    # Offline RTHS numerical workflow step.
    impulse_duration = 0.1     # Offline RTHS numerical workflow step.
    t_peak = 0.04                # Offline RTHS numerical workflow step.

    # Offline RTHS numerical workflow step.
    ext_force_neumann_time = np.zeros_like(t)
    mask = (t >= t_peak) & (t <= t_peak + impulse_duration)
    ext_force_neumann_time[mask] = impulse_amplitude * np.sin(
        np.pi * (t[mask] - t_peak) / impulse_duration
    )

    ext_pos_neumann = 5
    ext_force_dirichlet_time = None
    ext_pos_dirichlet = 0.0
    relaxation = 0.7
    adaptive_relaxation = True
    
    Fb, Mb, history, U_neumann, U_dirichlet, t, relaxation_history, delta_y_history, delta_rz_history = run_adaptive_relaxation_coupling(
        L_neumann=L_neumann,
        L_dirichlet=L_dirichlet,
        n_elems_neumann=n_elems_neumann,
        n_elems_dirichlet=n_elems_dirichlet,
        E=E,
        rho=rho,
        width_neumann=width_neumann,  
        thickness_neumann=thickness_neumann,  
        width_dirichlet=width_dirichlet, 
        thickness_dirichlet=thickness_dirichlet,
        ext_force_neumann_time=ext_force_neumann_time,
        ext_pos_neumann=ext_pos_neumann,
        ext_force_dirichlet_time=ext_force_dirichlet_time,
        ext_pos_dirichlet=ext_pos_dirichlet,
        dt=dt,
        total_time=T,
        tol=tol,
        max_iter=max_iter,
        relaxation_ini=relaxation,
        adaptive_relaxation=adaptive_relaxation,  
        relaxation_min=relaxation_min, 
        relaxation_max=relaxation_max
    )

    print('history:', history)

    # Offline RTHS numerical workflow step.
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    # RMSE-based convergence evaluation.
    axes[0].semilogy(range(1, len(history)+1), history, 'bo-', linewidth=1.5, markersize=6)
    axes[0].axhline(y=tol, color='r', linestyle='--', label=f'Tolerance={tol}')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('RMSE (log scale)')
    axes[0].set_title('Convergence History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Offline RTHS numerical workflow step.
    axes[1].plot(range(1, len(relaxation_history)+1), relaxation_history, 'go-', linewidth=1.5, markersize=6)
    axes[1].set_xlabel('Relaxation Update Step')
    axes[1].set_ylabel('Relaxation Factor (omega)')
    axes[1].set_title('Aitken Relaxation Factor Evolution')
    axes[1].set_ylim([relaxation_min - 0.1, relaxation_max + 0.1])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # Draw boundary force 
    plt.figure()
    plt.plot(t, Fb)
    plt.title('Converged boundary force (applied to neumann domain)')
    plt.xlabel('time (s)')
    plt.grid(True)
    plt.show()

    # Plot boundary displacement response
    n_nodes_neumann = n_elems_neumann + 1
    y_neumann = U_neumann[0:2*n_nodes_neumann:2, :]
    n_nodes_dirichlet = n_elems_dirichlet + 1
    interface_y_dirichlet = U_dirichlet[0:2*n_nodes_dirichlet:2, :]

    plt.figure()
    plt.plot(t, y_neumann[0, :], label='right boundary')
    plt.plot(t, interface_y_dirichlet[-1, :], '--', label='left boundary')
    plt.legend()
    plt.title('Boundary displacement comparison')
    plt.grid(True)
    plt.show()

    # Global displacement visualization
    n_nodes_dirichlet = n_elems_dirichlet + 1
    n_nodes_neumann = n_elems_neumann + 1

    interface_y_dirichlet = U_dirichlet[0:2*n_nodes_dirichlet:2, :]
    y_neumann = U_neumann[0:2*n_nodes_neumann:2, :]

    y_neumann_no_interface = y_neumann[1:, :]
    y_global = np.vstack([interface_y_dirichlet, y_neumann_no_interface])

    x_dirichlet = np.linspace(0.0, L_dirichlet, n_nodes_dirichlet)
    x_neumann = np.linspace(L_dirichlet, L_dirichlet + L_neumann, n_nodes_neumann)

    x_neumann_no_interface = x_neumann[1:]
    x_global = np.hstack([x_dirichlet, x_neumann_no_interface])

    plt.figure(figsize=(10, 4))
    time_indices = [
        0,
        int(0.25 * nsteps),
        int(0.50 * nsteps),
        int(0.75 * nsteps),
        nsteps - 1
    ]

    for k in time_indices:
        plt.plot(
            x_global,
            y_global[:, k],
            label=f"t = {t[k]:.2f} s"
        )

    plt.xlabel("x (m)")
    plt.ylabel("Transverse displacement y (m)")
    plt.title("Global displacement shape of coupled beam")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Time evolution animation
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], 'b-', lw=2, label='Beam displacement')
    
    ax.axvline(x=L_dirichlet, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Interface')
    
    ax.text(0.02, 0.95, 'Left Beam', transform=ax.transAxes, fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.text(0.5, 0.95, 'Right Beam', transform=ax.transAxes, fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax.plot(0, 0, 'ks', markersize=10, label='Fixed end')
    
    y_margin = 0.1 * (np.max(y_global) - np.min(y_global)) if np.max(y_global) != np.min(y_global) else 0.1
    ax.set_xlim(-0.1, L_dirichlet + L_neumann + 0.1)
    ax.set_ylim(np.min(y_global) - y_margin, np.max(y_global) + y_margin)
    
    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Transverse displacement y (m)")
    ax.set_title("Coupled beam dynamics - Real-time animation")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    time_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    max_text = ax.text(0.85, 0.95, '', transform=ax.transAxes, fontsize=10,
                      bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    min_text = ax.text(0.85, 0.90, '', transform=ax.transAxes, fontsize=10,
                      bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    def update(frame):
        line.set_data(x_global, y_global[:, frame])
        time_text.set_text(f"Time: {t[frame]:.2f} s")
        max_disp = np.max(y_global[:, frame])
        min_disp = np.min(y_global[:, frame])
        max_text.set_text(f"Max: {max_disp:.4f} m")
        min_text.set_text(f"Min: {min_disp:.4f} m")
        ax.set_title(f"Coupled beam dynamics - Time: {t[frame]:.2f} s")
        return line, time_text, max_text, min_text

    ani = FuncAnimation(
        fig,
        update,
        frames=nsteps,
        interval=30,
        blit=True,
        repeat=True
    )

    plt.show()

    # Offline RTHS implementation note.
    # CSV input/output step for post-processing.
    # Offline RTHS implementation note.
    # Offline RTHS numerical workflow step.
    # Offline RTHS numerical workflow step.
    interface_y_right = U_neumann[0, :]  # Offline RTHS numerical workflow step.
    interface_rz_right = U_neumann[1, :] # Offline RTHS numerical workflow step.

    # Offline RTHS numerical workflow step.
    interface_y_left = U_dirichlet[-2, :]   # Offline RTHS numerical workflow step.
    interface_rz_left = U_dirichlet[-1, :]  # Offline RTHS numerical workflow step.

    # Offline RTHS numerical workflow step.
    interface_data = {
        'time_s': t,
        'right_interface_y_m': interface_y_right,
        'right_interface_rz_rad': interface_rz_right,
        'left_interface_y_m': interface_y_left,
        'left_interface_rz_rad': interface_rz_left,
        'interface_force_Fb_N': Fb,
        'interface_moment_Mb_Nm': Mb
    }

    # DataFrame.
    df_interface = pd.DataFrame(interface_data)

    # CSV input/output step for post-processing.
    csv_filename = 'coupled_beam_interface_data.csv'
    df_interface.to_csv(csv_filename, index=False, float_format='%.10e')

    print(f"\n{'='*60}")
    print(f"Interface data saved to file: {csv_filename}")
    print(f"File contains {len(df_interface)} time steps and {len(df_interface.columns)} data columns")
    print(f"Data columns: {list(df_interface.columns)}")
    print(f"{'='*60}")    
