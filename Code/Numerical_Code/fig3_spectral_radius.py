import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Publication and post-processing configuration.
# Spectral-radius visualization and manuscript figure configuration.
# Publication and post-processing configuration.
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Publication and post-processing configuration.
# Publication and post-processing configuration.
# Publication and post-processing configuration.
from dirichlet_dynamic_stiffness import compute_dirichlet_interface_dynamic_stiffness
from neumann_dynamic_stiffness import compute_neumann_interface_dynamic_stiffness

def plot_spectral_radius_2d_lambda_omega(
    total_L=7.0,          
    length_ratio=0.1,     # [] L_E / total_L.
    lam_range=(0.01, 1.0),
    freq_range=(1, 100),  
    grid_size=30          
):
    """Plot the spectral radius as a function of frequency and relaxation factor.
    """
    # ---------------------------------------------------------
    # Offline RTHS numerical workflow step.
    # ---------------------------------------------------------
    L_E = total_L * length_ratio
    L_N = total_L * (1.0 - length_ratio)

    # ---------------------------------------------------------
    # Offline RTHS numerical workflow step.
    # ---------------------------------------------------------
    freq_array = np.linspace(freq_range[0], freq_range[1], grid_size)
    lam_array = np.linspace(lam_range[0], lam_range[1], grid_size)
    
    Lam_grid, Freq_grid = np.meshgrid(lam_array, freq_array)
    Rho_grid = np.zeros_like(Freq_grid, dtype=np.float64)

    # ---------------------------------------------------------
    # Spectral-radius visualization and manuscript figure configuration.
    # Spectral-radius visualization and manuscript figure configuration.
    # ---------------------------------------------------------
    E_val = 5e10 
    rho_val = 1160
    
    # Offline RTHS numerical workflow step.
    n_elems_E = max(int(L_E / 0.1), 10)  # Dirichlet.
    n_elems_N = max(int(L_N / 0.1), 5)   # Neumann.

    # Zn (constrain_right.
    Zn_array = compute_neumann_interface_dynamic_stiffness(
        L=L_N, n_elems=n_elems_N, E=E_val, rho=rho_val, width=0.5, thickness=0.005, freq_array=freq_array
    )
    
    # Ze (constrain_left.
    Ze_array = compute_dirichlet_interface_dynamic_stiffness(
        L=L_E, n_elems=n_elems_E, E=E_val, rho=rho_val, width=0.5, thickness=0.005, freq_array=freq_array
    )

    # ---------------------------------------------------------
    # Spectral-radius visualization and manuscript figure configuration.
    # ---------------------------------------------------------
    for j in range(grid_size):
        Zn = Zn_array[j]
        Ze = Ze_array[j]
        
        D = np.linalg.solve(Zn, Ze)
        
        d11, d12 = D[0, 0], D[0, 1]
        d21, d22 = D[1, 0], D[1, 1]
        
        sqrt_term = np.sqrt((d11 - d22)**2 + 4.0 * d12 * d21)
        trace_part = 2.0 + d11 + d22

        for i in range(grid_size):
            lam_k = lam_array[i]
            
            # Offline RTHS numerical workflow step.
            mu_1 = (2.0 + lam_k * ( sqrt_term - trace_part)) / 2.0
            mu_2 = (2.0 + lam_k * (-sqrt_term - trace_part)) / 2.0
            
            Rho_grid[j, i] = max(abs(mu_1), abs(mu_2))

    # Publication and post-processing configuration.
    # 5. 2D (Heatmap.
    # Publication and post-processing configuration.
    # Spectral-radius visualization and manuscript figure configuration.
    cm_to_inch = 1 / 2.54
    fig, ax = plt.subplots(figsize=(6.6 * cm_to_inch, 4.3 * cm_to_inch)) 
    fig.patch.set_facecolor('white') 
    
    # Offline RTHS numerical workflow step.
    Rho_grid_clipped = np.clip(Rho_grid, a_min=0, a_max=3)

    # Offline RTHS numerical workflow step.
    mesh = ax.pcolormesh(
        Lam_grid, Freq_grid, Rho_grid_clipped, 
        cmap='viridis', vmin=0, vmax=2, shading='auto'
    )
    
    # Publication and post-processing configuration.
    contour = ax.contour(
        Lam_grid, Freq_grid, Rho_grid, 
        levels=[1.0], colors='red', linewidths=0.8
    )
    
    # "1.0".
    ax.clabel(contour, inline=True, fontsize=10, fmt='%.1f', colors='red')

    # Offline RTHS numerical workflow step.
    ax.set_xlabel('Relaxation Factor', fontsize=10)
    ax.set_ylabel('Frequency (Hz)', fontsize=10)
   
    # Publication and post-processing configuration.
    # Offline RTHS numerical workflow step.
    # Publication and post-processing configuration.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))  
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))  
    # Publication and post-processing configuration.

    # Offline RTHS numerical workflow step.
    cbar = fig.colorbar(mesh, ax=ax, pad=0.05, extend='max')
    cbar.set_label('Spectral Radius', fontsize=10)

    # Offline RTHS numerical workflow step.
    cbar_ticks = [0, 0.5, 1, 1.5, 2]
    cbar_labels = ['0.0', '0.5', '1.0', '1.5', r'$\geq 2.0$']
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_labels)

    plt.tight_layout()
    
    # Publication and post-processing configuration.
    # Spectral-radius visualization and manuscript figure configuration.
    # Publication and post-processing configuration.
    # Spectral-radius visualization and manuscript figure configuration.
    file_name_base = f"heatmap_ratio_{length_ratio}"
    
    plt.savefig(f'{file_name_base}.svg', format='svg', dpi=600, bbox_inches='tight')
    
    print(f"Vector figure saved successfully: {file_name_base}.svg")

if __name__ == "__main__":
    # Spectral-radius visualization and manuscript figure configuration.
    plot_spectral_radius_2d_lambda_omega(
        total_L=10, 
        length_ratio=0.01,     
        lam_range=(0.01, 1.0), 
        freq_range=(0.01, 5), 
        grid_size=100  
    )
