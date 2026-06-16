import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

def compute_dirichlet_interface_dynamic_stiffness(
    L, n_elems, E, rho, width, thickness,
    freq_array, alpha=0.01, beta=0.005
):
    """Compute the condensed interface dynamic stiffness matrix Z_E(omega).
    
        The beam is treated as a free-free substructure. The right-end nodal degrees of freedom form the coupling interface, and all remaining degrees of freedom are condensed as internal degrees of freedom.
    
    """

    # --------------------------------------------------
    # Offline RTHS numerical workflow step.
    # --------------------------------------------------
    n_nodes = n_elems + 1
    ndof = 2 * n_nodes
    dx = L / n_elems
    A = width * thickness
    I = (width * thickness**3) / 12.0

    Ke = (E * I / dx**3) * np.array([
        [12, 6*dx, -12, 6*dx],
        [6*dx, 4*dx**2, -6*dx, 2*dx**2],
        [-12, -6*dx, 12, -6*dx],
        [6*dx, 2*dx**2, -6*dx, 4*dx**2]
    ])

    Me = (rho * A * dx / 420.0) * np.array([
        [156, 22*dx, 54, -13*dx],
        [22*dx, 4*dx**2, 13*dx, -3*dx**2],
        [54, 13*dx, 156, -22*dx],
        [-13*dx, -3*dx**2, -22*dx, 4*dx**2]
    ])

    K = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))

    for e in range(n_elems):
        dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
        K[np.ix_(dofs, dofs)] += Ke
        M[np.ix_(dofs, dofs)] += Me

    C = alpha * M + beta * K

    # --------------------------------------------------
    # Offline RTHS numerical workflow step.
    # --------------------------------------------------
    # Interface degrees of freedom used for condensation.
    if_dofs = [2 * n_elems, 2 * n_elems + 1]
    
    # Condensed dynamic-stiffness assembly and verification step.
    active_dofs = np.arange(ndof)
    
    # Internal degrees of freedom condensed by the Schur complement.
    it_dofs = np.setdiff1d(active_dofs, if_dofs)

    # IT 1, IF 2.
    M11 = M[np.ix_(it_dofs, it_dofs)]
    M12 = M[np.ix_(it_dofs, if_dofs)]
    M21 = M[np.ix_(if_dofs, it_dofs)]
    M22 = M[np.ix_(if_dofs, if_dofs)]

    K11 = K[np.ix_(it_dofs, it_dofs)]
    K12 = K[np.ix_(it_dofs, if_dofs)]
    K21 = K[np.ix_(if_dofs, it_dofs)]
    K22 = K[np.ix_(if_dofs, if_dofs)]

    C11 = C[np.ix_(it_dofs, it_dofs)]
    C12 = C[np.ix_(it_dofs, if_dofs)]
    C21 = C[np.ix_(if_dofs, it_dofs)]
    C22 = C[np.ix_(if_dofs, if_dofs)]

    # --------------------------------------------------
    # Offline RTHS implementation note.
    # --------------------------------------------------
    num_freqs = len(freq_array)
    num_if = len(if_dofs)
    Zn_array = np.zeros((num_freqs, num_if, num_if), dtype=np.complex128)

    for i, freq in enumerate(freq_array):
        omega = 2.0 * np.pi * freq
        j_omega = 1j * omega
        omega2 = omega**2

        Z11 = -omega2 * M11 + j_omega * C11 + K11
        Z12 = -omega2 * M12 + j_omega * C12 + K12
        Z21 = -omega2 * M21 + j_omega * C21 + K21
        Z22 = -omega2 * M22 + j_omega * C22 + K22

        # Z_N = Z22 - Z21 * (Z11 \ Z12.
        Y = solve(Z11, Z12)
        Zn_array[i, :, :] = Z22 - Z21 @ Y

    return Zn_array

# Offline RTHS implementation note.
# Frequency-response-function verification step.
# Offline RTHS implementation note.
if __name__ == "__main__":
    # Offline RTHS numerical workflow step.
    L = 7.0
    n_elems = 10
    E = 2e11
    rho = 7800
    width = 0.3
    thickness = 0.005
    freqs = np.linspace(1, 100, 500) 
    alpha = 0.01
    beta = 0.01

    # Offline RTHS numerical workflow step.
    Zn_results = compute_dirichlet_interface_dynamic_stiffness(
        L=L, n_elems=n_elems, E=E, rho=rho, width=width, thickness=thickness,
        freq_array=freqs, alpha=alpha, beta=beta
    )

    # Offline RTHS implementation note.
    # Frequency-response-function verification step.
    # Offline RTHS implementation note.
    n_nodes = n_elems + 1
    ndof = 2 * n_nodes
    dx = L / n_elems
    A = width * thickness
    I = (width * thickness**3) / 12.0

    Ke = (E * I / dx**3) * np.array([
        [12, 6*dx, -12, 6*dx], [6*dx, 4*dx**2, -6*dx, 2*dx**2],
        [-12, -6*dx, 12, -6*dx], [6*dx, 2*dx**2, -6*dx, 4*dx**2]
    ])
    Me = (rho * A * dx / 420.0) * np.array([
        [156, 22*dx, 54, -13*dx], [22*dx, 4*dx**2, 13*dx, -3*dx**2],
        [54, 13*dx, 156, -22*dx], [-13*dx, -3*dx**2, -22*dx, 4*dx**2]
    ])

    K_free = np.zeros((ndof, ndof))
    M_free = np.zeros((ndof, ndof))
    for e in range(n_elems):
        dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
        K_free[np.ix_(dofs, dofs)] += Ke
        M_free[np.ix_(dofs, dofs)] += Me
    C_free = alpha * M_free + beta * K_free

    # Offline RTHS implementation note.
    # Frequency-response-function verification step.
    # Offline RTHS implementation note.
    H_condensed_11 = np.zeros(len(freqs), dtype=np.complex128)
    H_global_11 = np.zeros(len(freqs), dtype=np.complex128)

    # Offline RTHS numerical workflow step.
    F_cond = np.array([1.0, 0.0]) # Offline RTHS numerical workflow step.
    F_global = np.zeros(ndof)
    F_global[2*n_elems] = 1.0     # Condensed dynamic-stiffness assembly and verification step.

    for i, freq in enumerate(freqs):
        omega = 2.0 * np.pi * freq
        j_omega = 1j * omega
        omega2 = omega**2

        # Frequency-response-function verification step.
        Zn = Zn_results[i]
        X_cond = solve(Zn, F_cond)
        H_condensed_11[i] = X_cond[0] # Offline RTHS numerical workflow step.

        # Frequency-response-function verification step.
        Z_global = -omega2 * M_free + j_omega * C_free + K_free
        X_global = solve(Z_global, F_global)
        H_global_11[i] = X_global[2*n_elems]  # Offline RTHS numerical workflow step.

    # Offline RTHS numerical workflow step.
    max_error = np.max(np.abs(H_global_11 - H_condensed_11))
    print(f"Maximum absolute error (global vs condensed): {max_error:.4e}")
    if max_error < 1e-10:
        print('Offline RTHS status message.')

    # Offline RTHS implementation note.
    # Bode Plot.
    # Offline RTHS implementation note.
    plt.figure(figsize=(10, 8))

    # Offline RTHS numerical workflow step.
    plt.subplot(2, 1, 1)
    plt.plot(freqs, 20 * np.log10(np.abs(H_global_11)), 'b-', linewidth=4, alpha=0.6, label='Global FRF (Free-Free Beam)')
    plt.plot(freqs, 20 * np.log10(np.abs(H_condensed_11)), 'r--', linewidth=2, label='Condensed FRF ($\\mathbf{Z}_{\\mathrm{N}}^{-1}$ at Right End)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Driving Point FRF Comparison at Right Interface')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()

    # Offline RTHS numerical workflow step.
    plt.subplot(2, 1, 2)
    plt.plot(freqs, np.angle(H_global_11, deg=True), 'b-', linewidth=4, alpha=0.6)
    plt.plot(freqs, np.angle(H_condensed_11, deg=True), 'r--', linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (Degrees)')
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
