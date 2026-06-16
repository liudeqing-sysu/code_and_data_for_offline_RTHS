import numpy as np
from scipy.linalg import lu_factor, lu_solve

def solve_force_driven_beam(
    L, n_elems, E, rho, width, thickness,
    F_boundary=0.0, M_boundary=0.0, ext_force=0.0, ext_pos=0.0,
    dt=1e-2, total_time=30.0, 
    alpha=0.01, beta=0.005,
    gamma_newmark=0.5, beta_newmark=0.25,
):
    """
    Euler–Bernoulli beam transverse vibration
    Pure acceleration-based implicit Newmark-beta formulation
    Right end is fixed (zero displacement and rotation)
    Left end is free with specified force and moment time histories
    Unknown: a_{n+1}
    """

    # --------------------------------------------------
    # Basic setup
    # --------------------------------------------------
    n_nodes = n_elems + 1
    ndof = 2 * n_nodes
    nsteps = int(np.floor(total_time / dt)) + 1

    A = width * thickness
    I = width * thickness**3 / 12.0
    dx = L / n_elems

    # --------------------------------------------------
    # Element matrices
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Global assembly
    # --------------------------------------------------
    K = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))

    for e in range(n_elems):
        dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
        K[np.ix_(dofs, dofs)] += Ke
        M[np.ix_(dofs, dofs)] += Me

    C = alpha * M + beta * K

    # --------------------------------------------------
    # Fixed right-end boundary (Elimination Method)
    # --------------------------------------------------
    right_node = n_nodes - 1
    fixed_dofs = [2*right_node, 2*right_node + 1]
    
    # Degree-of-freedom reduction and boundary-condition handling.
    all_dofs = np.arange(ndof)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    # Offline RTHS numerical workflow step.
    K_free = K[np.ix_(free_dofs, free_dofs)]
    M_free = M[np.ix_(free_dofs, free_dofs)]
    C_free = C[np.ix_(free_dofs, free_dofs)]

    # --------------------------------------------------
    # External forces
    # --------------------------------------------------
    F_ext = np.zeros((ndof, nsteps))
    left_dofs = (0, 1)

    Fb = np.full(nsteps, float(F_boundary)) if np.isscalar(F_boundary) \
         else np.asarray(F_boundary, dtype=float)
    Mb = np.full(nsteps, float(M_boundary)) if np.isscalar(M_boundary) \
         else np.asarray(M_boundary, dtype=float)

    F_ext[left_dofs[0], :] += Fb
    F_ext[left_dofs[1], :] += Mb

    Fe = np.full(nsteps, float(ext_force)) if np.isscalar(ext_force) \
         else np.asarray(ext_force, dtype=float)

    node_idx = int(np.round((L-ext_pos) / dx))
    node_idx = max(0, min(n_nodes - 1, node_idx))
    F_ext[2 * node_idx, :] += Fe

    # --------------------------------------------------
    # State variables
    # --------------------------------------------------
    U = np.zeros((ndof, nsteps))
    V = np.zeros((ndof, nsteps))
    A_acc = np.zeros((ndof, nsteps))

    # Initial acceleration from equilibrium.
    A_acc[free_dofs, 0] = np.linalg.solve(
        M_free,
        F_ext[free_dofs, 0] - C_free @ V[free_dofs, 0] - K_free @ U[free_dofs, 0]
    )

    # --------------------------------------------------
    # Pure acceleration Newmark effective matrix.
    # --------------------------------------------------
    K_eff_free = (
        M_free
        + gamma_newmark * dt * C_free
        + beta_newmark * dt**2 * K_free
    )

    LU_free, piv_free = lu_factor(K_eff_free)

    # --------------------------------------------------
    # Time integration
    # --------------------------------------------------
    for i in range(nsteps - 1):

        # Predictor step for the Newmark time-integration scheme.
        U_pred = (
            U[:, i]
            + dt * V[:, i]
            + (0.5 - beta_newmark) * dt**2 * A_acc[:, i]
        )

        V_pred = (
            V[:, i]
            + (1.0 - gamma_newmark) * dt * A_acc[:, i]
        )

        # Effective load vector for the reduced free-degree system.
        F_eff_free = (
            F_ext[free_dofs, i+1]
            - C_free @ V_pred[free_dofs]
            - K_free @ U_pred[free_dofs]
        )

        # Solve for acceleration.
        A_acc[free_dofs, i+1] = lu_solve((LU_free, piv_free), F_eff_free)

        # Corrector step for the Newmark time-integration scheme.
        U[:, i+1] = U_pred + beta_newmark * dt**2 * A_acc[:, i+1]
        V[:, i+1] = V_pred + gamma_newmark * dt * A_acc[:, i+1]

        if np.max(np.abs(U[:, i+1])) > 1e6:
            raise RuntimeError(
                f"Large displacement instability at step {i+1}"
            )

    boundary_y = U[left_dofs[0], :]
    boundary_rz = U[left_dofs[1], :]

    return boundary_y, boundary_rz, U