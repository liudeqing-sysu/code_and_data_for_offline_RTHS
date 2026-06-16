import numpy as np
from scipy.linalg import lu_factor, lu_solve


def solve_displacement_driven_beam(
    L, n_elems, E, rho, width, thickness,
    ext_force=0.0, ext_pos=0.0,
    free_y=0.0, free_rz=0.0,
    dt=1e-2, total_time=30.0,
    alpha=0.01, beta=0.005,
    gamma=0.5, beta_newmark=0.25,
    enforce_initial_rest=True,
    ramp_time=0.0,
):
    """
    Euler–Bernoulli beam transverse vibration
    Pure acceleration-based implicit Newmark-beta formulation
    Both ends are free with specified displacement and moment time histories
    at the right end
    """
    n_nodes = n_elems + 1
    ndof = 2 * n_nodes
    nsteps = int(np.floor(total_time / dt)) + 1

    A = width * thickness
    I = width * thickness**3 / 12.0

    dx = L / n_elems
    Ke = (E * I / dx**3) * np.array([
        [12, 6*dx, -12, 6*dx],
        [6*dx, 4*dx**2, -6*dx, 2*dx**2],
        [-12, -6*dx, 12, -6*dx],
        [6*dx, 2*dx**2, -6*dx, 4*dx**2],
    ])

    Me = (rho * A * dx / 420.0) * np.array([
        [156, 22*dx, 54, -13*dx],
        [22*dx, 4*dx**2, 13*dx, -3*dx**2],
        [54, 13*dx, 156, -22*dx],
        [-13*dx, -3*dx**2, -22*dx, 4*dx**2],
    ])

    K = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))
    for e in range(n_elems):
        dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
        K[np.ix_(dofs, dofs)] += Ke
        M[np.ix_(dofs, dofs)] += Me

    C = alpha * M + beta * K

    # external force
    F_ext = np.zeros((ndof, nsteps))
    Fe = np.full(nsteps, ext_force) if np.isscalar(ext_force) else np.asarray(ext_force)
    node_idx = int(np.clip(np.round(ext_pos / dx), 0, n_elems))
    F_ext[2*node_idx, :] = Fe

    # prescribed DOFs (right end)
    right_dofs = [2*n_elems, 2*n_elems + 1]
    y_pres = np.full(nsteps, free_y) if np.isscalar(free_y) else np.asarray(free_y)
    rz_pres = np.full(nsteps, free_rz) if np.isscalar(free_rz) else np.asarray(free_rz)

    if ramp_time > 0:
        Nr = int(np.ceil(ramp_time / dt))
        w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, Nr)))
        y_pres[:Nr] *= w
        rz_pres[:Nr] *= w

    V_pres_y = np.gradient(y_pres, dt)
    V_pres_rz = np.gradient(rz_pres, dt)
    A_pres_y = np.gradient(V_pres_y, dt)
    A_pres_rz = np.gradient(V_pres_rz, dt)

    if enforce_initial_rest:
        V_pres_y[0] = V_pres_rz[0] = 0.0
        A_pres_y[0] = A_pres_rz[0] = 0.0

    U = np.zeros((ndof, nsteps))
    V = np.zeros((ndof, nsteps))
    A_acc = np.zeros((ndof, nsteps))

    U[right_dofs[0], :] = y_pres
    U[right_dofs[1], :] = rz_pres
    V[right_dofs[0], :] = V_pres_y
    V[right_dofs[1], :] = V_pres_rz
    A_acc[right_dofs[0], :] = A_pres_y
    A_acc[right_dofs[1], :] = A_pres_rz

    # DOFs.
    known_dofs = right_dofs
    free_dofs = [i for i in range(ndof) if i not in known_dofs]

    M_ff = M[np.ix_(free_dofs, free_dofs)]
    M_fk = M[np.ix_(free_dofs, known_dofs)]
    C_ff = C[np.ix_(free_dofs, free_dofs)]
    C_fk = C[np.ix_(free_dofs, known_dofs)]
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_fk = K[np.ix_(free_dofs, known_dofs)]

    K_eff = M_ff + gamma*dt*C_ff + beta_newmark*dt**2*K_ff
    LU, piv = lu_factor(K_eff)

    # consistent initial acceleration
    rhs0 = (
        F_ext[free_dofs, 0]
        - M_fk @ A_acc[known_dofs, 0]
        - C_ff @ V[free_dofs, 0]
        - C_fk @ V[known_dofs, 0]
        - K_ff @ U[free_dofs, 0]
        - K_fk @ U[known_dofs, 0]
    )
    A_acc[free_dofs, 0] = np.linalg.solve(M_ff, rhs0)

    reaction_force = np.zeros(nsteps)
    reaction_moment = np.zeros(nsteps)

    for i in range(nsteps - 1):
        U_pred = (
            U[free_dofs, i]
            + dt * V[free_dofs, i]
            + (0.5 - beta_newmark) * dt**2 * A_acc[free_dofs, i]
        )
        V_pred = V[free_dofs, i] + (1 - gamma) * dt * A_acc[free_dofs, i]

        rhs = (
            F_ext[free_dofs, i+1]
            - M_fk @ A_acc[known_dofs, i+1]
            - C_ff @ V_pred
            - C_fk @ V[known_dofs, i+1]
            - K_ff @ U_pred
            - K_fk @ U[known_dofs, i+1]
        )

        A_acc[free_dofs, i+1] = lu_solve((LU, piv), rhs)
        U[free_dofs, i+1] = U_pred + beta_newmark * dt**2 * A_acc[free_dofs, i+1]
        V[free_dofs, i+1] = V_pred + gamma * dt * A_acc[free_dofs, i+1]

        rf = right_dofs[0]
        rm = right_dofs[1]
        reaction_force[i+1] = M[rf] @ A_acc[:, i+1] + C[rf] @ V[:, i+1] + K[rf] @ U[:, i+1]
        reaction_moment[i+1] = M[rm] @ A_acc[:, i+1] + C[rm] @ V[:, i+1] + K[rm] @ U[:, i+1]

    return -reaction_force, -reaction_moment, U