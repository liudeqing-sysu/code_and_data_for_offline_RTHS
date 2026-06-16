import numpy as np
from scipy.linalg import lu_factor, lu_solve


def solve_satellite_body_substructure(
    body_mass,
    body_inertia_z,
    interface_offset,
    F_boundary=0.0,
    M_boundary=0.0,
    ext_force=0.0,
    ext_pos=0.0,
    ext_moment=0.0,
    interface_load_scale=1.0,
    dt=1e-2,
    total_time=30.0,
    translational_stiffness=0.0,
    rotational_stiffness=0.0,
    translational_damping=0.0,
    rotational_damping=0.0,
    alpha=0.0,
    beta=0.0,
    gamma_newmark=0.5,
    beta_newmark=0.25,
):
    """
    Satellite-body numerical substructure in the y-rz plane.

    The satellite main body is modeled as a rigid bus with two generalized
    coordinates: y_body and rz_body. The solar-array interface is located a
    signed distance interface_offset away from the body mass center along the
    satellite x-axis, so

        y_interface = y_body + interface_offset * rz_body

    The same offset also converts the interface shear force into a body moment:

        M_body += interface_offset * F_boundary

    If one physical solar array in the experiment represents two symmetric
    arrays on the spacecraft, set interface_load_scale=2.0.

    Returns
    -------
    boundary_y, boundary_rz, U
        U rows are [interface_y, interface_rz, body_y, body_rz].
    """
    if body_mass <= 0.0:
        raise ValueError("body_mass must be positive")
    if body_inertia_z <= 0.0:
        raise ValueError("body_inertia_z must be positive")

    nsteps = int(np.floor(total_time / dt)) + 1

    def as_history(value, name):
        if np.isscalar(value):
            return np.full(nsteps, float(value))
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size != nsteps:
            raise ValueError(f"{name} must have length nsteps")
        return arr

    Fb = as_history(F_boundary, "F_boundary") * float(interface_load_scale)
    Mb = as_history(M_boundary, "M_boundary") * float(interface_load_scale)
    Fe = as_history(ext_force, "ext_force")
    Me = as_history(ext_moment, "ext_moment")

    M = np.diag([body_mass, body_inertia_z])
    K = np.diag([translational_stiffness, rotational_stiffness])
    C = np.diag([translational_damping, rotational_damping]) + alpha * M + beta * K

    Q = np.zeros((2, nsteps))
    Q[0, :] = Fb + Fe
    Q[1, :] = Mb + interface_offset * Fb + Me + ext_pos * Fe

    q = np.zeros((2, nsteps))
    qd = np.zeros((2, nsteps))
    qdd = np.zeros((2, nsteps))

    qdd[:, 0] = np.linalg.solve(M, Q[:, 0] - C @ qd[:, 0] - K @ q[:, 0])

    K_eff = M + gamma_newmark * dt * C + beta_newmark * dt**2 * K
    LU, piv = lu_factor(K_eff)

    for i in range(nsteps - 1):
        q_pred = q[:, i] + dt * qd[:, i] + (0.5 - beta_newmark) * dt**2 * qdd[:, i]
        qd_pred = qd[:, i] + (1.0 - gamma_newmark) * dt * qdd[:, i]

        Q_eff = Q[:, i + 1] - C @ qd_pred - K @ q_pred
        qdd[:, i + 1] = lu_solve((LU, piv), Q_eff)

        q[:, i + 1] = q_pred + beta_newmark * dt**2 * qdd[:, i + 1]
        qd[:, i + 1] = qd_pred + gamma_newmark * dt * qdd[:, i + 1]

        if np.max(np.abs(q[:, i + 1])) > 1e6:
            raise RuntimeError(f"Large satellite-body response at step {i + 1}")

    boundary_y = q[0, :] + interface_offset * q[1, :]
    boundary_rz = q[1, :]
    U = np.vstack((boundary_y, boundary_rz, q))

    return boundary_y, boundary_rz, U


def solve_satellite_body_two_sided_substructure(
    body_mass,
    body_inertia_z,
    interface_offset,
    F_boundary_right=0.0,
    M_boundary_right=0.0,
    F_boundary_left=0.0,
    M_boundary_left=0.0,
    ext_force=0.0,
    ext_pos=0.0,
    ext_moment=0.0,
    dt=1e-2,
    total_time=30.0,
    translational_stiffness=0.0,
    rotational_stiffness=0.0,
    translational_damping=0.0,
    rotational_damping=0.0,
    alpha=0.0,
    beta=0.0,
    gamma_newmark=0.5,
    beta_newmark=0.25,
):
    """
    Satellite-body numerical substructure with two solar-array interfaces.

    Right and left solar-array roots are located at +interface_offset and
    -interface_offset from the body mass center:

        y_right = y_body + interface_offset * rz_body
        y_left  = y_body - interface_offset * rz_body

    Returns
    -------
    right_y, right_rz, left_y, left_rz, U
        U rows are [right_y, right_rz, left_y, left_rz, body_y, body_rz].
    """
    if body_mass <= 0.0:
        raise ValueError("body_mass must be positive")
    if body_inertia_z <= 0.0:
        raise ValueError("body_inertia_z must be positive")

    nsteps = int(np.floor(total_time / dt)) + 1
    a = float(interface_offset)

    def as_history(value, name):
        if np.isscalar(value):
            return np.full(nsteps, float(value))
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size != nsteps:
            raise ValueError(f"{name} must have length nsteps")
        return arr

    Fr = as_history(F_boundary_right, "F_boundary_right")
    Mr = as_history(M_boundary_right, "M_boundary_right")
    Fl = as_history(F_boundary_left, "F_boundary_left")
    Ml = as_history(M_boundary_left, "M_boundary_left")
    Fe = as_history(ext_force, "ext_force")
    Me = as_history(ext_moment, "ext_moment")

    # Free-floating spacecraft body: leave K=C=0 unless the experiment needs an
    # equivalent support, control-law stiffness, or identified damping model.
    M = np.diag([body_mass, body_inertia_z])
    K = np.diag([translational_stiffness, rotational_stiffness])
    C = np.diag([translational_damping, rotational_damping]) + alpha * M + beta * K

    Q = np.zeros((2, nsteps))
    Q[0, :] = Fr + Fl + Fe
    # Moment equilibrium about the body mass center. Right root is at +a, left
    # root is at -a, so the left shear-force moment contribution changes sign.
    Q[1, :] = Mr + a * Fr + Ml - a * Fl + Me + ext_pos * Fe

    q = np.zeros((2, nsteps))
    qd = np.zeros((2, nsteps))
    qdd = np.zeros((2, nsteps))

    qdd[:, 0] = np.linalg.solve(M, Q[:, 0] - C @ qd[:, 0] - K @ q[:, 0])

    K_eff = M + gamma_newmark * dt * C + beta_newmark * dt**2 * K
    LU, piv = lu_factor(K_eff)

    for i in range(nsteps - 1):
        q_pred = q[:, i] + dt * qd[:, i] + (0.5 - beta_newmark) * dt**2 * qdd[:, i]
        qd_pred = qd[:, i] + (1.0 - gamma_newmark) * dt * qdd[:, i]

        Q_eff = Q[:, i + 1] - C @ qd_pred - K @ q_pred
        qdd[:, i + 1] = lu_solve((LU, piv), Q_eff)

        q[:, i + 1] = q_pred + beta_newmark * dt**2 * qdd[:, i + 1]
        qd[:, i + 1] = qd_pred + gamma_newmark * dt * qdd[:, i + 1]

        if np.max(np.abs(q[:, i + 1])) > 1e6:
            raise RuntimeError(f"Large satellite-body response at step {i + 1}")

    right_y = q[0, :] + a * q[1, :]
    left_y = q[0, :] - a * q[1, :]
    right_rz = q[1, :]
    left_rz = q[1, :]
    U = np.vstack((right_y, right_rz, left_y, left_rz, q))

    return right_y, right_rz, left_y, left_rz, U
