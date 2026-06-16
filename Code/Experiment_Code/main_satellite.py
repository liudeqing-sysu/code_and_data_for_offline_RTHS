import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from aitken_satellite import aitken_coupling_satellite
import re
from dobot_api import DobotApiDashboard

class DobotActivate:
    def __init__(self, ip):
        self.ip = ip
        self.dashboardPort = 29999
        self.dashboard = None
        

    def parseResultId(self, valueRecv):
        # Parse ErrorID from the robotic-arm TCP response.
        if "Not Tcp" in valueRecv:
            print("Control Mode Is Not Tcp")
            return [1]
        return [int(num) for num in re.findall(r'-?\d+', valueRecv)] or [2]


    def start(self):
        # Robotic-arm-based offline RTHS workflow step.
        self.dashboard = DobotApiDashboard(self.ip, self.dashboardPort)
        if self.parseResultId(self.dashboard.EnableRobot(2, 0, 0, 0, 1))[0] != 0: # Robotic-arm TCP parsing and validation step.
            print('Robotic-arm workflow status message.')
            return
        print('Robotic-arm workflow status message.')


    def parse_robot_response(self, recv_msg):
        """Parse a robotic-arm TCP response.
        
            The function returns the command ErrorID and a six-component pose, joint-angle, or force vector. Parsing failure returns a nonzero ErrorID and [0.0] * 6.
        
        """
        # Robotic-arm TCP parsing and validation step.
        clean_msg = recv_msg.strip().rstrip(';').replace('\n', '').replace('\r', '')
        
        # Parse ErrorID from the robotic-arm TCP response.
        try:
            error_id_str, rest_msg = clean_msg.split(',', 1)  # Robotic-arm-based offline RTHS workflow step.
            error_id = int(error_id_str.strip())
        except (ValueError, IndexError):
            # Robotic-arm TCP parsing and validation step.
            return (1, [0.0]*6)
        
        # Extract the braced data segment from the robotic-arm response.
        data_match = re.search(r'\{([^}]+)\}', rest_msg)  # Extract the braced data segment from the robotic-arm response.
        if not data_match:
            # Extract the braced data segment from the robotic-arm response.
            return (error_id, [0.0]*6)
        
        # Robotic-arm-based offline RTHS workflow step.
        data_str = data_match.group(1).strip()
        data_parts = [p.strip() for p in data_str.split(',')]
        
        # Robotic-arm TCP parsing and validation step.
        if len(data_parts) != 6:
            raise ValueError(f"Data dimension error: expected 6 values, got {len(data_parts)}; raw data segment='{data_str}'")
        
        # Return zero-filled defaults when parsing fails.
        data_list = []
        for part in data_parts:
            try:
                data_list.append(float(part))
            except ValueError:
                raise ValueError(f"Data format error: failed to convert raw value='{part}' to float; full data segment='{data_str}'")

        # Robotic-arm TCP parsing and validation step.
        return (error_id, data_list)


def run_satellite_hybrid_experiment(dobot, exp_config):
    (
        Fb_right,
        Mb_right,
        Fb_left,
        Mb_left,
        max_interface_rmse_history,
        right_interface_rmse_history,
        left_interface_rmse_history,
        U_body,
        t,
        right_interface_relaxation_history,
        left_interface_relaxation_history,
        delta_x_history,
        delta_rz_history,
        x_exp_meas,
        rz_exp_meas,
    ) = (
        aitken_coupling_satellite(
            body_params=exp_config["body_params"],
            dt=exp_config["dt"],
            total_time=exp_config["T"],
            tol=exp_config["tol"],
            max_iter=exp_config["max_iter"],
            relaxation_ini=exp_config.get("relaxation_ini", 0.7),
            adaptive_relaxation=exp_config.get("adaptive_relaxation", True),
            relaxation_min=exp_config.get("relaxation_min", 0.01),
            relaxation_max=exp_config.get("relaxation_max", 1.0),
            ext_force_time=exp_config["ext_force_time"],
            ext_moment_time=exp_config["ext_moment_time"],
            ext_pos=exp_config["ext_pos"],
            dobot=dobot,
            exp_ref=exp_config["exp_ref"],
            force_filter_cutoff_hz=exp_config.get("force_filter_cutoff_hz", 5.0),
            iteration_output_dir=exp_config.get("iteration_output_dir"),
        )
    )

    return {
        "Fb_right": Fb_right,
        "Mb_right": Mb_right,
        "Fb_left": Fb_left,
        "Mb_left": Mb_left,
        "max_interface_rmse_history": max_interface_rmse_history,
        "right_interface_rmse_history": right_interface_rmse_history,
        "left_interface_rmse_history": left_interface_rmse_history,
        "U_body": U_body,
        "time": t,
        "ext_force_time": exp_config["ext_force_time"],
        "ext_moment_time": exp_config["ext_moment_time"],
        "right_interface_relaxation_history": right_interface_relaxation_history,
        "left_interface_relaxation_history": left_interface_relaxation_history,
        "delta_x_history": delta_x_history,
        "delta_rz_history": delta_rz_history,
        "x_exp_meas": x_exp_meas,
        "rz_exp_meas": rz_exp_meas,
    }


def save_satellite_results(results, exp_config, output_dir):
    output_dir.mkdir(exist_ok=True)

    t = results["time"]
    U_body = results["U_body"]
    Fb_right = results["Fb_right"]
    Mb_right = results["Mb_right"]
    Fb_left = results["Fb_left"]
    Mb_left = results["Mb_left"]
    max_interface_rmse_history = results["max_interface_rmse_history"]
    right_interface_rmse_history = results["right_interface_rmse_history"]
    left_interface_rmse_history = results["left_interface_rmse_history"]
    right_interface_relaxation_history = results["right_interface_relaxation_history"]
    left_interface_relaxation_history = results["left_interface_relaxation_history"]
    x_exp_meas = results["x_exp_meas"]
    rz_exp_meas = results["rz_exp_meas"]
    ext_force_time = results["ext_force_time"]
    ext_moment_time = results["ext_moment_time"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].semilogy(range(1, len(right_interface_rmse_history) + 1),
                     right_interface_rmse_history, "b-o", label="right interface RMSE")
    axes[0].semilogy(range(1, len(left_interface_rmse_history) + 1),
                     left_interface_rmse_history, "c--s", label="left interface RMSE")
    axes[0].semilogy(range(1, len(max_interface_rmse_history) + 1),
                     max_interface_rmse_history, "k:", label="max interface RMSE")
    axes[0].axhline(exp_config["tol"], color="r", linestyle="--", label="Tolerance")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Satellite Hybrid Coupling Convergence")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(range(1, len(right_interface_relaxation_history) + 1),
                 right_interface_relaxation_history, "b-o", label="right interface omega")
    axes[1].plot(range(1, len(left_interface_relaxation_history) + 1),
                 left_interface_relaxation_history, "c--s", label="left interface omega")
    axes[1].set_xlabel("Aitken update")
    axes[1].set_ylabel("Relaxation factor")
    axes[1].set_title("Adaptive Relaxation Factor")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(output_dir / "01_convergence_history.svg", bbox_inches="tight")

    max_len = max(
        len(max_interface_rmse_history),
        len(right_interface_relaxation_history),
        len(left_interface_relaxation_history),
    )
    convergence_data = np.full((max_len, 6), np.nan)
    convergence_data[:, 0] = np.arange(1, max_len + 1)
    convergence_data[:len(right_interface_rmse_history), 1] = right_interface_rmse_history
    convergence_data[:len(left_interface_rmse_history), 2] = left_interface_rmse_history
    convergence_data[:len(max_interface_rmse_history), 3] = max_interface_rmse_history
    convergence_data[:len(right_interface_relaxation_history), 4] = right_interface_relaxation_history
    convergence_data[:len(left_interface_relaxation_history), 5] = left_interface_relaxation_history
    np.savetxt(
        output_dir / "convergence_history.csv",
        convergence_data,
        delimiter=",",
        header=(
            "index,right_interface_rmse,left_interface_rmse,"
            "max_interface_rmse,right_interface_relaxation,"
            "left_interface_relaxation"
        ),
        comments="",
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(t, Fb_right, "b-", label="right")
    axes[0].plot(t, Fb_left, "c--", label="left")
    axes[0].set_ylabel("Force (N)")
    axes[0].set_title("Solar-Array Interface Force")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, Mb_right, "r-", label="right")
    axes[1].plot(t, Mb_left, "m--", label="left")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Moment (N*m)")
    axes[1].set_title("Solar-Array Interface Moment")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(output_dir / "02_interface_force_moment.svg", bbox_inches="tight")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(t, ext_force_time, "k-")
    axes[0].set_ylabel("Force (N)")
    axes[0].set_title("Low-Frequency External Body Force")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, ext_moment_time, "k-")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Moment (N*m)")
    axes[1].set_title("Low-Frequency External Body Moment")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "02b_external_excitation.svg", bbox_inches="tight")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(t, U_body[0, :], "b-", label="Numerical right interface y")
    axes[0].plot(t, x_exp_meas[0, :], "b--", label="Experimental right interface x")
    axes[0].plot(t, U_body[2, :], "c-", label="Numerical left interface y")
    axes[0].plot(t, x_exp_meas[1, :], "c--", label="Experimental left interface x")
    axes[0].plot(t, U_body[4, :], "k:", label="Body COM y")
    axes[0].set_ylabel("Displacement (m)")
    axes[0].set_title("Interface Displacement: Numerical y vs Experimental x")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, U_body[1, :], "b-", label="Numerical right interface rz")
    axes[1].plot(t, rz_exp_meas[0, :], "b--", label="Experimental right interface rz")
    axes[1].plot(t, U_body[3, :], "c-", label="Numerical left interface rz")
    axes[1].plot(t, rz_exp_meas[1, :], "c--", label="Experimental left interface rz")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Rotation (rad)")
    axes[1].set_title("Interface Rotation")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(output_dir / "03_interface_response.svg", bbox_inches="tight")

    csv_data = np.column_stack((
        t,
        Fb_right,
        Mb_right,
        Fb_left,
        Mb_left,
        ext_force_time,
        ext_moment_time,
        U_body[0, :],
        x_exp_meas[0, :],
        U_body[1, :],
        rz_exp_meas[0, :],
        U_body[2, :],
        U_body[3, :],
        x_exp_meas[1, :],
        rz_exp_meas[1, :],
        U_body[4, :],
        U_body[5, :],
    ))
    header = (
        "Time(s),Fb_Right(N),Mb_Right(Nm),Fb_Left(N),Mb_Left(Nm),"
        "Ext_Force_Body(N),Ext_Moment_Body(Nm),"
        "Y_Num_Right_Interface(m),X_Exp_Right_Interface(m),"
        "Rz_Num_Right_Interface(rad),Rz_Exp_Right_Interface(rad),"
        "Y_Num_Left_Interface(m),Rz_Num_Left_Interface(rad),"
        "X_Exp_Left_Interface(m),Rz_Exp_Left_Interface(rad),"
        "Y_Body_COM(m),Rz_Body(rad)"
    )
    np.savetxt(output_dir / "data_time_series.csv", csv_data, delimiter=",", header=header, comments="")

    plot_iteration_error_history(results, exp_config, output_dir)

    if results["delta_x_history"] and results["delta_rz_history"]:
        # Each residual vector is stored as [right time history, left time
        # history]. Split before exporting so the CSV row count remains nsteps.
        nsteps = len(t)
        csv_cols = [t]
        header_cols = ["Time(s)"]
        for i, delta_x in enumerate(results["delta_x_history"], start=1):
            csv_cols.extend([delta_x[:nsteps], delta_x[nsteps:]])
            header_cols.extend([
                f"Iter{i}_Right_X_Displacement_Delta(m)",
                f"Iter{i}_Left_X_Displacement_Delta(m)",
            ])
        for i, delta_rz in enumerate(results["delta_rz_history"], start=1):
            csv_cols.extend([delta_rz[:nsteps], delta_rz[nsteps:]])
            header_cols.extend([
                f"Iter{i}_Right_Rotation_Delta(rad)",
                f"Iter{i}_Left_Rotation_Delta(rad)",
            ])
        np.savetxt(output_dir / "data_iteration_delta.csv",
                   np.column_stack(csv_cols), delimiter=",",
                   header=",".join(header_cols), comments="")


def plot_iteration_error_history(results, exp_config, output_dir):
    """
    Plot iteration residual histories available from the experiment.

    The experiment can provide interface x/rz residuals, but not full solar-array
    deformation fields, so this mirrors the numerical residual plots only for
    interface quantities.
    """
    t = results["time"]
    delta_x_history = results["delta_x_history"]
    delta_rz_history = results["delta_rz_history"]
    if not delta_x_history or not delta_rz_history:
        return

    nsteps = len(t)
    n_iter = len(delta_x_history)
    max_dx_right = []
    max_dx_left = []
    max_rz_right = []
    max_rz_left = []
    for dx, drz in zip(delta_x_history, delta_rz_history):
        max_dx_right.append(np.max(np.abs(dx[:nsteps])))
        max_dx_left.append(np.max(np.abs(dx[nsteps:])))
        max_rz_right.append(np.max(np.abs(drz[:nsteps])))
        max_rz_left.append(np.max(np.abs(drz[nsteps:])))

    iter_axis = np.arange(1, n_iter + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    axes[0].semilogy(iter_axis, results["right_interface_rmse_history"],
                     "b-o", label="right interface RMSE")
    axes[0].semilogy(iter_axis, results["left_interface_rmse_history"],
                     "c--s", label="left interface RMSE")
    axes[0].semilogy(iter_axis, results["max_interface_rmse_history"],
                     "k:", label="max interface RMSE")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Aitken Iteration Error Trend")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].semilogy(iter_axis, max_dx_right, "b-o", label="right |delta x|max")
    axes[1].semilogy(iter_axis, max_dx_left, "c--s", label="left |delta x|max")
    axes[1].set_ylabel("Displacement error (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].semilogy(iter_axis, max_rz_right,
                    "r-o", label="right |delta rz|max")
    axes[2].semilogy(iter_axis, max_rz_left,
                    "m--s", label="left |delta rz|max")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Rotation error (rad)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    plt.tight_layout()
    fig.savefig(output_dir / "04_iteration_error_trend.svg", bbox_inches="tight")
    plt.close(fig)

    selected_iters = np.unique(
        np.rint(np.linspace(1, n_iter, min(4, n_iter))).astype(int)
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for it in selected_iters:
        dx = delta_x_history[it - 1]
        axes[0].plot(t, dx[:nsteps], label=f"right iter {it}")
        axes[1].plot(t, dx[nsteps:], label=f"left iter {it}")
    axes[0].set_ylabel("Right delta x (m)")
    axes[1].set_ylabel("Left delta x (m)")
    axes[1].set_xlabel("Time (s)")
    axes[0].set_title("Interface Displacement Residual During Iterations")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "05_iteration_displacement_error.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for it in selected_iters:
        drz = delta_rz_history[it - 1]
        axes[0].plot(t, drz[:nsteps], label=f"right iter {it}")
        axes[1].plot(t, drz[nsteps:], label=f"left iter {it}")
    axes[0].set_ylabel("Right delta rz (rad)")
    axes[1].set_ylabel("Left delta rz (rad)")
    axes[1].set_xlabel("Time (s)")
    axes[0].set_title("Interface Rotation Residual During Iterations")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "06_iteration_rotation_error.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    dobot = DobotActivate("192.168.5.1")
    dobot.start()

    dt = 0.025
    T = 10.0
    nsteps = int(np.floor(T / dt)) + 1
    t = np.linspace(0.0, T, nsteps)

    # Smooth low-frequency external excitation. All components are below 2 Hz,
    # matching the numerical preview input style.
    ext_force_time = (
        500.0 * np.cos(2 * np.pi * 0.25 * t)
        - 100 * np.cos(2 * np.pi * 0.90 * t - np.pi/2)
        + 50.0 * np.cos(2 * np.pi * 1.60 * t + np.pi)
    )
    ext_moment_time = (
        -70 * np.cos(2 * np.pi * 0.25 * t + np.pi)
        - 20 * np.cos(2 * np.pi * 1.10 * t)
        - 10 * np.cos(2 * np.pi * 1.80 * t - np.pi/2)
    )
    ext_force_time -= np.mean(ext_force_time)
    ext_moment_time -= np.mean(ext_moment_time)

    exp_config = {
        "body_params": {
            "body_mass": 1500.0,
            "body_inertia_z": 300.0,
            "interface_offset": 0.5,
            "translational_stiffness": 0.0,
            "rotational_stiffness": 0.0,
            "translational_damping": 0.0,
            "rotational_damping": 0.0,
            # Rotation residual scale length. Use the full deployed span:
            # left wing + root-to-root satellite body span + right wing.
            "characteristic_length": 2 * 1.2 + 2 * 0.5,
        },
        "dt": dt,
        "T": T,
        "tol": 1e-3,
        "max_iter": 40,
        "relaxation_ini": 0.1,
        "adaptive_relaxation": True,
        "relaxation_min": 0.01,
        "relaxation_max": 1.0,
        # Low-pass cutoff for measured interface force and moment histories.
        # Components above 5 Hz are filtered out before body-substructure solve.
        "force_filter_cutoff_hz": 5.0,
        "iteration_output_dir": str(Path("./satellite_hybrid_results") / "iteration_results"),
        "ext_force_time": ext_force_time,
        "ext_moment_time": ext_moment_time,
        "ext_pos": 0.0,
        # Robot reference pose: the same initial pose is used for both
        # right-side and left-side solar-array tests.
        "exp_ref": {
            "x": -120.0,
            "y": -650.0,
            "z": 550,
            "rx": 90.0,
            "ry": 0.0,
            "rz": 0.0,
        },
    }

    results = run_satellite_hybrid_experiment(dobot, exp_config)
    output_dir = Path("./satellite_hybrid_results")
    save_satellite_results(results, exp_config, output_dir)

    print("\nSatellite hybrid experiment finished.")
    print(f"Iterations: {len(results['max_interface_rmse_history'])}")
    print(f"Final RMSE: {results['max_interface_rmse_history'][-1]:.3e}")
    print(f"Results saved to: {output_dir}")
