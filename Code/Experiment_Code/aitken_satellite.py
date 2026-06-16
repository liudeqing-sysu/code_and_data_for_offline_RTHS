import csv
import os
import numpy as np
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
from time import time, sleep
from satellite_body import solve_satellite_body_two_sided_substructure


def transform_force_tool_to_user(force_data, pose_data):
    """Transform force and moment data from the tool frame to the user frame.
    
        Only the rotational transformation is applied; no translational correction is introduced. The input force and moment vector follows [Fx, Fy, Fz, Mx, My, Mz], and the pose vector follows [X, Y, Z, Rx, Ry, Rz] in mm and degrees using the Dobot ZYX Euler-angle convention.
    
    """

    # Robotic-arm-based offline RTHS workflow step.
    rx, ry, rz = np.radians(pose_data[3:6])

    # Robotic-arm-based offline RTHS workflow step.
    cx, cy, cz = np.cos([rx, ry, rz])
    sx, sy, sz = np.sin([rx, ry, rz])

    # Dobot ZYX roll-pitch-yaw rotation matrix.
    R = np.array([
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy,   cy*sx,            cy*cx]
    ])

    # Robotic-arm-based offline RTHS workflow step.
    F_tool = np.array(force_data[:3])
    M_tool = np.array(force_data[3:6])

    # Robotic-arm-based offline RTHS workflow step.
    F_user = R @ F_tool
    M_user = R @ M_tool

    # Robotic-arm experimental-substructure execution step.
    force_user = np.concatenate([F_user, M_user])

    return force_user


def _save_experimental_force_history(output_dir, filename, t_array, raw_force_history, transformed_force_history):
    """Save raw and transformed six-axis force histories to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    header = ['time_s'] + [f'raw_{label}' for label in labels] + [f'trans_{label}' for label in labels]

    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for i, t in enumerate(t_array):
            row = [float(t)]
            row += [float(raw_force_history[i, j]) for j in range(6)]
            row += [float(transformed_force_history[i, j]) for j in range(6)]
            writer.writerow(row)

    return filepath

def solve_experimental_substructure(
    dobot,
    x_drive,
    rz_drive,
    dt,
    total_time,
    ref_x, ref_y, ref_z,
    ref_rx, ref_ry, ref_rz,
    command_delay_steps=4,
):
    """Execute the robotic-arm-based experimental substructure.
    
        Given commanded displacement and rotation histories, the function drives the experimental substructure, records the reaction force and moment, and returns the measured displacement and rotation histories.
    
    """
    
    nsteps = len(x_drive)
    if command_delay_steps < 0:
        raise ValueError("command_delay_steps must be non-negative")
    if command_delay_steps >= nsteps:
        raise ValueError("command_delay_steps must be smaller than nsteps")

    x_drive_commanded = np.empty_like(x_drive)
    rz_drive_commanded = np.empty_like(rz_drive)
    if command_delay_steps == 0:
        x_drive_commanded[:] = x_drive
        rz_drive_commanded[:] = rz_drive
    else:
        x_drive_commanded[:nsteps-command_delay_steps] = x_drive[command_delay_steps:]
        x_drive_commanded[nsteps-command_delay_steps:] = x_drive[-1]
        rz_drive_commanded[:nsteps-command_delay_steps] = rz_drive[command_delay_steps:]
        rz_drive_commanded[nsteps-command_delay_steps:] = rz_drive[-1]
        print(f"Applying command delay of {command_delay_steps} steps ({command_delay_steps*dt:.3f}s); "
              f"last {command_delay_steps} steps repeat final command.")

    reaction_force = np.zeros(nsteps)
    reaction_moment = np.zeros(nsteps)
    exp_x_meas = np.zeros(nsteps)
    exp_rz_meas = np.zeros(nsteps)
    
    # Robotic-arm-based offline RTHS workflow step.
    joints_array = []
    pose_targets = []
    last_joints = None
    
    print("Step 1: Computing inverse kinematics for all time steps...")
    # Robotic-arm experimental-substructure execution step.

    for i in range(nsteps):
        # Robotic-arm-based offline RTHS workflow step.
        # Robotic-arm experimental-substructure execution step.
        X = ref_x + x_drive_commanded[i] * 1000.0  # m -> mm
        Y = ref_y 
        Z = ref_z
        Rx = ref_rx
        Ry = ref_ry
        Rz = ref_rz + np.degrees(rz_drive_commanded[i])  # rad -> deg
        pose_targets.append([X, Y, Z, Rx, Ry, Rz])

        # Robotic-arm-based offline RTHS workflow step.
        inv_kin_raw = dobot.dashboard.InverseKin(
            X=X, Y=Y, Z=Z, Rx=Rx, Ry=Ry, Rz=Rz,
            user=0, tool=0,
            useJointNear=0,
        )
        print(inv_kin_raw)
        # Robotic-arm-based offline RTHS workflow step.
        error_id, joints = dobot.parse_robot_response(inv_kin_raw)
        
        if error_id == 0 and len(joints) == 6:
            joints_array.append(joints)
            last_joints = joints
        else:
            # Robotic-arm-based offline RTHS workflow step.
            raise RuntimeError(f"Inverse kinematics failed at t={i*dt:.3f}s, error_id={error_id}")
    
    # Robotic-arm-based offline RTHS workflow step.
    print("\nStep 2: Previewing inverse kinematics solution...")
    
    # Convert inverse-kinematics results to an array with shape (nsteps, 6).
    joints_array_vis = np.array(joints_array)  # shape: (nsteps, 6)
    pose_targets_vis = np.array(pose_targets)  # shape: (nsteps, 6)
    t_array = np.linspace(0, total_time, nsteps)
    
    # Robotic-arm-based offline RTHS workflow step.
    fig0, axes0 = plt.subplots(6, 1, figsize=(12, 10))
    fig0.suptitle('Inverse Kinematics Target Pose Trajectory', fontsize=14, fontweight='bold')
    pose_names = ['X (mm)', 'Y (mm)', 'Z (mm)', 'Rx (°)', 'Ry (°)', 'Rz (°)']

    for k in range(6):
        ax = axes0[k]
        ax.plot(t_array, pose_targets_vis[:, k], 'C0-', linewidth=2, label=pose_names[k])
        ax.set_ylabel(pose_names[k], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        if k < 5:
            ax.set_xticklabels([])
    axes0[-1].set_xlabel('Time (s)', fontsize=10)
    plt.tight_layout()
    plt.show()

    # Six-panel joint-angle trajectory visualization.
    fig, axes = plt.subplots(6, 1, figsize=(12, 10))
    fig.suptitle('Robotic-arm workflow status message.', fontsize=14, fontweight='bold')
    
    joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    
    for j in range(6):
        ax = axes[j]
        ax.plot(t_array, joints_array_vis[:, j], 'b-', linewidth=2, label=f'{joint_names[j]} trajectory')
        ax.set_ylabel(f'{joint_names[j]} (°)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
        if j < 5:
            ax.set_xticklabels([])
    
    axes[-1].set_xlabel('Time (s)', fontsize=10)
    plt.tight_layout()
    plt.show()
    
    # Robotic-arm-based offline RTHS workflow step.
    print("\n" + "="*70)
    print('Robotic-arm workflow status message.')
    print("="*70)
    user_input = input('Robotic-arm workflow status message.').strip().lower()
    
    if user_input not in ['yes', 'y']:
        print('Robotic-arm workflow status message.')
        raise RuntimeError("User cancelled the robot motion after reviewing joint trajectories.")
    
    print('Robotic-arm workflow status message.')
    
    # Robotic-arm-based offline RTHS workflow step.
    init_joints = joints_array[0]
    print(f"Moving robot to initial reference pose (first step) joints: {init_joints}")
    result_init = dobot.dashboard.ServoJ(*init_joints,t=4)
    parts_init = result_init.split(',')
    error_id_init = int(parts_init[0]) if parts_init else -1
    if error_id_init != 0:
        raise RuntimeError(f"Failed to move to initial reference pose, error_id={error_id_init}")
    # give robot a short time to settle
    sleep(10)

    # Robotic-arm-based offline RTHS workflow step.

    # Robotic-arm-based offline RTHS workflow step.
    loop_start_time = time()

    # Robotic-arm-based offline RTHS workflow step.
    error_id_force_origin,force_origin = dobot.parse_robot_response(dobot.dashboard.GetForce())
    if error_id_force_origin == 0 and len(force_origin) == 6:
        print(f"Initial force data (tool frame): {force_origin}")
    else:
        raise RuntimeError(f"Failed to get initial force data, error_id={error_id_force_origin}")
    
    for i in range(nsteps):
        # Robotic-arm-based offline RTHS workflow step.
        current_step_time = i * dt
        
        # Robotic-arm-based offline RTHS workflow step.
        elapsed_time = time() - loop_start_time
        
        # Robotic-arm-based offline RTHS workflow step.
        wait_time = current_step_time - elapsed_time
        if wait_time > 0:
            sleep(wait_time)

        # Robotic-arm-based offline RTHS workflow step.
        target_joints = joints_array[i]
        # Robotic-arm-based offline RTHS workflow step.
        result = dobot.dashboard.ServoJ(*target_joints)
    
        # Extract ErrorID from the robotic-arm command response.
        parts = result.split(',')
        error_id = int(parts[0]) if parts else -1
        if error_id != 0:
            raise RuntimeError(f"Failed to send ServoJ command at t={i*dt:.3f}s, error_id={error_id}")

        # Robotic-arm-based offline RTHS workflow step.
        force_raw = dobot.dashboard.GetForce()
        
        error_id_force, force_data = dobot.parse_robot_response(force_raw)
        
        if error_id_force == 0 and len(force_data) == 6:

            # Robotic-arm-based offline RTHS workflow step.
            pose_raw = dobot.dashboard.GetPose(user=0, tool=0)
            error_id_pose, pose_data = dobot.parse_robot_response(pose_raw)

            if error_id_pose == 0 and len(pose_data) == 6:

                force_data[0] = force_data[0] - force_origin[0]
                force_data[1] = force_data[1] - force_origin[1] 
                force_data[2] = force_data[2] - force_origin[2] 
                force_data[3] = force_data[3] - force_origin[3] 
                force_data[4] = force_data[4] - force_origin[4]
                force_data[5] = force_data[5] - force_origin[5] 
                # Robotic-arm experimental-substructure execution step.
                # force_data[1] = 0 
                # force_data[2] = 0 
                # force_data[3] = 0
                # force_data[4] = 0              
                # Robotic-arm experimental-substructure execution step.
                Force_moment = transform_force_tool_to_user(force_data, pose_data)
                # Robotic-arm-based offline RTHS workflow step.
                reaction_force[i] = Force_moment[0]     # Fx (user frame)
                reaction_moment[i] = Force_moment[5]   # Mz (user frame)

                # Robotic-arm-based offline RTHS workflow step.
                exp_x_meas[i] = (pose_data[0] - ref_x) * 1e-3  # mm -> m
                exp_rz_meas[i] = np.radians(pose_data[5] - ref_rz)  # Robotic-arm experimental-substructure execution step.

            else:
                raise RuntimeError(
                    f"Failed to get pose data at t={i*dt:.3f}s, error_id={error_id_pose}"
                )

        else:
            raise RuntimeError(
                f"Failed to get force data at t={i*dt:.3f}s, error_id={error_id_force}"
            )
    
    print(f"Experimental substructure completed: {nsteps} time steps acquired with time synchronization")

    
    # Robotic-arm experimental-substructure execution step.
    print("\nStep 3: Visualizing experimental measurements...")
    
    # Two-panel visualization of experimental response histories.
    fig1, axes1 = plt.subplots(2, 1, figsize=(12, 8))
    fig1.suptitle('Experimental Reaction Force & Moment', fontsize=14, fontweight='bold')
    
    # Robotic-arm experimental-substructure execution step.
    ax1_1 = axes1[0]
    ax1_1.plot(t_array, reaction_force, 'r-', linewidth=2, label='Reaction Force (Fx)')
    ax1_1.set_ylabel('Force (N)', fontsize=10)
    ax1_1.grid(True, alpha=0.3)
    ax1_1.legend(loc='upper right', fontsize=9)
    
    # Robotic-arm experimental-substructure execution step.
    ax1_2 = axes1[1]
    ax1_2.plot(t_array, reaction_moment, 'g-', linewidth=2, label='Reaction Moment (Mz)')
    ax1_2.set_ylabel('Moment (N·m)', fontsize=10)
    ax1_2.set_xlabel('Time (s)', fontsize=10)
    ax1_2.grid(True, alpha=0.3)
    ax1_2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Two-panel visualization of experimental response histories.
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))
    fig2.suptitle('Experimental Displacement & Rotation', fontsize=14, fontweight='bold')
    
    # Robotic-arm experimental-substructure execution step.
    ax2_1 = axes2[0]
    ax2_1.plot(t_array, exp_x_meas, 'b-', linewidth=2, label='Measured X Displacement')
    ax2_1.plot(t_array, x_drive, 'k--', linewidth=2, label='Commanded X Displacement')
    ax2_1.set_ylabel('Displacement (m)', fontsize=10)
    ax2_1.grid(True, alpha=0.3)
    ax2_1.legend(loc='upper right', fontsize=9)
    
    # Robotic-arm experimental-substructure execution step.
    ax2_2 = axes2[1]
    ax2_2.plot(t_array, exp_rz_meas, 'orange', linewidth=2, label='Measured Rz Rotation (rad)')
    ax2_2.plot(t_array, rz_drive, 'purple', linewidth=2, linestyle='--', label='Commanded Rz Rotation (rad)')
    ax2_2.set_ylabel('Rotation (rad)', fontsize=10)
    ax2_2.set_xlabel('Time (s)', fontsize=10)
    ax2_2.grid(True, alpha=0.3)
    ax2_2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Robotic-arm-based offline RTHS workflow step.
    print("\n" + "="*70)
    print('Robotic-arm workflow status message.')
    print('Robotic-arm workflow status message.')
    print("="*70)
    user_confirm = input('Robotic-arm workflow status message.').strip().lower()
    
    if user_confirm not in ['yes', 'y']:
        print('Robotic-arm workflow status message.')
        raise RuntimeError("User cancelled iteration after reviewing experimental measurements.")
    
    print('Robotic-arm workflow status message.')

    return reaction_force, reaction_moment, exp_x_meas, exp_rz_meas


def solve_experimental_substructure_v2(
    dobot,
    x_drive,
    rz_drive,
    dt,
    total_time,
    ref_x, ref_y, ref_z,
    ref_rx, ref_ry, ref_rz,
    save_data=False,
    output_dir='experimental_substructure_results',
    csv_filename='experimental_force_history.csv',
    command_delay_steps=3,
):
    """Execute the experimental substructure and save force-transformation data.
    
        This variant preserves the original experimental-substructure workflow and additionally stores raw force data and transformed force data in CSV format when requested.
    
    """

    nsteps = len(x_drive)
    if command_delay_steps < 0:
        raise ValueError("command_delay_steps must be non-negative")
    if command_delay_steps >= nsteps:
        raise ValueError("command_delay_steps must be smaller than nsteps")

    x_drive_commanded = np.empty_like(x_drive)
    rz_drive_commanded = np.empty_like(rz_drive)
    if command_delay_steps == 0:
        x_drive_commanded[:] = x_drive
        rz_drive_commanded[:] = rz_drive
    else:
        x_drive_commanded[:nsteps-command_delay_steps] = x_drive[command_delay_steps:]
        x_drive_commanded[nsteps-command_delay_steps:] = x_drive[-1]
        rz_drive_commanded[:nsteps-command_delay_steps] = rz_drive[command_delay_steps:]
        rz_drive_commanded[nsteps-command_delay_steps:] = rz_drive[-1]
        print(f"Applying command delay of {command_delay_steps} steps ({command_delay_steps*dt:.3f}s); "
              f"last {command_delay_steps} steps repeat final command.")

    reaction_force = np.zeros(nsteps)
    reaction_moment = np.zeros(nsteps)
    exp_x_meas = np.zeros(nsteps)
    exp_rz_meas = np.zeros(nsteps)
    raw_force_history = np.zeros((nsteps, 6), dtype=float)
    transformed_force_history = np.zeros((nsteps, 6), dtype=float)

    joints_array = []
    pose_targets = []
    last_joints = None

    print("Step 1: Computing inverse kinematics for all time steps...")

    for i in range(nsteps):
        X = ref_x + x_drive_commanded[i] * 1000.0
        Y = ref_y
        Z = ref_z
        Rx = ref_rx
        Ry = ref_ry
        Rz = ref_rz + np.degrees(rz_drive_commanded[i])
        pose_targets.append([X, Y, Z, Rx, Ry, Rz])

        inv_kin_raw = dobot.dashboard.InverseKin(
            X=X, Y=Y, Z=Z, Rx=Rx, Ry=Ry, Rz=Rz,
            user=0, tool=0,
            useJointNear=0,
        )
        print(inv_kin_raw)
        error_id, joints = dobot.parse_robot_response(inv_kin_raw)

        if error_id == 0 and len(joints) == 6:
            joints_array.append(joints)
            last_joints = joints
        else:
            raise RuntimeError(f"Inverse kinematics failed at t={i*dt:.3f}s, error_id={error_id}")

    print("\nStep 2: Previewing inverse kinematics solution...")

    joints_array_vis = np.array(joints_array)
    pose_targets_vis = np.array(pose_targets)
    t_array = np.linspace(0, total_time, nsteps)

    fig0, axes0 = plt.subplots(6, 1, figsize=(12, 10))
    fig0.suptitle('Inverse Kinematics Target Pose Trajectory', fontsize=14, fontweight='bold')
    pose_names = ['X (mm)', 'Y (mm)', 'Z (mm)', 'Rx (°)', 'Ry (°)', 'Rz (°)']

    for k in range(6):
        ax = axes0[k]
        ax.plot(t_array, pose_targets_vis[:, k], 'C0-', linewidth=2, label=pose_names[k])
        ax.set_ylabel(pose_names[k], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        if k < 5:
            ax.set_xticklabels([])
    axes0[-1].set_xlabel('Time (s)', fontsize=10)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(6, 1, figsize=(12, 10))
    fig.suptitle('Robotic-arm workflow status message.', fontsize=14, fontweight='bold')
    joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']

    for j in range(6):
        ax = axes[j]
        ax.plot(t_array, joints_array_vis[:, j], 'b-', linewidth=2, label=f'{joint_names[j]} trajectory')
        ax.set_ylabel(f'{joint_names[j]} (°)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        if j < 5:
            ax.set_xticklabels([])

    axes[-1].set_xlabel('Time (s)', fontsize=10)
    plt.tight_layout()
    plt.show()

    print("\n" + "="*70)
    print('Robotic-arm workflow status message.')
    print("="*70)
    user_input = input('Robotic-arm workflow status message.').strip().lower()

    if user_input not in ['yes', 'y']:
        print('Robotic-arm workflow status message.')
        raise RuntimeError("User cancelled the robot motion after reviewing joint trajectories.")

    print('Robotic-arm workflow status message.')

    # Robotic-arm-based offline RTHS workflow step.
    init_joints = joints_array[0]
    print(f"Moving robot to initial reference pose (first step) joints: {init_joints}")
    result_init = dobot.dashboard.ServoJ(*init_joints,t=4)
    parts_init = result_init.split(',')
    error_id_init = int(parts_init[0]) if parts_init else -1
    if error_id_init != 0:
        raise RuntimeError(f"Failed to move to initial reference pose, error_id={error_id_init}")
    sleep(6)

    loop_start_time = time()
    error_id_force_origin, force_origin = dobot.parse_robot_response(dobot.dashboard.GetForce())
    if error_id_force_origin == 0 and len(force_origin) == 6:
        print(f"Initial force data (tool frame): {force_origin}")
    else:
        raise RuntimeError(f"Failed to get initial force data, error_id={error_id_force_origin}")

    for i in range(nsteps):
        current_step_time = i * dt
        elapsed_time = time() - loop_start_time
        wait_time = current_step_time - elapsed_time
        if wait_time > 0:
            sleep(wait_time)

        target_joints = joints_array[i]
        result = dobot.dashboard.ServoJ(*target_joints)
        print(result)
        parts = result.split(',')
        error_id = int(parts[0]) if parts else -1
        if error_id != 0:
            raise RuntimeError(f"Failed to send ServoJ command at t={i*dt:.3f}s, error_id={error_id}")

        force_raw = dobot.dashboard.GetForce()
        error_id_force, force_data = dobot.parse_robot_response(force_raw)
        if error_id_force == 0 and len(force_data) == 6:
            pose_raw = dobot.dashboard.GetPose(user=0, tool=0)
            error_id_pose, pose_data = dobot.parse_robot_response(pose_raw)
            if error_id_pose == 0 and len(pose_data) == 6:
                force_data[0] = force_data[0] - force_origin[0]
                force_data[1] = force_data[1] - force_origin[1]
                force_data[2] = force_data[2] - force_origin[2]
                force_data[3] = force_data[3] - force_origin[3]
                force_data[4] = force_data[4] - force_origin[4]
                force_data[5] = force_data[5] - force_origin[5]
                raw_force_history[i, :] = force_data
                Force_moment = transform_force_tool_to_user(force_data, pose_data)
                transformed_force_history[i, :] = Force_moment
                reaction_force[i] = Force_moment[0]
                reaction_moment[i] = Force_moment[5]
                exp_x_meas[i] = (pose_data[0] - ref_x) * 1e-3
                exp_rz_meas[i] = np.radians(pose_data[5] - ref_rz)
            else:
                raise RuntimeError(
                    f"Failed to get pose data at t={i*dt:.3f}s, error_id={error_id_pose}"
                )
        else:
            raise RuntimeError(
                f"Failed to get force data at t={i*dt:.3f}s, error_id={error_id_force}"
            )

    print(f"Experimental substructure completed: {nsteps} time steps acquired with time synchronization")

    print("\nStep 3: Visualizing experimental measurements...")

    fig1, axes1 = plt.subplots(2, 1, figsize=(12, 8))
    fig1.suptitle('Experimental Reaction Force & Moment', fontsize=14, fontweight='bold')
    ax1_1 = axes1[0]
    ax1_1.plot(t_array, reaction_force, 'r-', linewidth=2, label='Reaction Force (Fx)')
    ax1_1.set_ylabel('Force (N)', fontsize=10)
    ax1_1.grid(True, alpha=0.3)
    ax1_1.legend(loc='upper right', fontsize=9)
    ax1_2 = axes1[1]
    ax1_2.plot(t_array, reaction_moment, 'g-', linewidth=2, label='Reaction Moment (Mz)')
    ax1_2.set_ylabel('Moment (N·m)', fontsize=10)
    ax1_2.set_xlabel('Time (s)', fontsize=10)
    ax1_2.grid(True, alpha=0.3)
    ax1_2.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.show()

    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))
    fig2.suptitle('Experimental Displacement & Rotation', fontsize=14, fontweight='bold')
    ax2_1 = axes2[0]
    ax2_1.plot(t_array, exp_x_meas, 'b-', linewidth=2, label='Measured X Displacement')
    ax2_1.plot(t_array, x_drive, 'k--', linewidth=2, label='Commanded X Displacement')
    ax2_1.set_ylabel('Displacement (m)', fontsize=10)
    ax2_1.grid(True, alpha=0.3)
    ax2_1.legend(loc='upper right', fontsize=9)
    ax2_2 = axes2[1]
    ax2_2.plot(t_array, exp_rz_meas, 'orange', linewidth=2, label='Measured Rz Rotation (rad)')
    ax2_2.plot(t_array, rz_drive, 'purple', linewidth=2, linestyle='--', label='Commanded Rz Rotation (rad)')
    ax2_2.set_ylabel('Rotation (rad)', fontsize=10)
    ax2_2.set_xlabel('Time (s)', fontsize=10)
    ax2_2.grid(True, alpha=0.3)
    ax2_2.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.show()

    print("\n" + "="*70)
    print('Robotic-arm workflow status message.')
    print('Robotic-arm workflow status message.')
    print("="*70)
    user_confirm = input('Robotic-arm workflow status message.').strip().lower()

    if user_confirm not in ['yes', 'y']:
        print('Robotic-arm workflow status message.')
        raise RuntimeError("User cancelled iteration after reviewing experimental measurements.")

    print('Robotic-arm workflow status message.')

    if save_data:
        csv_path = _save_experimental_force_history(output_dir, csv_filename, t_array, raw_force_history, transformed_force_history)
        print(f"Saved experimental raw/transformed force history to: {csv_path}")

    return reaction_force, -reaction_moment, exp_x_meas, exp_rz_meas

def _lowpass_filter_history(signal, dt, cutoff_hz=5.0, order=4):
    """
    Low-pass filter one measured force/moment history after a robot run.

    The hybrid loop solves one complete time history per iteration, so a
    zero-phase filter is appropriate here: it removes high-frequency force
    sensor noise without adding phase lag to the numerical substructure load.
    """
    data = np.asarray(signal, dtype=float).ravel()
    if cutoff_hz is None or cutoff_hz <= 0.0 or data.size < 4:
        return data.copy()

    fs = 1.0 / dt
    nyquist = 0.5 * fs
    if cutoff_hz >= nyquist:
        return data.copy()

    sos = butter(order, cutoff_hz / nyquist, btype="low", output="sos")
    return sosfiltfilt(sos, data)


def _save_hybrid_iteration_results(
    iteration_dir,
    iteration_index,
    t,
    ext_force_time,
    ext_moment_time,
    Fb_right,
    Mb_right,
    Fb_left,
    Mb_left,
    raw_force_right,
    raw_moment_right,
    raw_force_left,
    raw_moment_left,
    x_drive_right,
    rz_drive_right,
    x_drive_left,
    rz_drive_left,
    x_exp_meas_right,
    rz_exp_meas_right,
    x_exp_meas_left,
    rz_exp_meas_left,
    U_body,
    delta_x_history,
    delta_rz_history,
    right_interface_rmse_history,
    left_interface_rmse_history,
    max_interface_rmse_history,
    right_interface_relaxation_history,
    left_interface_relaxation_history,
):
    """
    Save one experiment iteration in the same style as the numerical preview.

    The experiment cannot provide full solar-array deformation fields, so the
    per-iteration output keeps the quantities actually measured or computed:
    interface force/moment, external excitation, numerical body-side response,
    measured/commanded experimental interface motion, and residual histories.
    """
    os.makedirs(iteration_dir, exist_ok=True)
    nsteps = len(t)

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
        os.path.join(iteration_dir, "convergence_history.csv"),
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
    iter_axis = np.arange(1, len(max_interface_rmse_history) + 1)
    axes[0].semilogy(iter_axis, right_interface_rmse_history, "b-o", label="right interface RMSE")
    axes[0].semilogy(iter_axis, left_interface_rmse_history, "c--s", label="left interface RMSE")
    axes[0].semilogy(iter_axis, max_interface_rmse_history, "k:", label="max interface RMSE")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Aitken Convergence")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(np.arange(1, len(right_interface_relaxation_history) + 1),
                 right_interface_relaxation_history, "b-o", label="right interface omega")
    axes[1].plot(np.arange(1, len(left_interface_relaxation_history) + 1),
                 left_interface_relaxation_history, "c--s", label="left interface omega")
    axes[1].set_xlabel("Aitken update")
    axes[1].set_ylabel("Relaxation factor")
    axes[1].set_title("Adaptive Relaxation Factor")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(iteration_dir, "01_convergence_history.svg"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(t, Fb_right, "b-", label="right filtered")
    axes[0].plot(t, Fb_left, "c--", label="left filtered")
    axes[0].plot(t, raw_force_right, "b:", alpha=0.6, label="right raw")
    axes[0].plot(t, raw_force_left, "c:", alpha=0.6, label="left raw")
    axes[0].set_ylabel("Force (N)")
    axes[0].set_title("Solar-Array Interface Force")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(t, Mb_right, "r-", label="right filtered")
    axes[1].plot(t, Mb_left, "m--", label="left filtered")
    axes[1].plot(t, raw_moment_right, "r:", alpha=0.6, label="right raw")
    axes[1].plot(t, raw_moment_left, "m:", alpha=0.6, label="left raw")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Moment (N*m)")
    axes[1].set_title("Solar-Array Interface Moment")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(iteration_dir, "02_interface_force_moment.svg"), bbox_inches="tight")
    plt.close(fig)

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
    fig.savefig(os.path.join(iteration_dir, "02b_external_excitation.svg"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(t, U_body[0, :], "b-", label="numerical right body-side x")
    axes[0].plot(t, x_exp_meas_right, "b--", label="experimental right measured x")
    axes[0].plot(t, x_drive_right, "b:", label="right command x")
    axes[0].plot(t, U_body[2, :], "c-", label="numerical left body-side x")
    axes[0].plot(t, x_exp_meas_left, "c--", label="experimental left measured x")
    axes[0].plot(t, x_drive_left, "c:", label="left command x")
    axes[0].plot(t, U_body[4, :], "k:", label="body COM x")
    axes[0].set_ylabel("Displacement (m)")
    axes[0].set_title("Interface Displacement")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].plot(t, U_body[1, :], "b-", label="numerical right body-side rz")
    axes[1].plot(t, rz_exp_meas_right, "b--", label="experimental right measured rz")
    axes[1].plot(t, rz_drive_right, "b:", label="right command rz")
    axes[1].plot(t, U_body[3, :], "c-", label="numerical left body-side rz")
    axes[1].plot(t, rz_exp_meas_left, "c--", label="experimental left measured rz")
    axes[1].plot(t, rz_drive_left, "c:", label="left command rz")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Rotation (rad)")
    axes[1].set_title("Interface Rotation")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(iteration_dir, "03_interface_response.svg"), bbox_inches="tight")
    plt.close(fig)

    time_series_data = np.column_stack((
        t, Fb_right, Mb_right, Fb_left, Mb_left,
        raw_force_right, raw_moment_right, raw_force_left, raw_moment_left,
        ext_force_time, ext_moment_time,
        U_body[0, :], x_exp_meas_right, x_drive_right,
        U_body[1, :], rz_exp_meas_right, rz_drive_right,
        U_body[2, :], x_exp_meas_left, x_drive_left,
        U_body[3, :], rz_exp_meas_left, rz_drive_left,
        U_body[4, :], U_body[5, :],
    ))
    np.savetxt(
        os.path.join(iteration_dir, "data_time_series.csv"),
        time_series_data,
        delimiter=",",
        header=(
            "Time(s),Fb_Right_Filtered(N),Mb_Right_Filtered(Nm),"
            "Fb_Left_Filtered(N),Mb_Left_Filtered(Nm),"
            "Fb_Right_Raw(N),Mb_Right_Raw(Nm),Fb_Left_Raw(N),Mb_Left_Raw(Nm),"
            "Ext_Force_Body(N),Ext_Moment_Body(Nm),"
            "X_Num_Right_Interface(m),X_Exp_Right_Interface(m),X_Cmd_Right_Interface(m),"
            "Rz_Num_Right_Interface(rad),Rz_Exp_Right_Interface(rad),Rz_Cmd_Right_Interface(rad),"
            "X_Num_Left_Interface(m),X_Exp_Left_Interface(m),X_Cmd_Left_Interface(m),"
            "Rz_Num_Left_Interface(rad),Rz_Exp_Left_Interface(rad),Rz_Cmd_Left_Interface(rad),"
            "X_Body_COM(m),Rz_Body(rad)"
        ),
        comments="",
    )

    if delta_x_history and delta_rz_history:
        csv_cols = [t]
        header_cols = ["Time(s)"]
        for i, delta_x in enumerate(delta_x_history, start=1):
            csv_cols.extend([delta_x[:nsteps], delta_x[nsteps:]])
            header_cols.extend([
                f"Iter{i}_Right_X_Displacement_Delta(m)",
                f"Iter{i}_Left_X_Displacement_Delta(m)",
            ])
        for i, delta_rz in enumerate(delta_rz_history, start=1):
            csv_cols.extend([delta_rz[:nsteps], delta_rz[nsteps:]])
            header_cols.extend([
                f"Iter{i}_Right_Rotation_Delta(rad)",
                f"Iter{i}_Left_Rotation_Delta(rad)",
            ])
        np.savetxt(
            os.path.join(iteration_dir, "data_iteration_delta.csv"),
            np.column_stack(csv_cols),
            delimiter=",",
            header=",".join(header_cols),
            comments="",
        )


def aitken_coupling_satellite(
    body_params,
    dt,
    total_time,
    tol,
    max_iter,
    relaxation_ini,
    ext_force_time,
    ext_moment_time,
    ext_pos,
    adaptive_relaxation,
    relaxation_min,
    relaxation_max,
    dobot,
    exp_ref,
    force_filter_cutoff_hz=5.0,
    iteration_output_dir=None,
):
    """
    Hybrid Aitken coupling for satellite body + experimental solar array.

    Numerical variables use y/rz. The robot experiment is driven in x/rz, so
    the interface displacement residual is reported as delta_x_history.
    """
    nsteps = int(np.floor(total_time / dt)) + 1
    t = np.linspace(0.0, total_time, nsteps)
    if iteration_output_dir is None:
        iteration_output_dir = os.path.join(os.getcwd(), "satellite_hybrid_results", "iteration_results")

    if dobot is None:
        raise ValueError("dobot instance must be provided for hybrid coupling")
    required_ref = ["x", "y", "z", "rx", "ry", "rz"]
    missing_ref = [key for key in required_ref if key not in exp_ref]
    if missing_ref:
        raise ValueError("exp_ref missing required keys: " + ", ".join(missing_ref))

    if ext_force_time is None:
        ext_force_time = np.zeros(nsteps)
    else:
        ext_force_time = np.asarray(ext_force_time, dtype=float).ravel()
        if ext_force_time.size != nsteps:
            raise ValueError("ext_force_time must have length nsteps")

    if ext_moment_time is None:
        ext_moment_time = np.zeros(nsteps)
    else:
        ext_moment_time = np.asarray(ext_moment_time, dtype=float).ravel()
        if ext_moment_time.size != nsteps:
            raise ValueError("ext_moment_time must have length nsteps")

    required = ["body_mass", "body_inertia_z", "interface_offset"]
    missing = [key for key in required if key not in body_params]
    if missing:
        raise ValueError("body_params missing required keys: " + ", ".join(missing))

    # L_char only scales rz into a displacement-equivalent residual for Aitken
    # RMSE. Set it in main_satellite.py to the full deployed span when the solar
    # array length is known: left wing + body root-to-root span + right wing.
    characteristic_length = body_params["characteristic_length"]
    solve_body_params = {
        key: value for key, value in body_params.items()
        if key not in ("characteristic_length", "interface_load_scale")
    }

    right_interface_relaxation_history = [relaxation_ini]
    left_interface_relaxation_history = [relaxation_ini]
    delta_x_history = []
    delta_rz_history = []
    max_interface_rmse_history = []
    right_interface_rmse_history = []
    left_interface_rmse_history = []
    x_exp_meas_final = np.zeros(nsteps)
    rz_exp_meas_final = np.zeros(nsteps)
    Fb_right = np.zeros(nsteps)
    Mb_right = np.zeros(nsteps)
    Fb_left = np.zeros(nsteps)
    Mb_left = np.zeros(nsteps)

    right_x_prev, right_rz_prev, left_x_prev, left_rz_prev, U_prev = solve_satellite_body_two_sided_substructure(
        F_boundary_right=np.zeros(nsteps),
        M_boundary_right=np.zeros(nsteps),
        F_boundary_left=np.zeros(nsteps),
        M_boundary_left=np.zeros(nsteps),
        ext_force=ext_force_time,
        ext_pos=ext_pos,
        ext_moment=ext_moment_time,
        dt=dt,
        total_time=total_time,
        **solve_body_params,
    )


    R_prev_right = None
    R_prev_left = None
    omega_right = relaxation_ini
    omega_left = relaxation_ini
    converged = False

    print("Hybrid coupling mode: Satellite body - right and left solar-array experiments")

    for it in range(1, max_iter + 1):
        if it == 1:
            x_drive_right = right_x_prev.copy()
            rz_drive_right = right_rz_prev.copy()
            x_drive_left = left_x_prev.copy()
            rz_drive_left = left_rz_prev.copy()
        else:
            x_drive_right = omega_right * right_x_prev + (1.0 - omega_right) * x_drive_right_prev
            rz_drive_right = omega_right * right_rz_prev + (1.0 - omega_right) * rz_drive_right_prev
            x_drive_left = omega_left * left_x_prev + (1.0 - omega_left) * x_drive_left_prev
            rz_drive_left = omega_left * left_rz_prev + (1.0 - omega_left) * rz_drive_left_prev

        # Run the right-side solar-array test first. exp_ref is the robot's
        # fixed initial pose; it is not tied to whether this pass represents the
        # left or right spacecraft wing.
        reaction_force_right, reaction_moment_right, x_exp_meas_right, rz_exp_meas_right = solve_experimental_substructure(
            dobot=dobot,
            x_drive=x_drive_right,
            rz_drive=rz_drive_right,
            dt=dt,
            total_time=total_time,
            ref_x=exp_ref["x"],
            ref_y=exp_ref["y"],
            ref_z=exp_ref["z"],
            ref_rx=exp_ref["rx"],
            ref_ry=exp_ref["ry"],
            ref_rz=exp_ref["rz"],
        )

        # Then run the left-side solar-array test from the same robot reference
        # pose. If the fixture reverses x or rz, correct that with calibration
        # or a side-specific sign convention before force assembly.
        reaction_force_left, reaction_moment_left, x_exp_meas_left, rz_exp_meas_left = solve_experimental_substructure(
            dobot=dobot,
            x_drive=x_drive_left,
            rz_drive=rz_drive_left,
            dt=dt,
            total_time=total_time,
            ref_x=exp_ref["x"],
            ref_y=exp_ref["y"],
            ref_z=exp_ref["z"],
            ref_rx=exp_ref["rx"],
            ref_ry=exp_ref["ry"],
            ref_rz=exp_ref["rz"],
        )

        # CSV input/output step for post-processing.
        try:
            base_iter_dir = iteration_output_dir
            os.makedirs(base_iter_dir, exist_ok=True)
            iter_dir = os.path.join(base_iter_dir, f"iter_{it:02d}")
            os.makedirs(iter_dir, exist_ok=True)

            # CSV input/output step for post-processing.
            right_csv = os.path.join(iter_dir, f"right_iter_{it:02d}_exp.csv")
            header_r = (
                "Time(s),ReactionForce(N),ReactionMoment(Nm),Exp_X(m),Exp_Rz(rad),Cmd_X(m),Cmd_Rz(rad)"
            )
            np.savetxt(
                right_csv,
                np.column_stack((t, reaction_force_right, reaction_moment_right,
                                 x_exp_meas_right, rz_exp_meas_right,
                                 x_drive_right, rz_drive_right)),
                delimiter=",", header=header_r, comments="",
            )

            # Quick-look figure comparing experimental and commanded responses.
            f_r, ax_r = plt.subplots(2, 1, figsize=(10, 6))
            f_r.suptitle(f"Iteration {it:02d} Right Experimental Results")
            ax_r[0].plot(t, reaction_force_right, 'r-', linewidth=1.5, label='Reaction Force (Fx)')
            ax_r[0].set_ylabel('Force (N)')
            ax_r[0].grid(True, alpha=0.3)
            ax_r[1].plot(t, x_exp_meas_right, 'b-', linewidth=1.5, label='Measured X')
            ax_r[1].plot(t, x_drive_right, 'k--', linewidth=1.2, label='Commanded X')
            ax_r[1].set_ylabel('Displacement (m)')
            ax_r[1].set_xlabel('Time (s)')
            ax_r[1].grid(True, alpha=0.3)
            ax_r[1].legend()
            f_r.tight_layout()
            f_r.savefig(os.path.join(iter_dir, f"right_iter_{it:02d}_exp_summary.svg"), bbox_inches='tight')
            plt.close(f_r)

            # CSV input/output step for post-processing.
            left_csv = os.path.join(iter_dir, f"left_iter_{it:02d}_exp.csv")
            header_l = (
                "Time(s),ReactionForce(N),ReactionMoment(Nm),Exp_X(m),Exp_Rz(rad),Cmd_X(m),Cmd_Rz(rad)"
            )
            np.savetxt(
                left_csv,
                np.column_stack((t, reaction_force_left, reaction_moment_left,
                                 x_exp_meas_left, rz_exp_meas_left,
                                 x_drive_left, rz_drive_left)),
                delimiter=",", header=header_l, comments="",
            )

            # Quick-look figure comparing experimental and commanded responses.
            f_l, ax_l = plt.subplots(2, 1, figsize=(10, 6))
            f_l.suptitle(f"Iteration {it:02d} Left Experimental Results")
            ax_l[0].plot(t, reaction_force_left, 'r-', linewidth=1.5, label='Reaction Force (Fx)')
            ax_l[0].set_ylabel('Force (N)')
            ax_l[0].grid(True, alpha=0.3)
            ax_l[1].plot(t, x_exp_meas_left, 'b-', linewidth=1.5, label='Measured X')
            ax_l[1].plot(t, x_drive_left, 'k--', linewidth=1.2, label='Commanded X')
            ax_l[1].set_ylabel('Displacement (m)')
            ax_l[1].set_xlabel('Time (s)')
            ax_l[1].grid(True, alpha=0.3)
            ax_l[1].legend()
            f_l.tight_layout()
            f_l.savefig(os.path.join(iter_dir, f"left_iter_{it:02d}_exp_summary.svg"), bbox_inches='tight')
            plt.close(f_l)

            print(f"Saved iteration {it:02d} experimental results to: {iter_dir}")
        except Exception as _save_exc:
            print(f"Warning: failed to save iteration {it:02d} experimental results: {_save_exc}")

        x_exp_meas_final = np.vstack([x_exp_meas_right, x_exp_meas_left])
        rz_exp_meas_final = np.vstack([rz_exp_meas_right, rz_exp_meas_left])

        Fb_right_new = _lowpass_filter_history(
            reaction_force_right, dt, cutoff_hz=force_filter_cutoff_hz
        )
        Mb_right_new = _lowpass_filter_history(
            reaction_moment_right, dt, cutoff_hz=force_filter_cutoff_hz
        )
        Fb_left_new = _lowpass_filter_history(
            reaction_force_left, dt, cutoff_hz=force_filter_cutoff_hz
        )
        Mb_left_new = _lowpass_filter_history(
            reaction_moment_left, dt, cutoff_hz=force_filter_cutoff_hz
        )

        right_x_curr, right_rz_curr, left_x_curr, left_rz_curr, U_curr = solve_satellite_body_two_sided_substructure(
            F_boundary_right=Fb_right_new,
            M_boundary_right=Mb_right_new,
            F_boundary_left=Fb_left_new,
            M_boundary_left=Mb_left_new,
            ext_force=ext_force_time,
            ext_pos=ext_pos,
            ext_moment=ext_moment_time,
            dt=dt,
            total_time=total_time,
            **solve_body_params,
        )

        delta_x_right = right_x_curr - x_drive_right
        delta_rz_right = right_rz_curr - rz_drive_right
        delta_x_left = left_x_curr - x_drive_left
        delta_rz_left = left_rz_curr - rz_drive_left
        delta_x = np.hstack([delta_x_right, delta_x_left])
        delta_rz = np.hstack([delta_rz_right, delta_rz_left])
        delta_x_history.append(delta_x.copy())
        delta_rz_history.append(delta_rz.copy())

        rmse_right = np.sqrt(np.mean(delta_x_right**2 + (delta_rz_right * characteristic_length)**2))
        rmse_left = np.sqrt(np.mean(delta_x_left**2 + (delta_rz_left * characteristic_length)**2))
        rmse = max(rmse_right, rmse_left)
        max_interface_rmse_history.append(rmse)
        right_interface_rmse_history.append(rmse_right)
        left_interface_rmse_history.append(rmse_left)
        print(f"Iteration {it:02d}: RMSE right = {rmse_right:.3e}, left = {rmse_left:.3e}, max = {rmse:.3e}")

        try:
            iter_dir = os.path.join(iteration_output_dir, f"iter_{it:02d}")
            _save_hybrid_iteration_results(
                iteration_dir=iter_dir,
                iteration_index=it,
                t=t,
                ext_force_time=ext_force_time,
                ext_moment_time=ext_moment_time,
                Fb_right=Fb_right_new,
                Mb_right=Mb_right_new,
                Fb_left=Fb_left_new,
                Mb_left=Mb_left_new,
                raw_force_right=reaction_force_right,
                raw_moment_right=reaction_moment_right,
                raw_force_left=reaction_force_left,
                raw_moment_left=reaction_moment_left,
                x_drive_right=x_drive_right,
                rz_drive_right=rz_drive_right,
                x_drive_left=x_drive_left,
                rz_drive_left=rz_drive_left,
                x_exp_meas_right=x_exp_meas_right,
                rz_exp_meas_right=rz_exp_meas_right,
                x_exp_meas_left=x_exp_meas_left,
                rz_exp_meas_left=rz_exp_meas_left,
                U_body=U_curr,
                delta_x_history=delta_x_history,
                delta_rz_history=delta_rz_history,
                right_interface_rmse_history=right_interface_rmse_history,
                left_interface_rmse_history=left_interface_rmse_history,
                max_interface_rmse_history=max_interface_rmse_history,
                right_interface_relaxation_history=right_interface_relaxation_history,
                left_interface_relaxation_history=left_interface_relaxation_history,
            )
            print(f"Saved iteration {it:02d} aligned hybrid results to: {iter_dir}")
        except Exception as _save_exc:
            print(f"Warning: failed to save aligned iteration {it:02d} results: {_save_exc}")

        # Save complete iteration results, including satellite-body predictions and RMSE.
        try:
            base_iter_dir = iteration_output_dir
            iter_dir = os.path.join(base_iter_dir, f"iter_{it:02d}")
            os.makedirs(iter_dir, exist_ok=True)

            # CSV input/output step for post-processing.
            right_csv_full = os.path.join(iter_dir, f"right_iter_{it:02d}_complete.csv")
            header_r_full = (
                "Time(s),ReactionForce(N),ReactionMoment(Nm),Exp_X(m),Exp_Rz(rad),Cmd_X(m),Cmd_Rz(rad),"
                "Body_X(m),Body_Rz(rad),Delta_X(m),Delta_Rz(rad),RMSE_Right"
            )
            np.savetxt(
                right_csv_full,
                np.column_stack((t, reaction_force_right, reaction_moment_right,
                                 x_exp_meas_right, rz_exp_meas_right,
                                 x_drive_right, rz_drive_right,
                                 right_x_curr, right_rz_curr,
                                 delta_x_right, delta_rz_right,
                                 np.full(nsteps, rmse_right))),
                delimiter=",", header=header_r_full, comments="",
            )

            f_r_full, ax_r_full = plt.subplots(2, 1, figsize=(10, 6))
            f_r_full.suptitle(f"Iteration {it:02d} Right Complete Comparison")
            ax_r_full[0].plot(t, x_drive_right, 'k--', linewidth=1.2, label='Commanded X')
            ax_r_full[0].plot(t, x_exp_meas_right, 'b-', linewidth=1.5, label='Measured X')
            ax_r_full[0].plot(t, right_x_curr, 'g-.', linewidth=1.5, label='Body Predicted X')
            ax_r_full[0].set_ylabel('Displacement (m)')
            ax_r_full[0].grid(True, alpha=0.3)
            ax_r_full[0].legend()
            ax_r_full[1].plot(t, rz_drive_right, 'k--', linewidth=1.2, label='Commanded Rz')
            ax_r_full[1].plot(t, rz_exp_meas_right, 'orange', linewidth=1.5, label='Measured Rz')
            ax_r_full[1].plot(t, right_rz_curr, 'g-.', linewidth=1.5, label='Body Predicted Rz')
            ax_r_full[1].set_ylabel('Rotation (rad)')
            ax_r_full[1].set_xlabel('Time (s)')
            ax_r_full[1].grid(True, alpha=0.3)
            ax_r_full[1].legend()
            f_r_full.tight_layout()
            f_r_full.savefig(os.path.join(iter_dir, f"right_iter_{it:02d}_complete.svg"), bbox_inches='tight')
            plt.close(f_r_full)

            # CSV input/output step for post-processing.
            left_csv_full = os.path.join(iter_dir, f"left_iter_{it:02d}_complete.csv")
            header_l_full = (
                "Time(s),ReactionForce(N),ReactionMoment(Nm),Exp_X(m),Exp_Rz(rad),Cmd_X(m),Cmd_Rz(rad),"
                "Body_X(m),Body_Rz(rad),Delta_X(m),Delta_Rz(rad),RMSE_Left"
            )
            np.savetxt(
                left_csv_full,
                np.column_stack((t, reaction_force_left, reaction_moment_left,
                                 x_exp_meas_left, rz_exp_meas_left,
                                 x_drive_left, rz_drive_left,
                                 left_x_curr, left_rz_curr,
                                 delta_x_left, delta_rz_left,
                                 np.full(nsteps, rmse_left))),
                delimiter=",", header=header_l_full, comments="",
            )

            f_l_full, ax_l_full = plt.subplots(2, 1, figsize=(10, 6))
            f_l_full.suptitle(f"Iteration {it:02d} Left Complete Comparison")
            ax_l_full[0].plot(t, x_drive_left, 'k--', linewidth=1.2, label='Commanded X')
            ax_l_full[0].plot(t, x_exp_meas_left, 'b-', linewidth=1.5, label='Measured X')
            ax_l_full[0].plot(t, left_x_curr, 'g-.', linewidth=1.5, label='Body Predicted X')
            ax_l_full[0].set_ylabel('Displacement (m)')
            ax_l_full[0].grid(True, alpha=0.3)
            ax_l_full[0].legend()
            ax_l_full[1].plot(t, rz_drive_left, 'k--', linewidth=1.2, label='Commanded Rz')
            ax_l_full[1].plot(t, rz_exp_meas_left, 'orange', linewidth=1.5, label='Measured Rz')
            ax_l_full[1].plot(t, left_rz_curr, 'g-.', linewidth=1.5, label='Body Predicted Rz')
            ax_l_full[1].set_ylabel('Rotation (rad)')
            ax_l_full[1].set_xlabel('Time (s)')
            ax_l_full[1].grid(True, alpha=0.3)
            ax_l_full[1].legend()
            f_l_full.tight_layout()
            f_l_full.savefig(os.path.join(iter_dir, f"left_iter_{it:02d}_complete.svg"), bbox_inches='tight')
            plt.close(f_l_full)

            print(f"Saved iteration {it:02d} complete results with RMSE to: {iter_dir}")
        except Exception as _save_exc:
            print(f"Warning: failed to save iteration {it:02d} complete results: {_save_exc}")

        if adaptive_relaxation:
            X_in_right = np.hstack([
                x_drive_right.ravel(),
                (rz_drive_right * characteristic_length).ravel(),
            ])
            X_out_right = np.hstack([
                right_x_curr.ravel(),
                (right_rz_curr * characteristic_length).ravel(),
            ])
            R_curr_right = X_out_right - X_in_right

            X_in_left = np.hstack([
                x_drive_left.ravel(),
                (rz_drive_left * characteristic_length).ravel(),
            ])
            X_out_left = np.hstack([
                left_x_curr.ravel(),
                (left_rz_curr * characteristic_length).ravel(),
            ])
            R_curr_left = X_out_left - X_in_left

            if it == 1:
                R_prev_right = R_curr_right.copy()
                R_prev_left = R_curr_left.copy()
            else:
                delta_R_right = R_curr_right - R_prev_right
                den = np.dot(delta_R_right, delta_R_right)
                if den > 1e-14:
                    omega_new = -omega_right * np.dot(R_prev_right, delta_R_right) / den
                    omega_old = omega_right
                    omega_right = max(relaxation_min, min(relaxation_max, omega_new))
                    print(f"  [Aitken Right {it}] omega: {omega_old:.4f} -> {omega_right:.4f}")
                right_interface_relaxation_history.append(omega_right)
                R_prev_right = R_curr_right.copy()

                delta_R_left = R_curr_left - R_prev_left
                den = np.dot(delta_R_left, delta_R_left)
                if den > 1e-14:
                    omega_new = -omega_left * np.dot(R_prev_left, delta_R_left) / den
                    omega_old = omega_left
                    omega_left = max(relaxation_min, min(relaxation_max, omega_new))
                    print(f"  [Aitken Left {it}]  omega: {omega_old:.4f} -> {omega_left:.4f}")
                left_interface_relaxation_history.append(omega_left)
                R_prev_left = R_curr_left.copy()

        if rmse_right < tol and rmse_left < tol:
            print(f"\nConverged in {it} iterations (right={rmse_right:.3e}, left={rmse_left:.3e})")
            Fb_right = Fb_right_new
            Mb_right = Mb_right_new
            Fb_left = Fb_left_new
            Mb_left = Mb_left_new
            U = U_curr
            converged = True
            break

        x_drive_right_prev = x_drive_right.copy()
        rz_drive_right_prev = rz_drive_right.copy()
        x_drive_left_prev = x_drive_left.copy()
        rz_drive_left_prev = rz_drive_left.copy()
        right_x_prev = right_x_curr.copy()
        right_rz_prev = right_rz_curr.copy()
        left_x_prev = left_x_curr.copy()
        left_rz_prev = left_rz_curr.copy()
        Fb_right = Fb_right_new
        Mb_right = Mb_right_new
        Fb_left = Fb_left_new
        Mb_left = Mb_left_new
        U = U_curr

    if not converged:
        print(f"\nReached max_iter={max_iter}, final RMSE={max_interface_rmse_history[-1]:.3e}")
        try:
            U
        except NameError:
            U = U_prev

    return (
        Fb_right,
        Mb_right,
        Fb_left,
        Mb_left,
        max_interface_rmse_history,
        right_interface_rmse_history,
        left_interface_rmse_history,
        U,
        t,
        right_interface_relaxation_history,
        left_interface_relaxation_history,
        delta_x_history,
        delta_rz_history,
        x_exp_meas_final,
        rz_exp_meas_final,
    )
