"""
Stage 3 Test: Joint Torque Response

This test loads a locally generated yaw-only airframe, fixes the base, applies
opposing torques about +Z on selected motors, and measures joint response.

Usage:
  # 1) Generate the yaw-only airframe USD + JSON (defaults from YAML)
  python scripts/convert_concentrated_urdf_yaw_only.py --headless \
      --output_dir cache/workspace_local/my_airframe_yaw_only \
      --config configs/joint_response_template.yaml

  #    (Optional) override params on the CLI
  python scripts/convert_concentrated_urdf_yaw_only.py --headless \
      --output_dir cache/workspace_local/my_airframe_yaw_only \
      --arm_proximal_length 0.1 --arm_distal_length 0.1 \
      --base_mass 0.3 --base_inertia 0.0005 0.0005 0.001

  # 2) Run the test (uses default config if --config not provided)
  python erc_isaac/joint_response.py --headless

  # 3) Run with an explicit config override
  python erc_isaac/joint_response.py --headless \
      --config configs/joint_response_template.yaml

  # 4) Optional video recording override (same as other tests)
  python erc_isaac/joint_response.py --headless --video 0.25

   # 5) Optional analytics video generation
    python erc_isaac/joint_response.py --headless --video 0.25 --analytics
"""

import os
import sys

# Add repo root (one level up from tests/) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs/joint_response_template.yaml"

parser = argparse.ArgumentParser(description="Yaw-only joint torque response test")
parser.add_argument(
    "--config",
    type=str,
    default=str(DEFAULT_CONFIG_PATH),
    help="Path to YAML config (relative to repo root or absolute).",
)
parser.add_argument(
    "--video",
    type=float,
    default=None,
    help="Record video at specified playback speed (e.g., 0.25 = 4x slower). Overrides YAML.",
)
parser.add_argument(
    "--analytics",
    action="store_true",
    help="Enable analytics video (plots). Uses YAML path unless overridden by --analytics_path.",
)
parser.add_argument(
    "--analytics_path",
    type=str,
    default=None,
    help="Override analytics video output path.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, PhysxCfg
from isaaclab.utils import configclass

from erc_isaac.common_utils import Recorder
from erc_isaac.joint_response_simulator import build_response_table, interpolate_torque

YAW_JOINT_NAMES = [f"arm_segment_1_{i}_to_dummy_link_yaw_{i}" for i in range(4)]


@configclass
class YawOnlySceneCfg(InteractiveSceneCfg):
    """Scene config for yaw-only joint response testing."""

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        actuators={
            "yaw_joints": ImplicitActuatorCfg(
                joint_names_expr=YAW_JOINT_NAMES,
                stiffness=0.0,
                damping=0.0,
                effort_limit_sim=20000.0,
                velocity_limit_sim=10000.0,
                effort_limit=20000.0,
                velocity_limit=10000.0,
                friction=0.0,
                dynamic_friction=0.0,
                viscous_friction=0.0,
                armature=0.0,
            )
        },
    )


def resolve_path(path_str: str) -> Path:
    """Resolve a path relative to repo root unless absolute."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (Path(__file__).parent.parent / path).resolve()


def load_config(path_str: str) -> dict:
    """Load YAML config file."""
    path = resolve_path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_airframe_paths(cfg: dict):
    """Resolve yaw-only USD and airframe data paths from config."""
    airframe_cfg = cfg.get("airframe", {})
    usd_path = resolve_path(airframe_cfg.get(
        "usd_path",
        "cache/workspace_local/my_airframe_yaw_only/concentrated_quadrotor_yaw_only.usd",
    ))
    airframe_data_path = resolve_path(airframe_cfg.get(
        "airframe_data_path",
        "cache/workspace_local/my_airframe_yaw_only/airframe_data.json",
    ))
    return usd_path, airframe_data_path


def _parse_hard_endstop_deg(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lower_deg, upper_deg = float(value[0]), float(value[1])
        return lower_deg, upper_deg
    deg = float(value)
    return -abs(deg), abs(deg)


def _build_torque_schedule(test_cfg: dict):
    schedule_cfg = test_cfg.get("torque_schedule", None)
    if not schedule_cfg:
        return None
    schedule = []
    t_accum = 0.0
    for entry in schedule_cfg:
        if entry is None:
            continue
        duration = float(entry.get("duration", 0.0))
        scale = float(entry.get("scale", 1.0))
        if duration <= 0.0:
            continue
        t_accum += duration
        schedule.append((t_accum, scale))
    return schedule if schedule else None


def _torque_scale_at_time(t: float, schedule):
    if schedule is None:
        return 1.0
    for t_end, scale in schedule:
        if t <= t_end:
            return scale
    return schedule[-1][1]

def _expand_for_playback_speed(times: np.ndarray, dt: float, playback_speed: float, target_fps: int = 60):
    sim_time_per_video_frame = (1.0 / target_fps) * playback_speed
    current_sim_time = 0.0
    frame_count = 0
    frame_indices = []
    for i in range(len(times)):
        expected_frame_count = int(current_sim_time / sim_time_per_video_frame)
        frames_to_write = expected_frame_count - frame_count
        if frames_to_write > 0:
            frame_indices.extend([i] * frames_to_write)
            frame_count += frames_to_write
        current_sim_time += dt
    if not frame_indices:
        frame_indices = [0]
    return np.array(frame_indices, dtype=np.int64)


def _save_analytics_video(
    times: np.ndarray,
    angles_deg: np.ndarray,
    applied_torque: np.ndarray,
    passive_torque: np.ndarray,
    damping_torque: np.ndarray,
    soft_endstop_torque: np.ndarray,
    total_reaction_torque: np.ndarray,
    response_angles_rad: np.ndarray,
    response_torques: np.ndarray,
    output_path: Path,
    fps: int = 60,
    playback_speed: float | None = None,
    dt: float | None = None,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")

    if playback_speed is not None:
        if dt is None:
            raise ValueError("dt is required when playback_speed is provided.")
        frame_indices = _expand_for_playback_speed(times, dt, playback_speed, target_fps=60)
        times_plot = times[frame_indices]
        angles_deg_plot = angles_deg[frame_indices]
        applied_torque_plot = applied_torque[frame_indices]
        passive_torque_plot = passive_torque[frame_indices]
        damping_torque_plot = damping_torque[frame_indices]
        soft_endstop_torque_plot = soft_endstop_torque[frame_indices]
        total_reaction_torque_plot = total_reaction_torque[frame_indices]
        fps = 60
    else:
        times_plot = times
        angles_deg_plot = angles_deg
        applied_torque_plot = applied_torque
        passive_torque_plot = passive_torque
        damping_torque_plot = damping_torque
        soft_endstop_torque_plot = soft_endstop_torque
        total_reaction_torque_plot = total_reaction_torque

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle("Joint Analytics (Replay)", fontsize=14)

    # Panel 1: torques vs time
    line_applied, = ax1.plot([], [], "k-", linewidth=1.5, label="applied motor_0 torque")
    line_total, = ax1.plot([], [], "r-", linewidth=2.0, label="reaction total")
    line_passive, = ax1.plot([], [], "b--", linewidth=1.5, label="reaction passive")
    line_damping, = ax1.plot([], [], "g--", linewidth=1.0, label="reaction viscous")
    line_soft, = ax1.plot([], [], "m--", linewidth=1.0, label="reaction soft endstop")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Torque (Nm)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    # Panel 2: joint angle vs time
    line_angle, = ax2.plot([], [], "b-", linewidth=2.0, label="joint_0 angle")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angle (deg)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    # Panel 3: torque vs angle response curve + moving dot
    response_angles_deg = response_angles_rad * 180.0 / np.pi
    ax3.plot(response_angles_deg, response_torques, "k-", linewidth=2.0, label="response curve")
    dot, = ax3.plot([], [], "ro", markersize=6, label="current")
    ax3.set_xlabel("Angle (deg)")
    ax3.set_ylabel("Torque (Nm)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", fontsize=8)

    # Limits
    t_min, t_max = float(times[0]), float(times[-1])
    ax1.set_xlim(t_min, t_max)
    ax2.set_xlim(t_min, t_max)

    torque_all = np.concatenate(
        [applied_torque, total_reaction_torque, passive_torque, damping_torque, soft_endstop_torque]
    )
    t_min_val = float(np.min(torque_all))
    t_max_val = float(np.max(torque_all))
    pad = 0.05 * max(1.0, t_max_val - t_min_val)
    ax1.set_ylim(t_min_val - pad, t_max_val + pad)

    ang_min = float(np.min(angles_deg))
    ang_max = float(np.max(angles_deg))
    pad_ang = 0.05 * max(1.0, ang_max - ang_min)
    ax2.set_ylim(ang_min - pad_ang, ang_max + pad_ang)

    resp_min = float(np.min(response_torques))
    resp_max = float(np.max(response_torques))
    pad_resp = 0.05 * max(1.0, resp_max - resp_min)
    ax3.set_ylim(resp_min - pad_resp, resp_max + pad_resp)

    def update(i):
        t_slice = times_plot[: i + 1]
        line_applied.set_data(t_slice, applied_torque_plot[: i + 1])
        line_total.set_data(t_slice, total_reaction_torque_plot[: i + 1])
        line_passive.set_data(t_slice, passive_torque_plot[: i + 1])
        line_damping.set_data(t_slice, damping_torque_plot[: i + 1])
        line_soft.set_data(t_slice, soft_endstop_torque_plot[: i + 1])

        line_angle.set_data(t_slice, angles_deg_plot[: i + 1])

        dot.set_data([angles_deg_plot[i]], [passive_torque_plot[i]])
        return line_applied, line_total, line_passive, line_damping, line_soft, line_angle, dot

    try:
        import cv2

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter (mp4v).")

        total_frames = len(times_plot)
        print(f"[Analytics] Writing {total_frames} frames at {fps} fps to {output_path}")
        try:
            for i in range(total_frames):
                update(i)
                fig.canvas.draw()
                frame_rgba = np.asarray(fig.canvas.buffer_rgba())
                frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
                writer.write(frame_bgr)
                if i % 100 == 0 and i != 0:
                    print(f"[Analytics] ... {i}/{total_frames} frames")
        finally:
            writer.release()
        print(f"[Analytics] Video saved to {output_path}")
    except Exception as exc:
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(times_plot)):
            update(i)
            fig.savefig(frames_dir / f"frame_{i:05d}.png", dpi=150)
        print(f"[Analytics] Video failed ({exc}). Saved frames to {frames_dir}")

    plt.close(fig)


def main():
    cfg = load_config(args_cli.config)
    usd_path, airframe_data_path = load_airframe_paths(cfg)
    output_cfg = cfg.get("output", {})
    results_dir = resolve_path(output_cfg.get("results_dir", "results"))
    results_dir.mkdir(exist_ok=True)

    if not usd_path.exists() or not airframe_data_path.exists():
        raise FileNotFoundError(
            "Yaw-only airframe USD/JSON not found. Generate it first with:\n"
            "  python scripts/convert_concentrated_urdf_yaw_only.py --headless "
            "--output_dir cache/workspace_local/my_airframe_yaw_only "
            "--config configs/joint_response_template.yaml"
        )

    with open(airframe_data_path, "r") as f:
        airframe_data = json.load(f)
    distal_arm_len = airframe_data.get("parameters", {}).get("arm_distal_length", 0.18)

    sim_cfg_data = cfg.get("simulator", {})
    sim_dt = float(sim_cfg_data.get("dt", 0.002))
    sim_duration = float(sim_cfg_data.get("sim_duration", 1.0))
    sim_device = sim_cfg_data.get("device", "cuda:0")
    sim_gravity = sim_cfg_data.get("gravity", [0.0, 0.0, 0.0])
    enable_stabilization = bool(sim_cfg_data.get("enable_stabilization", False))
    soft_endstop_cfg = sim_cfg_data.get("soft_endstop", sim_cfg_data.get("endstop", {}))
    soft_endstop_enabled = bool(soft_endstop_cfg.get("enabled", False))
    soft_endstop_angle_deg = float(soft_endstop_cfg.get("angle_deg", 90.0))
    soft_endstop_angle_rad = float(np.deg2rad(soft_endstop_angle_deg))
    soft_endstop_stiffness = float(soft_endstop_cfg.get("stiffness", 0.0))
    soft_endstop_damping = float(soft_endstop_cfg.get("damping", 0.0))

    sim_cfg = sim_utils.SimulationCfg(
        dt=sim_dt,
        device=sim_device,
        gravity=tuple(sim_gravity),
        physx=PhysxCfg(enable_stabilization=enable_stabilization),
    )
    sim = SimulationContext(sim_cfg)
    camera_cfg = cfg.get("camera", {})
    camera_target = camera_cfg.get("target", [0.0, 0.0, 0.0])
    camera_distance = float(camera_cfg.get("distance", 1.2))
    camera_yaw_deg = float(camera_cfg.get("yaw_deg", 0.0))
    camera_height = float(camera_cfg.get("height", camera_distance))
    yaw_rad = np.deg2rad(camera_yaw_deg)
    camera_eye = [
        camera_target[0] + camera_distance * float(np.cos(yaw_rad)),
        camera_target[1] + camera_distance * float(np.sin(yaw_rad)),
        camera_target[2] + camera_height,
    ]
    sim.set_camera_view(camera_eye, camera_target)
    print("[Simulation] Context created")

    scene_cfg = YawOnlySceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
    scene_cfg.robot.spawn.usd_path = str(usd_path)
    scene = InteractiveScene(scene_cfg)
    robot = scene["robot"]
    sim.reset()

    print(f"[Simulation] Robot loaded: {robot.num_bodies} bodies, {robot.num_joints} joints")
    print(f"[Simulation] Joint names: {robot.joint_names}")

    print("[Fix Base] Using fixed-base articulation (fix_root_link=True)")

    yaw_joint_indices = [robot.joint_names.index(name) for name in YAW_JOINT_NAMES]
    test_cfg = cfg.get("test", {})
    motor_names = test_cfg.get("motor_names", ["motor_0", "motor_3"])
    torque_directions = test_cfg.get("torque_directions", [1.0, -1.0])
    if len(motor_names) != len(torque_directions):
        raise ValueError("test.motor_names and test.torque_directions must have the same length.")
    motor_body_indices = []
    for name in motor_names:
        if name not in robot.body_names:
            available = [n for n in robot.body_names if n.startswith("motor_")]
            raise ValueError(f"Motor '{name}' not found. Available motors: {available}")
        motor_body_indices.append(robot.body_names.index(name))

    joint_pos_initial = torch.zeros((1, robot.num_joints), device=sim.device)
    joint_vel_initial = torch.zeros((1, robot.num_joints), device=sim.device)
    robot.write_joint_state_to_sim(joint_pos_initial, joint_vel_initial)
    sim.step()
    scene.update(sim_dt)

    n_steps = int(sim_duration / sim_dt)
    env_indices = torch.tensor([0], dtype=torch.int32, device=sim.device)
    env_indices_long = torch.tensor([0], dtype=torch.long, device=sim.device)

    times = []
    angles_deg_history = []
    applied_torque_history = []
    passive_torque_history = []
    damping_torque_history = []
    soft_endstop_torque_history = []
    total_reaction_torque_history = []

    joint_cfg = cfg.get("joint", {})
    stiffness = float(joint_cfg.get("stiffness", 0.3))
    joint_damping = float(joint_cfg.get("damping", 0.01))
    hard_endstop_deg = joint_cfg.get("hard_endstop_deg", None)
    hard_limits_deg = _parse_hard_endstop_deg(hard_endstop_deg)
    if hard_limits_deg is not None:
        lower_deg, upper_deg = hard_limits_deg
        lower_rad, upper_rad = np.deg2rad([lower_deg, upper_deg])
        applied_limits = False
        if hasattr(robot, "root_physx_view") and hasattr(robot.root_physx_view, "get_dof_limits"):
            try:
                dof_limits = robot.root_physx_view.get_dof_limits()
                if dof_limits is not None:
                    if dof_limits.ndim == 2:
                        dof_limits = dof_limits.unsqueeze(0)
                    dof_limits[:, yaw_joint_indices, 0] = lower_rad
                    dof_limits[:, yaw_joint_indices, 1] = upper_rad
                    robot.root_physx_view.set_dof_limits(dof_limits)
                    applied_limits = True
            except Exception as exc:
                print(f"[WARN] Failed to set hard endstops via root_physx_view: {exc}")
        if hasattr(robot, "data") and hasattr(robot.data, "joint_pos_limits"):
            try:
                limits = robot.data.joint_pos_limits
                if limits is not None:
                    if limits.ndim == 2:
                        limits = limits.unsqueeze(0)
                    limits[:, yaw_joint_indices, 0] = lower_rad
                    limits[:, yaw_joint_indices, 1] = upper_rad
                    applied_limits = True
            except Exception as exc:
                print(f"[WARN] Failed to set hard endstops via robot.data.joint_pos_limits: {exc}")
        if applied_limits:
            print(f"[Joint Limits] Hard endstops set to [{lower_deg:.1f}, {upper_deg:.1f}] deg")
        else:
            print(f"[Joint Limits] Requested hard endstops [{lower_deg:.1f}, {upper_deg:.1f}] deg; using sim defaults")
    torque_z = float(test_cfg.get("torque_z", 0.3))
    torque_schedule = _build_torque_schedule(test_cfg)

    response_cfg = cfg.get("joint_response", {})
    response_mode = str(response_cfg.get("mode", "linear")).lower()
    use_response_table = response_mode in ("piecewise", "function")
    response_angles = None
    response_torques = None
    if use_response_table:
        response_angles, response_torques = build_response_table(response_cfg, device=sim.device)

    print("\n" + "=" * 80)
    print("Joint Response Test")
    print("=" * 80)
    torque_desc = ", ".join(
        f"{direction:+.1f}*{torque_z:.4f} N*m on {name}"
        for name, direction in zip(motor_names, torque_directions)
    )
    if torque_schedule is None:
        print(f"Applying constant motor torques: {torque_desc}")
    else:
        total_sched = torque_schedule[-1][0]
        print(f"Applying scheduled motor torques (total {total_sched:.2f}s): {torque_desc}")
    print(f"Joint response mode: {response_mode}")
    print(f"Distal arm length: {distal_arm_len:.4f} m")
    print(f"Joint names: {YAW_JOINT_NAMES}")
    if soft_endstop_enabled:
        print(
            f"Soft endstop enabled: ±{soft_endstop_angle_deg:.1f}° "
            f"(k={soft_endstop_stiffness:.3f}, d={soft_endstop_damping:.3f})"
        )

    video_cfg = cfg.get("video", {})
    video_slowdown = args_cli.video if args_cli.video is not None else video_cfg.get("playback_speed", None)
    video_path_cfg = video_cfg.get("path", "results/test1b_yaw_only_joint_response.mp4")
    video_path = resolve_path(video_path_cfg)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    analytics_path_cfg = video_cfg.get("analytics_path", None)
    if args_cli.analytics_path is not None:
        analytics_path_cfg = args_cli.analytics_path
    elif args_cli.analytics:
        analytics_path_cfg = analytics_path_cfg or "results/test1b_yaw_only_analytics.mp4"
    analytics_fps = int(video_cfg.get("analytics_fps", 60))
    recorder = None

    if video_slowdown is not None:
        recorder = Recorder(
            playback_speed=float(video_slowdown),
            mp4_path=str(video_path),
            dt=sim_dt,
            sim=sim,
            camera_eye=camera_eye,
            camera_target=camera_target,
        )
        print(f"[INFO]: Video recording enabled at {video_slowdown}x speed (60 fps)")
        print(f"[INFO]: Output: {video_path}")

    try:
        for step in range(n_steps):
            t = step * sim_dt
            torque_scale = _torque_scale_at_time(t, torque_schedule)
            forces_and_torques = torch.zeros(1, robot.num_bodies, 6, device=sim.device)
            for motor_idx, direction in zip(motor_body_indices, torque_directions):
                forces_and_torques[0, motor_idx, 5] = direction * torque_z * torque_scale

            robot.root_physx_view.apply_forces_and_torques_at_position(
                forces_and_torques[:, :, :3],
                forces_and_torques[:, :, 3:],
                None,
                env_indices,
                is_global=False,
            )

            joint_pos = robot.data.joint_pos[0, yaw_joint_indices]
            joint_vel = robot.data.joint_vel[0, yaw_joint_indices]

            if use_response_table:
                passive_torque = interpolate_torque(joint_pos, response_angles, response_torques)
            else:
                passive_torque = -stiffness * joint_pos
            damping_torque = -joint_damping * joint_vel
            joint_torques = passive_torque + damping_torque
            soft_endstop_torque = torch.zeros_like(joint_pos)
            if soft_endstop_enabled:
                abs_pos = torch.abs(joint_pos)
                violation = torch.clamp(abs_pos - soft_endstop_angle_rad, min=0.0)
                soft_endstop_torque = (
                    -soft_endstop_stiffness * violation * torch.sign(joint_pos)
                    - soft_endstop_damping * joint_vel
                )
                soft_endstop_torque = torch.where(violation > 0, soft_endstop_torque, torch.zeros_like(joint_pos))
                joint_torques = joint_torques + soft_endstop_torque

            robot.set_joint_effort_target(joint_torques.unsqueeze(0), joint_ids=yaw_joint_indices, env_ids=env_indices_long)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            if recorder is not None:
                recorder.record_frame()

            if step % 1 == 0:
                times.append(step * sim_dt)
                angles_deg_history.append((joint_pos.detach().cpu().numpy() * 180.0 / np.pi))
                applied_torque_history.append(torque_z * torque_scale * float(torque_directions[0]))
                passive_torque_history.append(float(passive_torque[0].item()))
                damping_torque_history.append(float(damping_torque[0].item()))
                soft_endstop_torque_history.append(float(soft_endstop_torque[0].item()))
                total_reaction_torque_history.append(float(joint_torques[0].item()))

            if step % 10 == 0:
                print(f"  Step {step:5d}: t={t:.3f}s, joint_0 angle={angles_deg_history[-1][0]:7.3f} deg")
    finally:
        if recorder is not None:
            recorder.save()

    angles_deg_history = np.array(angles_deg_history)
    applied_torque_history = np.array(applied_torque_history)
    passive_torque_history = np.array(passive_torque_history)
    damping_torque_history = np.array(damping_torque_history)
    soft_endstop_torque_history = np.array(soft_endstop_torque_history)
    total_reaction_torque_history = np.array(total_reaction_torque_history)

    plt.figure(figsize=(10, 6))
    plt.plot(times, angles_deg_history[:, 0], "b-", linewidth=2, label="joint_0")
    if angles_deg_history.shape[1] > 1:
        plt.plot(times, angles_deg_history[:, 1], "g--", linewidth=1, label="joint_1")
        plt.plot(times, angles_deg_history[:, 2], "r--", linewidth=1, label="joint_2")
        plt.plot(times, angles_deg_history[:, 3], "m--", linewidth=1, label="joint_3")
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Joint Angle (degrees)", fontsize=14)
    plt.title("Joint Response (Opposing Motor Torques)", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plot_name = output_cfg.get("plot_name", "test1b_yaw_only_joint_response.png")
    plot_path = results_dir / plot_name
    plt.savefig(plot_path.as_posix(), dpi=150)
    print(f"Saved plot: {plot_path}")

    if analytics_path_cfg is not None:
        analytics_path = resolve_path(analytics_path_cfg)
        if use_response_table:
            response_angles_np = response_angles.detach().cpu().numpy()
            response_torques_np = response_torques.detach().cpu().numpy()
        else:
            angle_min = float(np.min(angles_deg_history[:, 0]) * np.pi / 180.0)
            angle_max = float(np.max(angles_deg_history[:, 0]) * np.pi / 180.0)
            if angle_min == angle_max:
                angle_min -= 0.1
                angle_max += 0.1
            response_angles_np = np.linspace(angle_min, angle_max, 200)
            response_torques_np = -stiffness * response_angles_np
        _save_analytics_video(
            times=np.array(times),
            angles_deg=angles_deg_history[:, 0],
            applied_torque=applied_torque_history,
            passive_torque=passive_torque_history,
            damping_torque=damping_torque_history,
            soft_endstop_torque=soft_endstop_torque_history,
            total_reaction_torque=total_reaction_torque_history,
            response_angles_rad=response_angles_np,
            response_torques=response_torques_np,
            output_path=analytics_path,
            fps=analytics_fps,
            playback_speed=video_slowdown,
            dt=sim_dt,
        )

    simulation_app.close()


if __name__ == "__main__":
    main()
