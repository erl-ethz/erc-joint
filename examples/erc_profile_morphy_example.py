"""Smoke test for ERC-generated soft-joint torque profiles.

This script does not run optimization. It:
1. Builds two example ERC joint torque functions from an explicit torque table.
2. Saves torque-profile and cam-profile plots to results/.
3. Exports cam XYZ files for SolidWorks.
4. Generates a Morphy USD containing those functions.
5. Runs a short fixed-base/free-joint simulation.
6. Logs joint angles and ERC torques to CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# ── User-configurable defaults (override via CLI or edit here) ────────────────
DEFAULT_SPRING_ID = "durovis_0.9x6.1x21" # or others in yaml
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="ERC torque-profile simulation")
parser.add_argument(
    "--workspace_id",
    default="workspace",
    help="Workspace folder name created under the local results directory.",
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="results/morphy_erc_profile_example",
    help="Local output directory for plots, CSV, video, analytics, and generated assets.",
)
parser.add_argument("--duration", type=float, default=1.0, help="Simulation duration in seconds")
parser.add_argument("--dt", type=float, default=0.001, help="Simulation timestep")
parser.add_argument("--damping", type=float, default=0.01, help="Soft-joint viscous damping")
parser.add_argument("--n_grid", type=int, default=151, help="ERC torque table samples per joint")
parser.add_argument("--spring_id", type=str, default=DEFAULT_SPRING_ID, help=f"Spring catalog ID (default: {DEFAULT_SPRING_ID})")
parser.add_argument("--video", type=float, default=None, metavar="SPEED", help="Record viewport video at given playback speed (e.g. 0.25 = 4x slower)")
parser.add_argument("--analytics", action="store_true", help="Save animated joint-response analytics video")
parser.add_argument("--analytics_path", type=str, default=None, help="Override analytics video output path")
parser.add_argument("--analytics_fps", type=int, default=24, help="FPS for analytics video (lower = fewer frames to render, same playback speed)")
parser.add_argument("--camera_eye", type=str, default="0.45,0.35,0.8", help="Viewport camera position as 'x,y,z' (default: 0.45,0.35,0.8)")
parser.add_argument("--camera_target", type=str, default="0.05,0.0,0.5", help="Viewport camera look-at point as 'x,y,z' (default: 0.05,0.0,0.5)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from erc_isaac.airframe_encoding import PARAM_NAMES
from erc_isaac.airframe_objective_functions import decode_into_airframe
from erc_isaac.common_utils import Recorder
from erc_isaac.erc_torque_profiles import (
    DEFAULT_SPRING_CATALOG,
    DEFAULT_SPRING_ID,
    build_tabular_erc_torque_function,
)
from erc_isaac.morphy_simulator import MorphySimulator

# ── Applied torque schedule (time_s, left_Nm, right_Nm) ──────────────────────
# Piecewise-constant: each row is active from its time until the next row.
TORQUE_SCHEDULE = [
    (0.0,  0.00,  0.00),
    (0.1,  0.09, -0.09),
    (0.2,  0.2, -0.2),
    (0.4,  0.0, 0.0),
    (0.7,  -0.09,  0.09),
    (0.8,  -0.2,  0.2),
    (0.9,  0.0,  0.0),
    (1.0,  0., -0.),
]
# ─────────────────────────────────────────────────────────────────────────────


# Explicit joint-angle / torque knots in degrees and Nm.
TORQUE_PROFILE_KNOTS_DEG_NM = (
    (-2.0, 0.4),
    (0.0, 0.0),
    (1.0, -0.2),
    (39.0, 0.0),
    (77.5, 0.0),
    (78.0, -0.4),
)


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _expand_for_playback_speed(
    n_steps: int, dt: float, playback_speed: float, target_fps: int = 60
):
    import numpy as np
    sim_time_per_frame = (1.0 / target_fps) * playback_speed
    frame_count = 0
    indices = []
    for i in range(n_steps):
        expected = int((i * dt) / sim_time_per_frame)
        to_write = expected - frame_count
        if to_write > 0:
            indices.extend([i] * to_write)
            frame_count += to_write
    return np.array(indices, dtype=np.int64) if indices else np.array([0], dtype=np.int64)


def _cam_contact_point(joint_angle_rad: float, support_angles, cam_mm):
    """Return the (x_mm, y_mm) contact point on the cam for a given joint angle."""
    import numpy as np
    support_angle = joint_angle_rad * 0.5          # joint → support angle
    i = int(np.searchsorted(support_angles, support_angle, side="right")) - 1
    i = int(np.clip(i, 0, len(cam_mm) - 1))
    return cam_mm[i, 0], cam_mm[i, 1]


def save_analytics_video(
    times,
    left_angles_deg,
    right_angles_deg,
    left_applied_nm,
    right_applied_nm,
    left_profile_nm,
    right_profile_nm,
    left_table,           # (n,2)  [joint_angle_rad, torque_nm]
    right_table,
    left_cam_mm,          # (n,2)  [x_mm, y_mm]  repaired cam
    right_cam_mm,
    left_support_angles,  # (n,)   support angles (rad) matching cam rows
    right_support_angles,
    output_path: Path,
    left_pre_repair_table=None,
    right_pre_repair_table=None,
    left_pre_repair_cam_mm=None,
    right_pre_repair_cam_mm=None,
    playback_speed: float | None = None,
    dt: float | None = None,
    fps: int = 24,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if playback_speed is not None and dt is not None:
        idx = _expand_for_playback_speed(len(times), dt, playback_speed, fps)
    else:
        idx = np.arange(len(times))

    times_f      = times[idx]
    left_ang_f   = left_angles_deg[idx]
    right_ang_f  = right_angles_deg[idx]
    left_app_f   = left_applied_nm[idx]
    right_app_f  = right_applied_nm[idx]
    left_prof_f  = left_profile_nm[idx]
    right_prof_f = right_profile_nm[idx]

    left_curve_deg  = left_table[:, 0]  * 180.0 / math.pi
    left_curve_nm   = left_table[:, 1]
    right_curve_deg = right_table[:, 0] * 180.0 / math.pi
    right_curve_nm  = right_table[:, 1]
    left_pre_curve_deg = None if left_pre_repair_table is None else left_pre_repair_table[:, 0] * 180.0 / math.pi
    left_pre_curve_nm = None if left_pre_repair_table is None else left_pre_repair_table[:, 1]
    right_pre_curve_deg = None if right_pre_repair_table is None else right_pre_repair_table[:, 0] * 180.0 / math.pi
    right_pre_curve_nm = None if right_pre_repair_table is None else right_pre_repair_table[:, 1]

    # pre-compute cam contact points for every frame
    left_ang_rad_f  = left_ang_f  * math.pi / 180.0
    right_ang_rad_f = right_ang_f * math.pi / 180.0
    left_cx  = np.array([_cam_contact_point(a, left_support_angles,  left_cam_mm)[0]  for a in left_ang_rad_f])
    left_cy  = np.array([_cam_contact_point(a, left_support_angles,  left_cam_mm)[1]  for a in left_ang_rad_f])
    right_cx = np.array([_cam_contact_point(a, right_support_angles, right_cam_mm)[0] for a in right_ang_rad_f])
    right_cy = np.array([_cam_contact_point(a, right_support_angles, right_cam_mm)[1] for a in right_ang_rad_f])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle("ERC Joint Response Analytics", fontsize=13)
    ax_app, ax_cam = axes[0, 0], axes[0, 1]
    ax_ang, ax_erc = axes[1, 0], axes[1, 1]

    # ── [0,0] applied torques ────────────────────────────────────────────────
    line_la, = ax_app.plot([], [], "b-", lw=1.5, label="left")
    line_ra, = ax_app.plot([], [], "r-", lw=1.5, label="right")
    ax_app.set_xlim(times[0], times[-1])
    all_app = np.concatenate([left_applied_nm, right_applied_nm])
    pad = max(0.01, 0.1 * float(np.max(np.abs(all_app))))
    ax_app.set_ylim(float(np.min(all_app)) - pad, float(np.max(all_app)) + pad)
    ax_app.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax_app.set_xlabel("time [s]"); ax_app.set_ylabel("torque [Nm]")
    ax_app.set_title("Applied torque"); ax_app.legend(fontsize=8); ax_app.grid(True, alpha=0.3)

    # ── [0,1] cam profiles + contact dots ────────────────────────────────────
    if left_pre_repair_cam_mm is not None:
        ax_cam.plot(
            left_pre_repair_cam_mm[:, 0],
            left_pre_repair_cam_mm[:, 1],
            "b--",
            lw=1.0,
            alpha=0.35,
            label="left cam (pre-repair)",
        )
    if right_pre_repair_cam_mm is not None:
        ax_cam.plot(
            right_pre_repair_cam_mm[:, 0],
            right_pre_repair_cam_mm[:, 1],
            "r--",
            lw=1.0,
            alpha=0.35,
            label="right cam (pre-repair)",
        )
    ax_cam.plot(left_cam_mm[:, 0],  left_cam_mm[:, 1],  "b-", lw=1.5, alpha=0.7, label="left cam")
    ax_cam.plot(right_cam_mm[:, 0], right_cam_mm[:, 1], "r-", lw=1.5, alpha=0.7, label="right cam")
    ax_cam.scatter([0.0], [0.0], color="k", s=40, zorder=5, marker="+")   # pivot
    cdot_l, = ax_cam.plot([], [], "bo", ms=8, zorder=6, label="left contact")
    cdot_r, = ax_cam.plot([], [], "ro", ms=8, zorder=6, label="right contact")
    arm_l, = ax_cam.plot([], [], "b-", lw=1, alpha=0.5, zorder=4)
    arm_r, = ax_cam.plot([], [], "r-", lw=1, alpha=0.5, zorder=4)
    # always include the pivot (0,0) in the axis extent
    cam_extent_terms = [left_cam_mm, right_cam_mm, np.array([[0.0, 0.0]])]
    if left_pre_repair_cam_mm is not None:
        cam_extent_terms.append(left_pre_repair_cam_mm)
    if right_pre_repair_cam_mm is not None:
        cam_extent_terms.append(right_pre_repair_cam_mm)
    all_cam_xy = np.concatenate(cam_extent_terms)
    cam_pad = 0.08 * max(1.0, float(np.max(np.abs(all_cam_xy))))
    ax_cam.set_xlim(float(np.min(all_cam_xy[:, 0])) - cam_pad, float(np.max(all_cam_xy[:, 0])) + cam_pad)
    ax_cam.set_ylim(float(np.min(all_cam_xy[:, 1])) - cam_pad, float(np.max(all_cam_xy[:, 1])) + cam_pad)
    ax_cam.set_aspect("equal")
    ax_cam.set_xlabel("x [mm]"); ax_cam.set_ylabel("y [mm]")
    ax_cam.set_title("Cam profile — contact point"); ax_cam.legend(fontsize=8); ax_cam.grid(True, alpha=0.3)

    # ── [1,0] joint angles ───────────────────────────────────────────────────
    line_lang, = ax_ang.plot([], [], "b-", lw=1.5, label="left")
    line_rang, = ax_ang.plot([], [], "r-", lw=1.5, label="right")
    ax_ang.set_xlim(times[0], times[-1])
    all_ang = np.concatenate([left_angles_deg, right_angles_deg])
    pad = max(1.0, 0.1 * float(np.max(np.abs(all_ang))))
    ax_ang.set_ylim(float(np.min(all_ang)) - pad, float(np.max(all_ang)) + pad)
    ax_ang.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax_ang.set_xlabel("time [s]"); ax_ang.set_ylabel("angle [deg]")
    ax_ang.set_title("Joint angles"); ax_ang.legend(fontsize=8); ax_ang.grid(True, alpha=0.3)

    # ── [1,1] ERC torque profile + operating dots ────────────────────────────
    if left_pre_curve_deg is not None:
        ax_erc.plot(
            left_pre_curve_deg,
            left_pre_curve_nm,
            "b--",
            lw=1.0,
            alpha=0.35,
            label="left profile (pre-repair)",
        )
    if right_pre_curve_deg is not None:
        ax_erc.plot(
            right_pre_curve_deg,
            right_pre_curve_nm,
            "r--",
            lw=1.0,
            alpha=0.35,
            label="right profile (pre-repair)",
        )
    ax_erc.plot(left_curve_deg,  left_curve_nm,  "b-", lw=1.5, alpha=0.5, label="left profile")
    ax_erc.plot(right_curve_deg, right_curve_nm, "r-", lw=1.5, alpha=0.5, label="right profile")
    edot_l, = ax_erc.plot([], [], "bo", ms=8, label="left current")
    edot_r, = ax_erc.plot([], [], "ro", ms=8, label="right current")
    torque_angle_terms = [left_curve_deg, right_curve_deg]
    torque_value_terms = [left_curve_nm, right_curve_nm]
    if left_pre_curve_deg is not None:
        torque_angle_terms.append(left_pre_curve_deg)
        torque_value_terms.append(left_pre_curve_nm)
    if right_pre_curve_deg is not None:
        torque_angle_terms.append(right_pre_curve_deg)
        torque_value_terms.append(right_pre_curve_nm)
    all_c_ang = np.concatenate(torque_angle_terms)
    all_c_nm  = np.concatenate(torque_value_terms)
    pad_a = 0.05 * max(1.0, float(np.max(all_c_ang)) - float(np.min(all_c_ang)))
    pad_t = 0.05 * max(0.01, float(np.max(np.abs(all_c_nm))))
    ax_erc.set_xlim(float(np.min(all_c_ang)) - pad_a, float(np.max(all_c_ang)) + pad_a)
    ax_erc.set_ylim(float(np.min(all_c_nm)) - pad_t,  float(np.max(all_c_nm)) + pad_t)
    ax_erc.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax_erc.axvline(0, color="k", lw=0.5, alpha=0.4)
    ax_erc.set_xlabel("angle [deg]"); ax_erc.set_ylabel("torque [Nm]")
    ax_erc.set_title("ERC profile — operating point"); ax_erc.legend(fontsize=8); ax_erc.grid(True, alpha=0.3)

    def _update(i):
        sl = slice(None, i + 1)
        line_la.set_data(times_f[sl],   left_app_f[sl])
        line_ra.set_data(times_f[sl],  right_app_f[sl])
        line_lang.set_data(times_f[sl],  left_ang_f[sl])
        line_rang.set_data(times_f[sl], right_ang_f[sl])
        edot_l.set_data([left_ang_f[i]],  [left_prof_f[i]])
        edot_r.set_data([right_ang_f[i]], [right_prof_f[i]])
        cdot_l.set_data([left_cx[i]],  [left_cy[i]])
        cdot_r.set_data([right_cx[i]], [right_cy[i]])
        arm_l.set_data([0.0, left_cx[i]],  [0.0, left_cy[i]])
        arm_r.set_data([0.0, right_cx[i]], [0.0, right_cy[i]])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(times_f)
    try:
        import cv2
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        if not vw.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open")
        print(f"[Analytics] Writing {total} frames → {output_path}")
        try:
            for i in range(total):
                _update(i)
                fig.canvas.draw()
                buf = np.asarray(fig.canvas.buffer_rgba())
                vw.write(cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR))
                if i % 50 == 0 and i > 0:
                    print(f"[Analytics] ... {i}/{total}")
        finally:
            vw.release()
        print(f"[Analytics] Saved → {output_path}")
    except Exception as exc:
        frames_dir = output_path.parent / f"{output_path.stem}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i in range(total):
            _update(i)
            fig.savefig(frames_dir / f"frame_{i:05d}.png", dpi=100)
        print(f"[Analytics] cv2 failed ({exc}). Frames saved → {frames_dir}")
    plt.close(fig)


def lookup_torque(t: float, schedule: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Return (left_Nm, right_Nm) for time t using piecewise-constant lookup."""
    left, right = schedule[0][1], schedule[0][2]
    for time_s, l, r in schedule:
        if t >= time_s:
            left, right = l, r
        else:
            break
    return left, right


def result_torque_to_output_table(result, torque_nm: torch.Tensor):
    import numpy as np

    angles = result.output_torque_table[:, 0].detach().cpu().numpy()
    output_torque = result.output_torque_table[:, 1].detach().cpu().numpy()
    repaired_torque = result.repaired_torque_nm.detach().cpu().numpy()
    source_torque = torque_nm.detach().cpu().numpy()
    direct_err = float(np.mean(np.abs(output_torque - repaired_torque)))
    inverted_err = float(np.mean(np.abs(output_torque + repaired_torque)))
    sign = -1.0 if inverted_err < direct_err else 1.0
    return np.column_stack((angles, sign * source_torque))


def save_erc_design_report(result, joint_name: str, results_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    angles_deg = (result.output_torque_table[:, 0].detach().cpu() * 180.0 / math.pi).numpy()
    target = result.target_torque_nm.detach().cpu().numpy()
    scaled = result.scaled_torque_nm.detach().cpu().numpy()
    admissible = result.repaired_torque_nm.detach().cpu().numpy()

    cam = result.cam_xy_m.detach().cpu().numpy() * 1000.0
    repaired_cam = result.repaired_cam_xy_m.detach().cpu().numpy() * 1000.0
    radius = result.support_radius_m.detach().cpu().numpy() * 1000.0
    repaired_radius = result.repaired_support_radius_m.detach().cpu().numpy() * 1000.0
    curvature = result.curvature_radius_m.detach().cpu().numpy() * 1000.0
    repaired_curvature = result.repaired_curvature_radius_m.detach().cpu().numpy() * 1000.0

    status_parts = [f"alpha={result.alpha:.4g}", f"RMSE={result.torque_rmse_nm:.4g} Nm"]
    if result.was_scaled:
        status_parts.append("energy-scaled")
    if result.was_repaired:
        status_parts.append("REPAIRED")
    else:
        status_parts.append("no repair needed")
    status = "  |  ".join(status_parts)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"ERC Design — {joint_name} joint\n{status}", fontsize=12)

    ax = axes[0, 0]
    ax.plot(angles_deg, target, color="#303030", label="requested")
    if result.was_scaled:
        ax.plot(angles_deg, scaled, "--", color="#b7791f", label="energy-scaled")
    if result.was_repaired:
        ax.plot(angles_deg, admissible, color="#0072b2", label="admissible (repaired)")
    else:
        ax.plot(angles_deg, admissible, color="#0072b2", label="admissible")
    ax.set_title("Torque Profile")
    ax.set_xlabel("joint angle [deg]")
    ax.set_ylabel("torque [Nm]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    if result.was_repaired:
        ax.plot(cam[:, 0], cam[:, 1], color="#aaaaaa", label="pre-repair", lw=1)
    ax.plot(repaired_cam[:, 0], repaired_cam[:, 1], color="#0072b2", label="cam profile")
    ax.scatter([0.0], [0.0], color="#303030", s=20, zorder=5)
    ax.set_title("Cam Profile")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    if result.was_repaired:
        ax.plot(angles_deg, radius, color="#aaaaaa", label="pre-repair", lw=1)
    ax.plot(angles_deg, repaired_radius, color="#0072b2", label="support radius")
    ax.set_title("Support Radius")
    ax.set_xlabel("joint angle [deg]")
    ax.set_ylabel("r [mm]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.axhline(0.0, color="#303030", lw=1)
    if result.was_repaired:
        ax.plot(angles_deg, curvature, color="#aaaaaa", label="pre-repair", lw=1)
    ax.plot(angles_deg, repaired_curvature, color="#0072b2", label="convexity radius")
    ax.set_title("Convexity Radius  (must be ≥ 0)")
    ax.set_xlabel("joint angle [deg]")
    ax.set_ylabel("r + r'' [mm]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    pdf_path = results_dir / f"erc_profile_{joint_name}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ERC] Profile plot saved to {pdf_path}")

    xyz_path = results_dir / f"erc_profile_{joint_name}_cam.xyz"
    result.export_xyz(xyz_path)
    print(f"[ERC] Cam XYZ saved to {xyz_path}")


def main() -> None:
    results_dir = resolve_repo_path(args_cli.results_dir)
    workspace_dir = results_dir / args_cli.workspace_id
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[ERC] Building example ERC torque functions")
    left_build = build_tabular_erc_torque_function(
        TORQUE_PROFILE_KNOTS_DEG_NM,
        "torque_fn_left",
        False,
        DEFAULT_SPRING_CATALOG,
        args_cli.spring_id,
        args_cli.n_grid,
        1.1,
        0.6,
    )
    right_build = build_tabular_erc_torque_function(
        TORQUE_PROFILE_KNOTS_DEG_NM,
        "torque_fn_right",
        True,
        DEFAULT_SPRING_CATALOG,
        args_cli.spring_id,
        args_cli.n_grid,
        1.1,
        0.6,
    )
    torque_fn_left = left_build.function_string
    torque_fn_right = right_build.function_string
    left_result = left_build.design_result
    right_result = right_build.design_result

    print(f"[ERC] Left  — repaired={left_result.was_repaired}, scaled={left_result.was_scaled}, RMSE={left_result.torque_rmse_nm:.4g} Nm")
    print(f"[ERC] Right — repaired={right_result.was_repaired}, scaled={right_result.was_scaled}, RMSE={right_result.torque_rmse_nm:.4g} Nm")

    save_erc_design_report(left_result, "left", results_dir)
    save_erc_design_report(right_result, "right", results_dir)

    print(f"[ERC] Generating USD workspace: {workspace_dir}")
    geometry_x_0_1 = [0.5] * len(PARAM_NAMES)
    usd_path = decode_into_airframe(
        geometry_x_0_1,
        workspace_dir,
        torque_fn_left,
        torque_fn_right,
    )

    device = getattr(args_cli, "device", None) or "cuda:0"
    sim = MorphySimulator(
        usd_path=str(usd_path),
        n_envs=1,
        dt=args_cli.dt,
        device=device,
        gravity=(0.0, 0.0, 0.0),
        motor_model="motors_disabled",
        revolute_joint_damping_coeff=args_cli.damping,
        fix_base=True,
        results_dir=str(results_dir),
    )

    initial_pos = torch.zeros((1, 2), device=sim.device)
    initial_vel = torch.zeros((1, 2), device=sim.device)
    sim.robot.write_joint_state_to_sim(initial_pos, initial_vel)

    recorder = None
    if args_cli.video is not None:
        video_path = results_dir / "erc_profile.mp4"
        cam_eye    = [float(v) for v in args_cli.camera_eye.split(",")]
        cam_target = [float(v) for v in args_cli.camera_target.split(",")]
        recorder = Recorder(
            playback_speed=args_cli.video,
            mp4_path=str(video_path),
            dt=sim.dt,
            sim=sim.sim,
            camera_eye=cam_eye,
            camera_target=cam_target,
        )
        print(f"[ERC] Video recording at {args_cli.video}x speed → {video_path}")

    csv_path = results_dir / "erc_profile_joint_response.csv"
    n_steps = int(args_cli.duration / args_cli.dt)
    log_stride = max(1, n_steps // 200)

    # analytics history (collected at every step when --analytics is set)
    an_times, an_la, an_ra, an_lapp, an_rapp, an_lprof, an_rprof = [], [], [], [], [], [], []

    print(f"[ERC] Running {n_steps} steps, logging to {csv_path}")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_s",
                "left_angle_deg",
                "right_angle_deg",
                "left_profile_torque_nm",
                "right_profile_torque_nm",
                "left_damping_torque_nm",
                "right_damping_torque_nm",
                "left_applied_torque_nm",
                "right_applied_torque_nm",
            ]
        )

        for step in range(n_steps + 1):
            t = step * args_cli.dt
            joint_pos, joint_vel = sim.get_joint_state()
            interp, damping = sim.compute_joint_torques_decomposed(joint_pos, joint_vel)

            left_applied, right_applied = lookup_torque(t, TORQUE_SCHEDULE)

            if step % log_stride == 0:
                writer.writerow(
                    [
                        t,
                        float(joint_pos[0, 0].detach().cpu() * 180.0 / torch.pi),
                        float(joint_pos[0, 1].detach().cpu() * 180.0 / torch.pi),
                        float(interp[0, 0].detach().cpu()),
                        float(interp[0, 1].detach().cpu()),
                        float(damping[0, 0].detach().cpu()),
                        float(damping[0, 1].detach().cpu()),
                        left_applied,
                        right_applied,
                    ]
                )

            if args_cli.analytics:
                an_times.append(t)
                an_la.append(float(joint_pos[0, 0].detach().cpu() * 180.0 / torch.pi))
                an_ra.append(float(joint_pos[0, 1].detach().cpu() * 180.0 / torch.pi))
                an_lapp.append(left_applied)
                an_rapp.append(right_applied)
                an_lprof.append(float(interp[0, 0].detach().cpu()))
                an_rprof.append(float(interp[0, 1].detach().cpu()))

            if step < n_steps:
                applied = torch.tensor(
                    [[left_applied, right_applied]], dtype=torch.float32, device=sim.device
                )
                sim.step(extra_joint_torques=applied)
                if recorder is not None:
                    recorder.record_frame()

    if recorder is not None:
        recorder.save()

    if args_cli.analytics:
        import numpy as np
        left_pre_repair_table = None
        right_pre_repair_table = None
        left_pre_repair_cam_mm = None
        right_pre_repair_cam_mm = None
        if left_result.was_repaired:
            left_pre_repair_table = result_torque_to_output_table(left_result, left_result.scaled_torque_nm)
            left_pre_repair_cam_mm = left_result.cam_xy_m.detach().cpu().numpy() * 1000.0
        if right_result.was_repaired:
            right_pre_repair_table = result_torque_to_output_table(right_result, right_result.scaled_torque_nm)
            right_pre_repair_cam_mm = right_result.cam_xy_m.detach().cpu().numpy() * 1000.0
        analytics_path = (
            resolve_repo_path(args_cli.analytics_path)
            if args_cli.analytics_path
            else results_dir / "erc_profile_analytics.mp4"
        )
        save_analytics_video(
            times=np.array(an_times),
            left_angles_deg=np.array(an_la),
            right_angles_deg=np.array(an_ra),
            left_applied_nm=np.array(an_lapp),
            right_applied_nm=np.array(an_rapp),
            left_profile_nm=np.array(an_lprof),
            right_profile_nm=np.array(an_rprof),
            left_table=left_result.output_torque_table.detach().cpu().numpy(),
            right_table=right_result.output_torque_table.detach().cpu().numpy(),
            left_cam_mm=left_result.repaired_cam_xy_m.detach().cpu().numpy() * 1000.0,
            right_cam_mm=right_result.repaired_cam_xy_m.detach().cpu().numpy() * 1000.0,
            left_support_angles=left_result.support_angles_rad.detach().cpu().numpy(),
            right_support_angles=right_result.support_angles_rad.detach().cpu().numpy(),
            left_pre_repair_table=left_pre_repair_table,
            right_pre_repair_table=right_pre_repair_table,
            left_pre_repair_cam_mm=left_pre_repair_cam_mm,
            right_pre_repair_cam_mm=right_pre_repair_cam_mm,
            output_path=analytics_path,
            playback_speed=args_cli.video,
            dt=args_cli.dt,
            fps=args_cli.analytics_fps,
        )

    sim.plot_joint_dynamics()
    print("[ERC] Complete")
    print(f"[ERC] CSV: {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
