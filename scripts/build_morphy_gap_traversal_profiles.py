from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from erc_design import build_erc_design


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "morphy_gap_traversal_profiles"
DEFAULT_SPRING_ID = "sodemann_41570-316"

RIGHT_ARM_KNOTS_DEG_NM = [
    (-1.0, -0.40),
    (0.0, 0.0),
    (0.5, 0.119),
    (37.5, 0.004),
    (70.0, 0.004),
    (72.25, 0.047),
    (73.30, 0.4),
]

LEFT_ARM_KNOTS_DEG_NM = [
    (-1.08, -0.40),
    (0.0, 0.0),
    (1.0, 0.074),
    (15.75, 0.012),
    (60.25, 0.011),
    (72.25, 0.041),
    (73.40, 0.4),
]


def write_table_csv(path: Path, table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["joint_angle_rad", "torque_nm"])
        writer.writerows(table.detach().cpu().tolist())


def write_profile_png(path: Path, *, name: str, knots_deg_nm: list[tuple[float, float]], result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    requested_angles_deg = [point[0] for point in knots_deg_nm]
    requested_torque_nm = [point[1] for point in knots_deg_nm]
    output_angles_deg = (result.output_torque_table[:, 0].detach().cpu().numpy()) * 180.0 / 3.141592653589793
    output_torque_nm = -result.output_torque_table[:, 1].detach().cpu().numpy()
    repaired_cam_mm = result.repaired_cam_xy_m.detach().cpu().numpy() * 1000.0
    curvature_mm = result.repaired_curvature_radius_m.detach().cpu().numpy() * 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    axes[0].plot(output_angles_deg, output_torque_nm, color="#0072b2", label="repaired output")
    axes[0].scatter(requested_angles_deg, requested_torque_nm, color="#111111", s=18, label="requested knots")
    axes[0].set_title(f"{name} torque")
    axes[0].set_xlabel("joint angle [deg]")
    axes[0].set_ylabel("torque [Nm]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(repaired_cam_mm[:, 0], repaired_cam_mm[:, 1], color="#0072b2")
    axes[1].scatter([0.0], [0.0], color="#111111", s=18)
    axes[1].set_title(f"{name} cam profile")
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("y [mm]")
    axes[1].axis("equal")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(output_angles_deg, curvature_mm, color="#0072b2")
    axes[2].axhline(0.0, color="#111111", linewidth=1)
    axes[2].set_title(f"{name} convexity")
    axes[2].set_xlabel("joint angle [deg]")
    axes[2].set_ylabel("r + r'' [mm]")
    axes[2].grid(True, alpha=0.3)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def export_profile(
    *,
    name: str,
    knots_deg_nm: list[tuple[float, float]],
    output_dir: Path,
    spring_id: str,
    n_grid: int,
    safety_factor: float,
) -> None:
    result = build_erc_design(
        knots_deg_nm,
        spring_id=spring_id,
        n_grid=n_grid,
        safety_factor=safety_factor,
    )

    profile_dir = output_dir / name
    profile_dir.mkdir(parents=True, exist_ok=True)

    result.export_sldcrv(profile_dir / f"{name}_cam_profile.sldcrv")
    result.export_xyz(profile_dir / f"{name}_cam_profile.xyz")
    write_table_csv(profile_dir / f"{name}_torque_table.csv", result.output_torque_table)
    write_profile_png(profile_dir / f"{name}_profile.png", name=name, knots_deg_nm=knots_deg_nm, result=result)

    print(f"{name}:")
    print(f"  spring_id={spring_id}")
    print(f"  scaled={result.was_scaled} alpha={result.alpha:.6g}")
    print(f"  repaired={result.was_repaired}")
    print(f"  torque_rmse_nm={result.torque_rmse_nm:.6g}")
    print(f"  out_dir={profile_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build repaired ERC cam profiles for Morphy gap traversal left/right arms."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for exported left/right profile artifacts.",
    )
    parser.add_argument(
        "--spring-id",
        type=str,
        default=DEFAULT_SPRING_ID,
        help="Spring identifier from configs/springs.yaml.",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=1000,
        help="Number of support-function samples used during design.",
    )
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=1.1,
        help="Energy safety factor for admissibility scaling.",
    )
    args = parser.parse_args()

    export_profile(
        name="right_arm",
        knots_deg_nm=RIGHT_ARM_KNOTS_DEG_NM,
        output_dir=args.output_dir,
        spring_id=args.spring_id,
        n_grid=args.n_grid,
        safety_factor=args.safety_factor,
    )
    export_profile(
        name="left_arm",
        knots_deg_nm=LEFT_ARM_KNOTS_DEG_NM,
        output_dir=args.output_dir,
        spring_id=args.spring_id,
        n_grid=args.n_grid,
        safety_factor=args.safety_factor,
    )


if __name__ == "__main__":
    main()
