from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from erc_design import build_erc_design
from erc_design.designer import resample_by_arclength


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "rigid_and_let_go_profiles"
DEFAULT_SPRING_ID = "sodemann_31420"

ProjectionPlane = Literal["XY", "YZ", "XZ"]
PROJECTION_PLANE_AXES: dict[ProjectionPlane, tuple[int, int]] = {
    "XY": (0, 1),
    "YZ": (1, 2),
    "XZ": (0, 2),
}

ENDSTOP_ANGLE = 82.0
THRESHOLD = 0.04
THRESHOLD_ANGLE_START_DEG = 0.1
THRESHOLD_ANGLE_END_DEG = 15.0
RIGID_AND_LET_GO_KNOTS_DEG_NM = [
    (-ENDSTOP_ANGLE, -THRESHOLD),
    (-ENDSTOP_ANGLE + 2.0, 0.0),
    (-THRESHOLD_ANGLE_END_DEG, 0.0),
    (-THRESHOLD_ANGLE_START_DEG, -THRESHOLD),
    (THRESHOLD_ANGLE_START_DEG, THRESHOLD),
    (THRESHOLD_ANGLE_END_DEG, 0.0),
    (ENDSTOP_ANGLE - 2.0, 0.0),
    (ENDSTOP_ANGLE, THRESHOLD),
]


def write_table_csv(path: Path, table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["joint_angle_rad", "torque_nm"])
        writer.writerows(table.detach().cpu().tolist())


def transform_profile_in_plane_mm(
    xy_m,
    *,
    rotation_deg: float,
    translate_x_mm: float,
    translate_y_mm: float,
    n_points: int | None = None,
):
    """Rotate about (0, 0), then translate a profile in its selected 2D plane."""

    xy = xy_m.detach().cpu()
    if n_points is not None and n_points != xy.shape[0]:
        xy = resample_by_arclength(xy, n_points)

    angle_rad = math.radians(rotation_deg)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    transformed_mm = xy.new_empty(xy.shape)
    transformed_mm[:, 0] = (
        cos_angle * xy[:, 0] - sin_angle * xy[:, 1]
    ) * 1000.0 + translate_x_mm
    transformed_mm[:, 1] = (
        sin_angle * xy[:, 0] + cos_angle * xy[:, 1]
    ) * 1000.0 + translate_y_mm
    return transformed_mm


def write_projected_curve(
    path: Path,
    xy_m,
    *,
    projection_plane: ProjectionPlane,
    rotation_deg: float,
    translate_x_mm: float,
    translate_y_mm: float,
    n_points: int | None = None,
) -> None:
    """Write a transformed 2D profile as SolidWorks-friendly mm XYZ rows."""

    if n_points is None:
        n_points = int(xy_m.shape[0] * 2)
    in_plane_mm = transform_profile_in_plane_mm(
        xy_m,
        rotation_deg=rotation_deg,
        translate_x_mm=translate_x_mm,
        translate_y_mm=translate_y_mm,
        n_points=n_points,
    )

    xyz_mm = in_plane_mm.new_zeros((in_plane_mm.shape[0], 3))
    first_axis, second_axis = PROJECTION_PLANE_AXES[projection_plane]
    xyz_mm[:, first_axis] = in_plane_mm[:, 0]
    xyz_mm[:, second_axis] = in_plane_mm[:, 1]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\r\n") as handle:
        for row in xyz_mm:
            handle.write(f"{row[0]:.9f} {row[1]:.9f} {row[2]:.9f}\n")


def write_profile_png(
    path: Path,
    *,
    name: str,
    knots_deg_nm: list[tuple[float, float]],
    result,
    projection_plane: ProjectionPlane,
    rotation_deg: float,
    translate_x_mm: float,
    translate_y_mm: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    requested_angles_deg = [point[0] for point in knots_deg_nm]
    requested_torque_nm = [point[1] for point in knots_deg_nm]
    output_angles_deg = (result.output_torque_table[:, 0].detach().cpu().numpy()) * 180.0 / 3.141592653589793
    output_torque_nm = -result.output_torque_table[:, 1].detach().cpu().numpy()
    repaired_cam_mm = transform_profile_in_plane_mm(
        result.repaired_cam_xy_m,
        rotation_deg=rotation_deg,
        translate_x_mm=translate_x_mm,
        translate_y_mm=translate_y_mm,
    ).numpy()
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
    axes[1].set_title(f"{name} cam profile ({projection_plane})")
    axes[1].set_xlabel(f"{projection_plane[0]} / x' [mm]")
    axes[1].set_ylabel(f"{projection_plane[1]} / y' [mm]")
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
    projection_plane: ProjectionPlane,
    rotation_deg: float,
    translate_x_mm: float,
    translate_y_mm: float,
) -> None:
    result = build_erc_design(
        knots_deg_nm,
        spring_id=spring_id,
        n_grid=n_grid,
        safety_factor=safety_factor,
    )

    profile_dir = output_dir / name
    profile_dir.mkdir(parents=True, exist_ok=True)

    for extension in ("sldcrv", "xyz"):
        write_projected_curve(
            profile_dir / f"{name}_cam_profile.{extension}",
            result.repaired_cam_xy_m,
            projection_plane=projection_plane,
            rotation_deg=rotation_deg,
            translate_x_mm=translate_x_mm,
            translate_y_mm=translate_y_mm,
        )
    write_table_csv(profile_dir / f"{name}_torque_table.csv", result.output_torque_table)
    write_profile_png(
        profile_dir / f"{name}_profile.png",
        name=name,
        knots_deg_nm=knots_deg_nm,
        result=result,
        projection_plane=projection_plane,
        rotation_deg=rotation_deg,
        translate_x_mm=translate_x_mm,
        translate_y_mm=translate_y_mm,
    )

    print(f"{name}:")
    print(f"  spring_id={spring_id}")
    print(f"  projection_plane={projection_plane}")
    print(f"  rotation_deg={rotation_deg:g}")
    print(f"  translation_x_prime_y_prime_mm=({translate_x_mm:g}, {translate_y_mm:g})")
    print(f"  scaled={result.was_scaled} alpha={result.alpha:.6g}")
    print(f"  repaired={result.was_repaired}")
    print(f"  torque_rmse_nm={result.torque_rmse_nm:.6g}")
    print(f"  out_dir={profile_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a repaired ERC cam profile for the rigid_and_let_go configuration."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for exported rigid_and_let_go profile artifacts.",
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
    parser.add_argument(
        "--projection-plane",
        type=str.upper,
        choices=tuple(PROJECTION_PLANE_AXES),
        default="XY",
        help="3D plane for the exported curve: XY (default), YZ, or XZ.",
    )
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=0.0,
        help="Counterclockwise rotation in the selected plane about (0, 0), in degrees.",
    )
    parser.add_argument(
        "--translate-x-mm",
        type=float,
        default=0.0,
        help="Translation along the selected plane's x' (first named) axis, in mm.",
    )
    parser.add_argument(
        "--translate-y-mm",
        type=float,
        default=0.0,
        help="Translation along the selected plane's y' (second named) axis, in mm.",
    )
    args = parser.parse_args()

    export_profile(
        name="rigid_and_let_go",
        knots_deg_nm=RIGID_AND_LET_GO_KNOTS_DEG_NM,
        output_dir=args.output_dir,
        spring_id=args.spring_id,
        n_grid=args.n_grid,
        safety_factor=args.safety_factor,
        projection_plane=args.projection_plane,
        rotation_deg=args.rotation_deg,
        translate_x_mm=args.translate_x_mm,
        translate_y_mm=args.translate_y_mm,
    )


if __name__ == "__main__":
    main()
