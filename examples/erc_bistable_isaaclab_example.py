from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from erc_design import (
    FunctionTorqueProfileConfig,
    build_function_profile,
    build_erc_design_from_function,
    torque_table_to_function_string,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "bistable_erc_example.yaml"


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_simulation_cfg(cfg: dict) -> dict:
    return cfg.get("isaaclab_simulation") or cfg.get("external_simulation", {})


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def write_table_csv(path: Path, table: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["joint_angle_rad", "torque_nm"])
        for row in table.detach().cpu().tolist():
            writer.writerow(row)


def write_function_file(path: Path, function_source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source = "import torch\n" + function_source.strip() + "\n"
    path.write_text(source, encoding="utf-8")


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def append_forward_arg(cmd: list[str], flag: str, value: str | float | None) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def absolutize_config_path(path_value: str | Path | None) -> str | None:
    if path_value is None:
        return None
    return str(resolve_repo_path(path_value))


def colorize_upj_usd(usd_path: Path) -> None:
    try:
        from pxr import Gf, Sdf, Usd, UsdShade
    except Exception as exc:
        print(f"[UPJ] Skipping USD recolor; pxr unavailable: {exc}")
        return

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD for recoloring: {usd_path}")

    color_rules = {
        "base_link": (0.533, 0.533, 0.533),
        "arm_segment_1_": (1.0, 0.549, 0.0),
        "arm_segment_2_": (1.0, 0.843, 0.0),
        "dummy_link_yaw_": (0.10, 0.10, 0.10),
        "motor_": (0.196, 0.804, 0.196),
    }

    looks_path = Sdf.Path("/quadrotor/Looks")
    UsdShade.Material.Define(stage, looks_path.AppendChild("Placeholder"))
    placeholder = stage.GetPrimAtPath(looks_path.AppendChild("Placeholder"))
    if placeholder.IsValid():
        stage.RemovePrim(placeholder.GetPath())

    def ensure_material(name: str, color: tuple[float, float, float]):
        material_path = looks_path.AppendChild(name)
        shader_path = material_path.AppendChild("PreviewSurface")
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    materials = {
        key: ensure_material(f"ERC_{key.rstrip('_')}", color)
        for key, color in color_rules.items()
    }

    changed = 0
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if not path_str.endswith("/visuals"):
            continue
        material = None
        if "/base_link/visuals" in path_str:
            material = materials["base_link"]
        else:
            for key in ("arm_segment_1_", "arm_segment_2_", "dummy_link_yaw_", "motor_"):
                if f"/{key}" in path_str:
                    material = materials[key]
                    break
        if material is None:
            continue
        UsdShade.MaterialBindingAPI(prim).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        changed += 1

    stage.Save()
    print(f"[UPJ] Rebound materials on visual prims: {changed}")


def build_upj_joint_response_config(
    *,
    cfg: dict,
    result,
    recommended_applied_torque_nm: float,
    generated_config_path: Path,
    analytics_fps: int | None = None,
) -> dict:
    ext_cfg = get_simulation_cfg(cfg)
    default_joint_response_cfg_path = resolve_repo_path(
        ext_cfg.get("template_config", "configs/joint_response_template.yaml")
    )
    if not default_joint_response_cfg_path.exists():
        raise FileNotFoundError(
            f"Joint-response template config not found: {default_joint_response_cfg_path}"
        )

    upj_cfg = load_yaml(default_joint_response_cfg_path)
    overrides = ext_cfg.get("overrides", {})
    if overrides:
        deep_update(upj_cfg, overrides)

    workspace_dir = resolve_repo_path(ext_cfg["workspace_dir"])
    usd_path = workspace_dir / "concentrated_quadrotor_yaw_only.usd"
    airframe_data_path = workspace_dir / "airframe_data.json"
    sim_results_dir = resolve_repo_path(ext_cfg["simulation_output_dir"])

    upj_cfg["airframe"] = {
        "usd_path": str(usd_path),
        "airframe_data_path": str(airframe_data_path),
    }
    upj_cfg["joint_response"] = {
        "mode": "piecewise",
        "num_samples": int(result.output_torque_table.shape[0]),
        "piecewise": {
            "points": result.output_torque_table.detach().cpu().tolist(),
        },
    }
    upj_cfg.setdefault("test", {})
    upj_cfg["test"]["torque_z"] = float(recommended_applied_torque_nm)
    upj_cfg.setdefault("output", {})
    upj_cfg["output"]["results_dir"] = str(sim_results_dir)
    upj_cfg["output"]["plot_name"] = "upj_joint_response_chart.png"
    upj_cfg["output"]["joint_response_plot_name"] = "upj_programmable_joint_response_profile.png"
    upj_cfg.setdefault("video", {})
    upj_cfg["video"]["path"] = absolutize_config_path(upj_cfg["video"].get("path"))
    upj_cfg["video"]["analytics_path"] = absolutize_config_path(
        upj_cfg["video"].get("analytics_path")
    )
    if analytics_fps is not None:
        upj_cfg["video"]["analytics_fps"] = int(analytics_fps)

    write_yaml(generated_config_path, upj_cfg)
    return upj_cfg


def ensure_upj_assets(
    *,
    cfg: dict,
    generated_config_path: Path,
    forward_args: list[str],
) -> None:
    ext_cfg = get_simulation_cfg(cfg)
    workspace_dir = resolve_repo_path(ext_cfg["workspace_dir"])
    usd_path = workspace_dir / "concentrated_quadrotor_yaw_only.usd"
    airframe_data_path = workspace_dir / "airframe_data.json"

    if usd_path.exists() and airframe_data_path.exists():
        colorize_upj_usd(usd_path)
        return
    if not bool(ext_cfg.get("convert_if_missing", True)):
        raise FileNotFoundError(
            "UPJ USD asset is missing and convert_if_missing is disabled."
        )

    converter_path = PROJECT_ROOT / "scripts" / "convert_concentrated_urdf_yaw_only.py"
    if not converter_path.exists():
        raise FileNotFoundError(f"Local converter not found: {converter_path}")

    cmd = [
        sys.executable,
        str(converter_path),
        "--output_dir",
        str(workspace_dir),
        "--config",
        str(generated_config_path),
        "--headless",
    ]
    print(f"[UPJ] Generating USD asset: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    if not usd_path.exists() or not airframe_data_path.exists():
        raise FileNotFoundError(
            "UPJ asset conversion completed without producing the expected USD/JSON files."
        )
    colorize_upj_usd(usd_path)


def run_upj_joint_response(
    *,
    cfg: dict,
    generated_config_path: Path,
    forward_args: list[str],
) -> None:
    test_path = PROJECT_ROOT / "erc_isaac" / "joint_response.py"
    if not test_path.exists():
        raise FileNotFoundError(f"Local joint_response.py not found: {test_path}")

    cmd = [sys.executable, str(test_path), "--config", str(generated_config_path), *forward_args]
    print(f"[UPJ] Launching simulation: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def save_plot(path: Path, result, profile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    angles_deg = result.output_torque_table[:, 0].detach().cpu().numpy() * 180.0 / math.pi
    requested_angles_deg = profile.angles.detach().cpu().numpy() * 180.0 / math.pi
    requested_torque = profile.torques.detach().cpu().numpy()
    repaired_torque = result.output_torque_table[:, 1].detach().cpu().numpy()
    curvature_mm = result.repaired_curvature_radius_m.detach().cpu().numpy() * 1000.0

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    ax0.plot(angles_deg, repaired_torque, color="#0072b2", label="ERC repaired Isaac-facing output")
    ax0.scatter(
        requested_angles_deg,
        requested_torque,
        color="#111111",
        s=18,
        label="Isaac-facing function samples",
    )
    ax0.set_xlabel("joint angle [deg]")
    ax0.set_ylabel("torque [Nm]")
    ax0.set_title("Bistable Function Through ERC Repair Pipeline")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    ax1.plot(angles_deg, curvature_mm, color="#0072b2")
    ax1.axhline(0.0, color="#111111", linewidth=1)
    ax1.set_xlabel("joint angle [deg]")
    ax1.set_ylabel("repaired r + r'' [mm]")
    ax1.set_title("Convexity Radius After Repair")
    ax1.grid(True, alpha=0.3)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a bistable repaired ERC torque table and launch the local "
            "Isaac Lab joint-response test."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare repaired ERC assets and the local simulation config, but do not run the simulation.",
    )
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Force regeneration of the local yaw-only USD asset.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Forward headless mode to the local Isaac Lab scripts.",
    )
    parser.add_argument(
        "--video",
        type=float,
        default=None,
        help="Record viewport video at the given playback speed in the local simulation.",
    )
    parser.add_argument(
        "--analytics",
        action="store_true",
        help="Enable analytics video generation in the local simulation.",
    )
    parser.add_argument(
        "--analytics_path",
        type=str,
        default=None,
        help="Override the analytics video output path used by the local simulation.",
    )
    parser.add_argument(
        "--analytics_fps",
        type=int,
        default=None,
        help="Override analytics video FPS in the local simulation.",
    )
    args, forward_args = parser.parse_known_args()

    if args.headless:
        forward_args.append("--headless")
    append_forward_arg(forward_args, "--video", args.video)
    if args.analytics:
        forward_args.append("--analytics")
    append_forward_arg(
        forward_args,
        "--analytics_path",
        absolutize_config_path(args.analytics_path),
    )

    cfg = load_yaml(args.config)
    design_cfg = cfg.get("erc_design", {})
    fn_cfg = cfg.get("function_profile", {})
    validation_cfg = cfg.get("validation", {})
    isaac_cfg = cfg.get("isaac_example", {})
    output_cfg = cfg.get("output", {})

    profile = build_function_profile(
        FunctionTorqueProfileConfig(
            name=str(fn_cfg.get("name", "expression")),
            expression=fn_cfg.get("expression"),
            angle_min_rad=math.radians(float(fn_cfg.get("angle_min_deg", -85.0))),
            angle_max_rad=math.radians(float(fn_cfg.get("angle_max_deg", 85.0))),
            num_samples=int(fn_cfg.get("num_samples", 400)),
            amplitude=float(fn_cfg.get("amplitude", 1.0)),
            frequency=float(fn_cfg.get("frequency", 1.0)),
            phase=float(fn_cfg.get("phase", 0.0)),
            offset=float(fn_cfg.get("offset", 0.0)),
            torque_clip_nm=None
            if fn_cfg.get("torque_clip_nm") is None
            else float(fn_cfg.get("torque_clip_nm")),
            a=float(fn_cfg.get("a", 0.3)),
            b=float(fn_cfg.get("b", 50.0)),
            endstop_angle_rad=math.radians(float(fn_cfg.get("endstop_angle_deg", 60.0))),
            linear_coef=float(fn_cfg.get("linear_coef", 0.1)),
            tan_coef=float(fn_cfg.get("tan_coef", 2.5e-2)),
        )
    )
    spring_cfg = cfg.get("spring", {})
    result = build_erc_design_from_function(
        FunctionTorqueProfileConfig(
            name=str(fn_cfg.get("name", "expression")),
            expression=fn_cfg.get("expression"),
            angle_min_rad=math.radians(float(fn_cfg.get("angle_min_deg", -85.0))),
            angle_max_rad=math.radians(float(fn_cfg.get("angle_max_deg", 85.0))),
            num_samples=int(fn_cfg.get("num_samples", 400)),
            amplitude=float(fn_cfg.get("amplitude", 1.0)),
            frequency=float(fn_cfg.get("frequency", 1.0)),
            phase=float(fn_cfg.get("phase", 0.0)),
            offset=float(fn_cfg.get("offset", 0.0)),
            torque_clip_nm=None
            if fn_cfg.get("torque_clip_nm") is None
            else float(fn_cfg.get("torque_clip_nm")),
            a=float(fn_cfg.get("a", 0.3)),
            b=float(fn_cfg.get("b", 50.0)),
            endstop_angle_rad=math.radians(float(fn_cfg.get("endstop_angle_deg", 60.0))),
            linear_coef=float(fn_cfg.get("linear_coef", 0.1)),
            tan_coef=float(fn_cfg.get("tan_coef", 2.5e-2)),
        ),
        spring_catalog_path=resolve_repo_path(spring_cfg.get("catalog_path", "configs/springs.yaml")),
        spring_id=str(spring_cfg.get("spring_id", "durovis_0.9x6.1x21")),
        n_grid=int(design_cfg.get("n_grid", 1000)),
        safety_factor=float(design_cfg.get("safety_factor", 1.1)),
        joint_angle_limits_rad=tuple(
            math.radians(float(value))
            for value in design_cfg.get("joint_angle_limits_deg", [-85.0, 85.0])
        ),
        energy_reference_angle_rad=math.radians(
            float(design_cfg.get("energy_reference_angle_deg", 0.0))
        ),
        max_torque_rmse_nm=None
        if validation_cfg.get("max_torque_rmse_nm") is None
        else float(validation_cfg.get("max_torque_rmse_nm")),
    )

    results_dir = resolve_repo_path(output_cfg.get("results_dir", "results/bistable_erc_example"))
    table_path = results_dir / str(output_cfg.get("table_csv", "bistable_repaired_torque_table.csv"))
    plot_path = results_dir / str(output_cfg.get("plot_png", "bistable_repaired_profile.png"))
    function_path = results_dir / str(output_cfg.get("function_py", "bistable_erc_torque_fn.py"))
    xyz_path = results_dir / str(output_cfg.get("curve_xyz", "bistable_repaired_cam.xyz"))
    sldcrv_path = results_dir / str(output_cfg.get("curve_sldcrv", "bistable_repaired_cam.sldcrv"))
    summary_path = results_dir / str(output_cfg.get("summary_yaml", "bistable_erc_summary.yaml"))
    isaac_load_path = results_dir / str(output_cfg.get("isaac_load_yaml", "bistable_isaac_load.yaml"))
    generated_upj_config_path = resolve_repo_path(
        get_simulation_cfg(cfg).get(
            "generated_config", "results/bistable_erc_example/upj_joint_response_config.yaml"
        )
    )

    write_table_csv(table_path, result.output_torque_table)
    write_function_file(
        function_path,
        torque_table_to_function_string(result.output_torque_table, fn_name="erc_torque_fn"),
    )
    result.export_xyz(xyz_path)
    result.export_sldcrv(sldcrv_path)
    save_plot(plot_path, result, profile)

    peak_passive_torque_nm = float(result.output_torque_table[:, 1].abs().max())
    applied_torque_scale = float(isaac_cfg.get("applied_torque_scale", 1.2))
    recommended_applied_torque_nm = applied_torque_scale * peak_passive_torque_nm

    write_yaml(
        summary_path,
        {
            "spring_id": str(spring_cfg.get("spring_id", "durovis_0.9x6.1x21")),
            "was_scaled": bool(result.was_scaled),
            "alpha": float(result.alpha),
            "was_repaired": bool(result.was_repaired),
            "torque_rmse_nm": float(result.torque_rmse_nm),
            "peak_passive_torque_nm": peak_passive_torque_nm,
            "recommended_applied_torque_nm": recommended_applied_torque_nm,
        },
    )
    write_yaml(
        isaac_load_path,
        {
            "isaac_joint_response_example": {
                "applied_torque_scale": applied_torque_scale,
                "peak_passive_torque_nm": peak_passive_torque_nm,
                "recommended_applied_torque_nm": recommended_applied_torque_nm,
            }
        },
    )
    build_upj_joint_response_config(
        cfg=cfg,
        result=result,
        recommended_applied_torque_nm=recommended_applied_torque_nm,
        generated_config_path=generated_upj_config_path,
        analytics_fps=args.analytics_fps,
    )

    if args.force_convert:
        ext_cfg = get_simulation_cfg(cfg)
        workspace_dir = resolve_repo_path(ext_cfg["workspace_dir"])
        for path in (
            workspace_dir / "concentrated_quadrotor_yaw_only.usd",
            workspace_dir / "airframe_data.json",
        ):
            if path.exists():
                path.unlink()

    if not args.prepare_only:
        ensure_upj_assets(
            cfg=cfg,
            generated_config_path=generated_upj_config_path,
            forward_args=forward_args,
        )
        run_upj_joint_response(
            cfg=cfg,
            generated_config_path=generated_upj_config_path,
            forward_args=forward_args,
        )

    print(f"Spring: {spring_cfg.get('spring_id', 'durovis_0.9x6.1x21')}")
    print(f"Scaled: {result.was_scaled} (alpha={result.alpha:.6g})")
    print(f"Repaired: {result.was_repaired}")
    print(f"Torque RMSE [Nm]: {result.torque_rmse_nm:.6g}")
    print(f"Peak passive torque [Nm]: {peak_passive_torque_nm:.6g}")
    print(f"Recommended applied torque [Nm]: {recommended_applied_torque_nm:.6g}")
    print(f"Torque table: {table_path}")
    print(f"Isaac function: {function_path}")
    print(f"Plot: {plot_path}")
    print(f"XYZ curve: {xyz_path}")
    print(f"SLDCRV curve: {sldcrv_path}")
    print(f"Summary: {summary_path}")
    print(f"Isaac load config: {isaac_load_path}")
    print(f"UPJ simulation config: {generated_upj_config_path}")


if __name__ == "__main__":
    main()
