#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to convert a yaw-only concentrated quadrotor URDF to USD format.

Usage:
  # 1) Use defaults from YAML config
  python scripts/convert_concentrated_urdf_yaw_only.py --headless \
      --output_dir cache/workspace_local/my_airframe_yaw_only \
      --config configs/joint_response_template.yaml

  # 2) Override any parameter on the CLI
  python scripts/convert_concentrated_urdf_yaw_only.py --headless \
      --output_dir cache/workspace_local/my_airframe_yaw_only \
      --arm_proximal_length 0.02 --arm_distal_length 0.18 \
      --base_mass 0.3 --base_inertia 0.0005 0.0005 0.001
"""

PARAM_NAMES = [
    "arm_proximal_length",
    "arm_distal_length",
    "base_mass",
    "base_inertia",
]

PARAM_BOUNDS = {
    "arm_proximal_length": (0.01, 0.15),
    "arm_distal_length": (0.05, 0.2),
    "base_mass": (0.1, 0.5),
    "base_inertia": (0.0001, 0.01),
}


def validate_params(params: dict):
    """Validate that parameters are within their bounds."""
    for name, value in params.items():
        if name in PARAM_BOUNDS:
            min_val, max_val = PARAM_BOUNDS[name]
            if isinstance(value, (list, tuple)):
                for v in value:
                    if not isinstance(v, (int, float)):
                        raise ValueError(f"Parameter '{name}' must be a scalar value.")
                    if not (min_val <= v <= max_val):
                        raise ValueError(f"Parameter '{name}' = {v} out of bounds [{min_val}, {max_val}]")
            else:
                if not (min_val <= value <= max_val):
                    raise ValueError(f"Parameter '{name}' = {value} out of bounds [{min_val}, {max_val}]")


def generate_airframe_json(params: dict, output_json_path: str):
    """Generate airframe properties JSON from parameter dict by creating USD and extracting data.

    Args:
        params: Dictionary of parameter values
        output_json_path: Path where to save the JSON file
    """
    import json
    import os
    import tempfile
    import shutil
    import subprocess
    import sys

    temp_workspace = tempfile.mkdtemp(prefix="airframe_validation_")

    try:
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "convert_concentrated_urdf_yaw_only.py"),
            "--headless",
            "--output_dir", temp_workspace,
        ]

        for name in PARAM_NAMES:
            cmd.extend([f"--{name}", str(params[name])])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            raise RuntimeError(f"USD generation failed: {result.stderr}")

        temp_json = os.path.join(temp_workspace, "airframe_data.json")
        if not os.path.exists(temp_json):
            raise RuntimeError(
                f"airframe_data.json not found in {temp_workspace}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        shutil.copy(temp_json, output_json_path)

        with open(output_json_path, "r") as f:
            airframe_data = json.load(f)

        return airframe_data
    finally:
        shutil.rmtree(temp_workspace, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    from isaaclab.app import AppLauncher
    import yaml
    import os

    parser = argparse.ArgumentParser(
        description="Convert yaw-only concentrated quadrotor URDF to USD with parametric geometry."
    )
    parser.add_argument("--config", type=str, default="configs/joint_response_template.yaml")
    parser.add_argument("--arm_proximal_length", type=float, help="First segment arm length before the joint (m)")
    parser.add_argument("--arm_distal_length", type=float, help="Last segment arm length after the joint (m)")
    parser.add_argument("--base_mass", type=float, help="Base mass (kg)")
    parser.add_argument(
        "--base_inertia",
        type=float,
        nargs=3,
        help="Base inertia tensor diagonal values (Ixx Iyy Izz) in kg*m^2",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for USD files")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    config_path = args_cli.config
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path))
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    airframe_params = (cfg.get("airframe_generation") or {}).copy()
    params = {
        "arm_proximal_length": args_cli.arm_proximal_length
        if args_cli.arm_proximal_length is not None
        else airframe_params.get("arm_proximal_length"),
        "arm_distal_length": args_cli.arm_distal_length
        if args_cli.arm_distal_length is not None
        else airframe_params.get("arm_distal_length"),
        "base_mass": args_cli.base_mass if args_cli.base_mass is not None else airframe_params.get("base_mass"),
        "base_inertia": args_cli.base_inertia
        if args_cli.base_inertia is not None
        else airframe_params.get("base_inertia"),
    }
    missing = [k for k, v in params.items() if v is None]
    if missing:
        raise ValueError(
            f"Missing required parameters: {missing}. Provide them in {config_path} under airframe_generation "
            "or pass on the command line."
        )
    validate_params(params)

    print(f"[Convert URDF to USD] Parameters: {params}")

    app_launcher = AppLauncher(args_cli, multi_gpu=False)
    simulation_app = app_launcher.app

    import os
    import math
    from pxr import UsdPhysics
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg, spawn_rigid_body_material
    import isaacsim.core.utils.prims as prim_utils

    def normalize_output_dir(path: str) -> str:
        """Normalize legacy myairframe folder name to my_airframe."""
        norm = os.path.normpath(path)
        parts = [("my_airframe" if p == "myairframe" else p) for p in norm.split(os.sep)]
        return os.sep.join(parts)

    def generate_parametric_urdf(output_path, params):
        """Generate a yaw-only parametric URDF based on input parameters."""
        import numpy as np

        proximal_arm_len = params["arm_proximal_length"]
        distal_arm_len = params["arm_distal_length"]
        base_mass = params["base_mass"]
        base_inertia = params["base_inertia"]

        arm_width = 0.01
        arm_height = 0.01
        arm_density = 2000.0
        motor_mass = 0.016249999999999999
        motor_radius = 0.045
        motor_height = 0.035
        base_box_dimensions = [0.07, 0.05, 0.08]

        proximal_arm_mass = arm_density * proximal_arm_len * arm_width * arm_height
        distal_arm_mass = arm_density * distal_arm_len * arm_width * arm_height
        proximal_arm_com = proximal_arm_len / 2
        distal_arm_com = distal_arm_len / 2

        arms = [
            {"idx": 0, "origin": (-base_box_dimensions[0] / 2, -base_box_dimensions[1] / 2, 0), "yaw": np.pi * 5 / 4},
            {"idx": 1, "origin": (base_box_dimensions[0] / 2, -base_box_dimensions[1] / 2, 0), "yaw": np.pi * 7 / 4},
            {"idx": 2, "origin": (-base_box_dimensions[0] / 2, base_box_dimensions[1] / 2, 0), "yaw": np.pi * 3 / 4},
            {"idx": 3, "origin": (base_box_dimensions[0] / 2, base_box_dimensions[1] / 2, 0), "yaw": np.pi / 4},
        ]

        lines = [
            '<?xml version="1.0"?>',
            '<robot name="quadrotor">',
            '  <link name="base_link">',
            '    <visual>',
            '      <origin xyz="0 0 0" rpy="0 0 0"/>',
            "      <geometry>",
            f'        <box size="{base_box_dimensions[0]} {base_box_dimensions[1]} {base_box_dimensions[2]}"/>',
            "      </geometry>",
            '      <material name="White">',
            '        <color rgba="1 1 1 1.0"/>',
            "      </material>",
            "    </visual>",
            "    <inertial>",
            f'      <mass value="{base_mass}"/>',
            f'      <inertia ixx="{base_inertia[0]}" ixy="0.0" ixz="0.0" iyx="0.0" iyy="{base_inertia[1]}" iyz="0.0" izx="0.0" izy="0.0" izz="{base_inertia[2]}"/>',
            "    </inertial>",
            "  </link>",
        ]

        for arm in arms:
            i = arm["idx"]
            origin = arm["origin"]
            yaw = arm["yaw"]

            lines += [
                f"",
                f"  <!-- Arm {i} (idx {i}) -->",
                f'  <link name="arm_segment_1_{i}">',
                "    <visual>",
                f'      <origin xyz="{proximal_arm_com} 0 0" rpy="0 0 0"/>',
                "      <geometry>",
                f'        <box size="{proximal_arm_len} {arm_width} {arm_height}"/>',
                "      </geometry>",
                '      <material name="Orange">',
                '        <color rgba="1 0.5 0 1.0"/>',
                "      </material>",
                "    </visual>",
                "    <inertial>",
                f'      <origin xyz="{proximal_arm_com} 0 0" rpy="0 0 0"/>',
                f'      <mass value="{proximal_arm_mass}"/>',
                '      <inertia ixx="0.00002" ixy="0.0" ixz="0.0" iyy="0.00002" iyz="0.0" izz="0.00002"/>',
                "    </inertial>",
                "  </link>",
                f'  <joint name="base_link_to_arm_segment_1_{i}" type="fixed" dont_collapse="true">',
                "    <parent link=\"base_link\"/>",
                f'    <child link="arm_segment_1_{i}"/>',
                f'    <origin xyz="{origin[0]} {origin[1]} {origin[2]}" rpy="0 0 {yaw}"/>',
                "  </joint>",
                f'  <link name="dummy_link_yaw_{i}">',
                "    <visual>",
                '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                "      <geometry>",
                '        <sphere radius="1e-3"/>',
                "      </geometry>",
                '      <material name="Black">',
                '        <color rgba="0 0 0 1.0"/>',
                "      </material>",
                "    </visual>",
                "    <inertial>",
                '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                '      <mass value="1e-9"/>',
                '      <inertia ixx="1e-12" ixy="0.0" ixz="0.0" iyy="1e-12" iyz="0.0" izz="1e-12"/>',
                "    </inertial>",
                "  </link>",
                f'  <joint name="arm_segment_1_{i}_to_dummy_link_yaw_{i}" type="revolute" dont_collapse="false">',
                f'    <parent link="arm_segment_1_{i}"/>',
                f'    <child link="dummy_link_yaw_{i}"/>',
                f'    <origin xyz="{proximal_arm_len} 0 0" rpy="0.0 0.0 0.0"/>',
                '    <axis xyz="0 0 1"/>',
                '    <limit lower="-1.57" upper="1.57" effort="20.0" velocity="10.0"/>',
                "  </joint>",
                f'  <link name="arm_segment_2_{i}">',
                "    <visual>",
                f'      <origin xyz="{distal_arm_com} 0 0" rpy="0 0 0"/>',
                "      <geometry>",
                f'        <box size="{distal_arm_len} {arm_width} {arm_height}"/>',
                "      </geometry>",
                '      <material name="Orange">',
                '        <color rgba="1 0.5 0 1.0"/>',
                "      </material>",
                "    </visual>",
                "    <inertial>",
                f'      <origin xyz="{distal_arm_com} 0 0" rpy="0 0 0"/>',
                f'      <mass value="{distal_arm_mass}"/>',
                '      <inertia ixx="0.00002" ixy="0.0" ixz="0.0" iyy="0.00002" iyz="0.0" izz="0.00002"/>',
                "    </inertial>",
                "  </link>",
                f'  <joint name="dummy_link_yaw_{i}_to_arm_segment_2_{i}" type="fixed" dont_collapse="true">',
                f'    <parent link="dummy_link_yaw_{i}"/>',
                f'    <child link="arm_segment_2_{i}"/>',
                '    <origin xyz="0 0 0" rpy="0 0 0"/>',
                "  </joint>",
                f'  <link name="motor_{i}">',
                "    <visual>",
                f'      <origin xyz="0 0 {motor_height/2}" rpy="0 0 0"/>',
                "      <geometry>",
                f'        <cylinder radius="{motor_radius}" length="{motor_height}"/>',
                "      </geometry>",
                '      <material name="PastelBlue">',
                '        <color rgba="0.68 0.85 0.9 1.0"/>',
                "      </material>",
                "    </visual>",
                "    <inertial>",
                '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                f'      <mass value="{motor_mass}"/>',
                '      <inertia ixx="1.0e-08" ixy="0.0" ixz="0.0" iyy="1.0e-08" iyz="0.0" izz="1.0e-08"/>',
                "    </inertial>",
                "  </link>",
                f'  <joint name="arm_segment_2_{i}_to_motor_{i}" type="fixed" dont_collapse="true">',
                f'    <parent link="arm_segment_2_{i}"/>',
                f'    <child link="motor_{i}"/>',
                f'    <origin xyz="{distal_arm_len} 0 0" rpy="0 0 0"/>',
                "  </joint>",
            ]

        lines.append("</robot>")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        print("Generated yaw-only parametric URDF with:")
        print(f"  Proximal arm segment length: {proximal_arm_com} m")
        print(f"  Distal arm segment length: {distal_arm_com} m")

    def extract_airframe_data_from_usd(usd_path: str) -> dict:
        """Extract airframe physical properties directly from USD file."""
        import numpy as np
        from isaacsim.core.utils.stage import open_stage, get_current_stage
        from pxr import UsdPhysics, UsdGeom
        import traceback

        try:
            open_stage(usd_path)
            stage = get_current_stage()

            root_prim = stage.GetPrimAtPath("/quadrotor")
            if not root_prim.IsValid():
                raise RuntimeError(f"Root prim '/quadrotor' not found in {usd_path}")
        except Exception as e:
            print(f"[ERROR] Failed to extract USD data: {e}")
            traceback.print_exc()
            raise

        total_mass = 0.0
        motor_info = []

        base_link_prim = stage.GetPrimAtPath("/quadrotor/base_link")
        if not base_link_prim.IsValid():
            raise RuntimeError(f"base_link not found in {usd_path}")

        base_link_xform = UsdGeom.Xformable(base_link_prim)
        base_link_world_transform = base_link_xform.ComputeLocalToWorldTransform(0)
        base_link_world_transform_inv = base_link_world_transform.GetInverse()

        def traverse_prims(prim):
            nonlocal total_mass, motor_info

            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                mass_api = UsdPhysics.MassAPI(prim)
                if mass_api:
                    mass = mass_api.GetMassAttr().Get()
                    if mass is not None:
                        total_mass += mass

            prim_name = prim.GetName()
            if prim_name.startswith("motor_"):
                motor_idx = int(prim_name.split("_")[1])
                xform = UsdGeom.Xformable(prim)
                motor_world_transform = xform.ComputeLocalToWorldTransform(0)
                motor_body_transform = base_link_world_transform_inv * motor_world_transform
                translation = motor_body_transform.ExtractTranslation()
                rotation = motor_body_transform.ExtractRotationMatrix()
                thrust_direction_vec = rotation.GetRow(2)
                thrust_direction = [
                    float(thrust_direction_vec[0]),
                    float(thrust_direction_vec[1]),
                    float(thrust_direction_vec[2]),
                ]
                motor_info.append(
                    (motor_idx, [float(translation[0]), float(translation[1]), float(translation[2])], thrust_direction)
                )

            for child in prim.GetAllChildren():
                traverse_prims(child)

        traverse_prims(root_prim)

        motor_info.sort(key=lambda x: x[0])
        motor_positions = [pos for _, pos, _ in motor_info]
        motor_thrust_directions = [thrust_dir for _, _, thrust_dir in motor_info]
        motor_directions = [1, -1, -1, 1]

        return {
            "total_mass": total_mass,
            "num_motors": len(motor_positions),
            "motor_positions_body_frame": motor_positions,
            "motor_thrust_directions": motor_thrust_directions,
            "motor_directions": motor_directions,
        }

    def save_airframe_data(output_dir: str, usd_path: str, params: dict):
        """Save airframe physical properties to JSON by reading from USD."""
        import json

        airframe_data = extract_airframe_data_from_usd(usd_path)
        airframe_data["motor_idx_list"] = [5, 5, 5, 5]
        airframe_data["parameters"] = params

        json_path = os.path.join(output_dir, "airframe_data.json")
        with open(json_path, "w") as f:
            json.dump(airframe_data, f, indent=2)

        print(f"Saved airframe data to {json_path} (extracted from USD):")
        print(f"  Total mass: {airframe_data['total_mass']:.4f} kg")
        print(f"  Number of motors: {airframe_data['num_motors']}")
        print("  Motor positions (body frame):")
        for i, pos in enumerate(airframe_data["motor_positions_body_frame"]):
            print(f"    Motor {i}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    def main():
        """Convert the yaw-only concentrated quadrotor URDF to USD."""
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        urdf_path = os.path.join(script_dir, "src/concentrated_quadrotor_yaw_only.urdf")

        if args_cli.output_dir:
            output_dir = os.path.abspath(args_cli.output_dir)
        else:
            output_dir = os.path.join(script_dir, "cache/workspace_local/usd")
        normalized_output_dir = normalize_output_dir(output_dir)
        if normalized_output_dir != output_dir:
            print(f"[Info] Normalized output_dir from {output_dir} to {normalized_output_dir}")
        output_dir = normalized_output_dir

        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print("Converting Yaw-Only Concentrated Quadrotor")
        print("  - Base: Box (7x5x8 cm)")
        print("  - Arms: Parametric length with yaw-only joint")
        print("  - Motors: Cylinders (radius=4.5cm)")
        print(f"{'='*80}\n")

        generate_parametric_urdf(urdf_path, params)

        urdf_converter_cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=output_dir,
            usd_file_name="concentrated_quadrotor_yaw_only.usd",
            force_usd_conversion=True,
            make_instanceable=False,
            fix_base=False,
            merge_fixed_joints=False,
            self_collision=False,
            collider_type="convex_hull",
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                ),
            ),
        )

        urdf_converter = UrdfConverter(urdf_converter_cfg)
        output_path = urdf_converter.usd_path
        print("Conversion complete!")
        print(f"USD file saved to: {output_path}")

        save_airframe_data(output_dir, output_path, params)
        print(f"\n{'='*80}\n")

    main()
    simulation_app.close()
