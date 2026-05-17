from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Union

from erc_isaac.airframe_encoding import PARAM_NAMES


def decode_into_airframe(
    x: Union[List[float], Dict[str, float], "numpy.ndarray"],
    workspace_dir: Union[str, Path],
    torque_fn_left: str,
    torque_fn_right: str,
) -> Path:
    """Generate a Morphy USD asset in a local workspace."""
    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "convert_morphy_urdf.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--headless",
        "--output_dir",
        str(workspace_dir),
        "--torque_fn_left",
        torque_fn_left,
        "--torque_fn_right",
        torque_fn_right,
    ]

    if isinstance(x, dict):
        for key, value in x.items():
            cmd.extend([f"--{key}", str(float(value))])
        params_to_save = x
    else:
        if len(x) != len(PARAM_NAMES):
            raise ValueError(f"Expected {len(PARAM_NAMES)} parameters, got {len(x)}")
        cmd.extend(["--params_list", *[str(float(value)) for value in x]])
        params_to_save = {"x_0_1": [float(value) for value in x]}

    print(f"[decode_into_airframe] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"USD generation failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

    workspace_usd_path = workspace_dir / "morphy_prog.usd"
    if not workspace_usd_path.exists():
        raise RuntimeError(f"USD file not generated at expected location: {workspace_usd_path}")

    params_file = workspace_dir / "params.json"
    with params_file.open("w", encoding="utf-8") as handle:
        json.dump(params_to_save, handle, indent=2)

    return workspace_usd_path
