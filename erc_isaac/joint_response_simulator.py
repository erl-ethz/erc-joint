"""Programmable joint response utilities.

Usage:
  # Print a generated response table (uses config defaults)
  python erc_isaac/joint_response_simulator.py --config configs/joint_response_template.yaml

  # Print full table (no truncation)
  python erc_isaac/joint_response_simulator.py --config configs/joint_response_template.yaml --print_limit 0

  # Plot response curve and save to results/ (see output.joint_response_plot_name)
  python erc_isaac/joint_response_simulator.py --config configs/joint_response_template.yaml --plot
"""

from __future__ import annotations

from pathlib import Path
import argparse
import ast
import math
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt


def _to_tensor_points(points, device: str | None = None) -> torch.Tensor:
    if isinstance(points, torch.Tensor):
        pts = points.clone().detach().float()
        return pts.to(device=device) if device is not None else pts
    return torch.tensor(points, dtype=torch.float32, device=device)


def dense_from_points(points, num_samples: int = 1000, device: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a dense, piecewise-linear table from angle/torque points.

    Args:
        points: Iterable or tensor of [angle, torque] rows (angles in radians)
        num_samples: Approximate total samples to generate
        device: torch device

    Returns:
        angles_dense, torques_dense (1D tensors)
    """
    pts = _to_tensor_points(points, device=device)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be a list/tensor of shape (N, 2) -> [angle, torque]")
    if pts.shape[0] < 2:
        raise ValueError("points must contain at least two rows")

    sorted_pts = pts[torch.argsort(pts[:, 0])]
    segments = int(sorted_pts.shape[0] - 1)
    steps_per_segment = max(2, int(num_samples / segments))

    x_dense_list = []
    y_dense_list = []

    for i in range(segments):
        x_i, y_i = sorted_pts[i, 0], sorted_pts[i, 1]
        x_j, y_j = sorted_pts[i + 1, 0], sorted_pts[i + 1, 1]

        x_segment = torch.linspace(x_i, x_j, steps_per_segment, device=sorted_pts.device)
        if x_j != x_i:
            y_segment = y_i + (y_j - y_i) * (x_segment - x_i) / (x_j - x_i)
        else:
            y_segment = torch.full((steps_per_segment,), y_i, dtype=torch.float32, device=sorted_pts.device)

        x_dense_list.append(x_segment)
        y_dense_list.append(y_segment)

    x_dense = torch.cat(x_dense_list, dim=0)
    y_dense = torch.cat(y_dense_list, dim=0)
    return x_dense, y_dense


def _eval_expression(expr: str, theta: torch.Tensor) -> torch.Tensor:
    """Safely evaluate a math expression over a torch tensor.

    Allowed ops: +, -, *, /, ** and a small whitelist of torch functions.
    Variables: theta, pi, e
    """

    def _as_tensor(value) -> torch.Tensor:
        if isinstance(value, (int, float)):
            return torch.tensor(value, device=theta.device, dtype=theta.dtype)
        raise ValueError(f"Unsupported literal '{value}' in expression.")

    allowed_funcs = {
        "sin": torch.sin,
        "cos": torch.cos,
        "tan": torch.tan,
        "asin": torch.asin,
        "acos": torch.acos,
        "atan": torch.atan,
        "sinh": torch.sinh,
        "cosh": torch.cosh,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "exp": torch.exp,
        "log": torch.log,
        "log10": torch.log10,
        "sqrt": torch.sqrt,
        "abs": torch.abs,
        "sign": torch.sign,
        "pow": torch.pow,
        "minimum": torch.minimum,
        "maximum": torch.maximum,
        "relu": torch.relu,
        "clamp": torch.clamp,
        "floor": torch.floor,
        "ceil": torch.ceil,
    }
    allowed_names = {
        "theta": theta,
        "pi": torch.tensor(math.pi, device=theta.device, dtype=theta.dtype),
        "e": torch.tensor(math.e, device=theta.device, dtype=theta.dtype),
    }

    def _eval(node) -> torch.Tensor:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return _as_tensor(node.value)
        if isinstance(node, ast.Num):  # pragma: no cover (py<3.8)
            return _as_tensor(node.n)
        if isinstance(node, ast.Name):
            if node.id in allowed_names:
                return allowed_names[node.id]
            raise ValueError(f"Unsupported name '{node.id}' in expression.")
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError(f"Unsupported binary operator '{type(node.op).__name__}'.")
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError(f"Unsupported unary operator '{type(node.op).__name__}'.")
        if isinstance(node, ast.Call):
            if node.keywords:
                raise ValueError("Keyword arguments are not supported in expression functions.")
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function names are supported in expressions.")
            func_name = node.func.id
            if func_name not in allowed_funcs:
                raise ValueError(f"Unsupported function '{func_name}' in expression.")
            args = [_eval(arg) for arg in node.args]
            return allowed_funcs[func_name](*args)
        raise ValueError(f"Unsupported expression element '{type(node).__name__}'.")

    parsed = ast.parse(expr, mode="eval")
    return _eval(parsed)


def _function_response(func_cfg: dict, num_samples: int, device: str | None) -> tuple[torch.Tensor, torch.Tensor]:
    name = str(func_cfg.get("name", "sin")).lower()
    angle_min = float(func_cfg.get("angle_min", -1.0))
    angle_max = float(func_cfg.get("angle_max", 1.0))
    angles = torch.linspace(angle_min, angle_max, num_samples, device=device)

    amplitude = float(func_cfg.get("amplitude", 1.0))
    frequency = float(func_cfg.get("frequency", 1.0))
    phase = float(func_cfg.get("phase", 0.0))
    offset = float(func_cfg.get("offset", 0.0))

    if name == "sin":
        base = torch.sin(frequency * angles + phase)
        torques = amplitude * base + offset
    elif name == "cos":
        base = torch.cos(frequency * angles + phase)
        torques = amplitude * base + offset
    elif name == "tan":
        base = torch.tan(frequency * angles + phase)
        torques = amplitude * base + offset
    elif name == "expression":
        expr = func_cfg.get("expression", None)
        if not expr:
            raise ValueError("expression mode requires joint_response.function.expression")
        torques = _eval_expression(expr, angles)
        torques = torques + offset
    elif name in ("saturating_dual_stiffness", "sigmoid_tan"):
        a = float(func_cfg.get("a", 0.3))
        b = float(func_cfg.get("b", 50.0))
        endstop_angle = float(func_cfg.get("endstop_angle", math.pi / 3))
        linear_coef = float(func_cfg.get("linear_coef", 0.1))
        tan_coef = float(func_cfg.get("tan_coef", 2.5e-2))
        exp_arg = torch.clamp(-b * angles, -100, 100)
        torques = -a * (
            1.0 / (1.0 + torch.exp(exp_arg))
            - 0.5
            + linear_coef * angles
            + tan_coef * torch.tan(angles / endstop_angle * torch.pi / 2.0)
        )
        torques = torques + offset
    else:
        raise ValueError(
            f"Unsupported function name '{name}'. Use sin, cos, tan, expression, or saturating_dual_stiffness."
        )

    torque_clip = func_cfg.get("torque_clip", None)
    if torque_clip is not None:
        clip_val = float(torque_clip)
        torques = torch.clamp(torques, -clip_val, clip_val)

    return angles, torques


def build_response_table(response_cfg: dict, device: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dense joint response table from config.

    Args:
        response_cfg: Config dict with mode and parameters
        device: torch device
    """
    mode = str(response_cfg.get("mode", "linear")).lower()
    num_samples = int(response_cfg.get("num_samples", 1000))

    if mode == "piecewise":
        piecewise_cfg = response_cfg.get("piecewise", response_cfg)
        points = piecewise_cfg.get("points", None)
        if points is None:
            raise ValueError("piecewise mode requires joint_response.piecewise.points")
        return dense_from_points(points, num_samples=num_samples, device=device)

    if mode == "function":
        func_cfg = response_cfg.get("function", response_cfg)
        return _function_response(func_cfg, num_samples=num_samples, device=device)

    raise ValueError(f"Unsupported joint_response.mode '{mode}' (use piecewise or function).")


def interpolate_torque(query_angles: torch.Tensor, table_angles: torch.Tensor, table_torques: torch.Tensor) -> torch.Tensor:
    """Linear interpolation from a dense lookup table.

    Args:
        query_angles: Tensor of angles (any shape)
        table_angles: 1D tensor of sorted angles
        table_torques: 1D tensor of torques
    """
    if table_angles.ndim != 1 or table_torques.ndim != 1:
        raise ValueError("table_angles and table_torques must be 1D tensors")
    if table_angles.shape[0] != table_torques.shape[0]:
        raise ValueError("table_angles and table_torques must have the same length")

    q = query_angles
    q_flat = q.reshape(-1)
    idx = torch.bucketize(q_flat, table_angles)
    idx0 = torch.clamp(idx - 1, 0, table_angles.shape[0] - 1)
    idx1 = torch.clamp(idx, 0, table_angles.shape[0] - 1)

    x0 = table_angles[idx0]
    x1 = table_angles[idx1]
    y0 = table_torques[idx0]
    y1 = table_torques[idx1]

    denom = x1 - x0
    w = torch.where(denom != 0, (q_flat - x0) / denom, torch.zeros_like(denom))
    y = y0 + w * (y1 - y0)
    return y.reshape(q.shape)


def plot_interpolated_response_from_table(
    points,
    response_descriptor: str = "",
    num_samples: int = 1000,
    save_plot_file: bool = True,
    show_plot: bool = True,
    output_path: str | Path | None = None,
):
    """Plot interpolated response curve from angle/torque points."""
    if isinstance(points, torch.Tensor):
        points_t = points.clone().detach().float()
    else:
        points_t = torch.tensor(points, dtype=torch.float32)

    sorted_points = points_t[torch.argsort(points_t[:, 0])]
    x_dense, y_dense = dense_from_points(sorted_points, num_samples=num_samples, device=None)

    angles = sorted_points[:, 0].tolist()
    efforts = sorted_points[:, 1].tolist()

    plt.figure(figsize=(12, 8))
    plt.plot(180 / np.pi * np.array(angles), efforts, "b--", linewidth=2.5, label="User-defined torque response", alpha=0.8)
    plt.plot(180 / np.pi * x_dense.numpy(), y_dense.numpy(), "ro", markersize=4, label="Dense look-up points", zorder=5)

    plt.grid(True, alpha=0.3)
    plt.xlabel("Joint angle (degrees)", fontsize=16)
    plt.ylabel("Joint torque (Nm)", fontsize=16)
    plt.title("Programmable Joints Response Curve", fontsize=18)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params(axis="both", which="major", labelsize=16)

    plt.figtext(
        0.12,
        0.18,
        f"Input type: {response_descriptor if response_descriptor else 'table'}\n"
        f"Interpolation: {len(x_dense) + 2} samples\n"
        f"Angle range: [{180/np.pi*min(angles):.1f}, {180/np.pi*max(angles):.1f}] deg\n"
        f"Effort range: [{min(efforts):.3f}, {max(efforts):.3f}] Nm",
        fontsize=16,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
    )

    plt.tight_layout()
    if save_plot_file:
        if output_path is None:
            filename = f"programmable_joint_response_curve_{response_descriptor.replace(' ', '_')}.png"
            output_path = filename
        plt.savefig(str(output_path), dpi=300)
    if show_plot:
        plt.show()
    return plt.gcf()


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def _resolve_output_path(cfg: dict, default_name: str) -> Path:
    output_cfg = cfg.get("output", {})
    results_dir = output_cfg.get("results_dir", "results")
    plot_name = output_cfg.get("joint_response_plot_name", default_name)
    base = Path(results_dir) / plot_name
    if base.is_absolute():
        base.parent.mkdir(parents=True, exist_ok=True)
        return base
    resolved = (Path(__file__).parent.parent / base).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def main():
    parser = argparse.ArgumentParser(description="Generate dense joint response table from config.")
    parser.add_argument("--config", type=str, default="configs/joint_response_template.yaml")
    parser.add_argument("--print_limit", type=int, default=20, help="Max lines to print (0 prints all).")
    parser.add_argument("--plot", action="store_true", help="Plot the response curve.")
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = str((Path(__file__).parent.parent / config_path).resolve())
    cfg = _load_yaml(config_path)
    response_cfg = cfg.get("joint_response", {})
    mode = str(response_cfg.get("mode", "linear")).lower()
    if mode not in ("piecewise", "function"):
        raise ValueError("joint_response.mode must be 'piecewise' or 'function' to generate a table.")

    angles, torques = build_response_table(response_cfg, device=None)
    print(f"[joint_response] mode={mode} samples={angles.shape[0]}")

    limit = int(args.print_limit)
    total = angles.shape[0]
    if limit == 0:
        limit = total
    for i in range(min(limit, total)):
        print(f"{angles[i].item(): .6f}, {torques[i].item(): .6f}")
    if limit < total:
        print(f"... ({total - limit} more rows)")

    if args.plot and mode == "piecewise":
        piecewise_cfg = response_cfg.get("piecewise", response_cfg)
        plot_path = _resolve_output_path(cfg, "programmable_joint_response_curve_piecewise.png")
        plot_interpolated_response_from_table(
            piecewise_cfg.get("points", []),
            response_descriptor="piecewise",
            output_path=plot_path,
        )
        print(f"[joint_response] Plot saved to {plot_path}")
    elif args.plot and mode == "function":
        pts = torch.stack([angles, torques], dim=1)
        plot_path = _resolve_output_path(cfg, "programmable_joint_response_curve_function.png")
        plot_interpolated_response_from_table(
            pts,
            response_descriptor=response_cfg.get("function", {}).get("name", "function"),
            output_path=plot_path,
        )
        print(f"[joint_response] Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
