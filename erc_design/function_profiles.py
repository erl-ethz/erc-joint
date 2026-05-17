from __future__ import annotations

import ast
import math
from dataclasses import dataclass

import torch

from .profiles import PiecewiseLinearTorqueProfile


@dataclass(frozen=True)
class FunctionTorqueProfileConfig:
    """Configuration for generating a torque profile from a function."""

    name: str = "expression"
    angle_min_rad: float = -1.0
    angle_max_rad: float = 1.0
    num_samples: int = 400
    amplitude: float = 1.0
    frequency: float = 1.0
    phase: float = 0.0
    offset: float = 0.0
    expression: str | None = None
    torque_clip_nm: float | None = None
    a: float = 0.3
    b: float = 50.0
    endstop_angle_rad: float = math.pi / 3.0
    linear_coef: float = 0.1
    tan_coef: float = 2.5e-2


def build_function_profile(
    config: FunctionTorqueProfileConfig,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> PiecewiseLinearTorqueProfile:
    """Discretize a function-defined torque law into a profile."""

    if config.num_samples < 2:
        raise ValueError("num_samples must be at least 2")
    if config.angle_min_rad >= config.angle_max_rad:
        raise ValueError("angle_min_rad must be smaller than angle_max_rad")

    angles = torch.linspace(
        config.angle_min_rad,
        config.angle_max_rad,
        config.num_samples,
        dtype=dtype,
        device=device,
    )
    torques = evaluate_torque_function(config, angles)
    return PiecewiseLinearTorqueProfile.from_xy(angles, torques, dtype=dtype, device=device)


def evaluate_torque_function(
    config: FunctionTorqueProfileConfig,
    angles_rad: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a supported torque function on a tensor of joint angles."""

    name = config.name.lower()
    amplitude = float(config.amplitude)
    frequency = float(config.frequency)
    phase = float(config.phase)
    offset = float(config.offset)

    if name == "sin":
        torques = amplitude * torch.sin(frequency * angles_rad + phase) + offset
    elif name == "cos":
        torques = amplitude * torch.cos(frequency * angles_rad + phase) + offset
    elif name == "tan":
        torques = amplitude * torch.tan(frequency * angles_rad + phase) + offset
    elif name == "expression":
        if not config.expression:
            raise ValueError("expression mode requires an expression string")
        torques = evaluate_expression(config.expression, angles_rad) + offset
    elif name in {"saturating_dual_stiffness", "sigmoid_tan"}:
        exp_arg = torch.clamp(-float(config.b) * angles_rad, -100.0, 100.0)
        torques = -float(config.a) * (
            1.0 / (1.0 + torch.exp(exp_arg))
            - 0.5
            + float(config.linear_coef) * angles_rad
            + float(config.tan_coef)
            * torch.tan(angles_rad / float(config.endstop_angle_rad) * torch.pi / 2.0)
        )
        torques = torques + offset
    else:
        raise ValueError(
            f"unsupported torque function {config.name!r}; "
            "use sin, cos, tan, expression, or saturating_dual_stiffness"
        )

    if config.torque_clip_nm is not None:
        clip_value = float(config.torque_clip_nm)
        torques = torch.clamp(torques, -clip_value, clip_value)
    return torques


def evaluate_expression(expression: str, theta: torch.Tensor) -> torch.Tensor:
    """Safely evaluate a small math expression over a torch tensor."""

    def as_tensor(value) -> torch.Tensor:
        if isinstance(value, (int, float)):
            return torch.tensor(value, device=theta.device, dtype=theta.dtype)
        raise ValueError(f"unsupported literal {value!r} in expression")

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

    def eval_node(node) -> torch.Tensor:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant):
            return as_tensor(node.value)
        if isinstance(node, ast.Num):
            return as_tensor(node.n)
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ValueError(f"unsupported name {node.id!r} in expression")
            return allowed_names[node.id]
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
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
            raise ValueError(f"unsupported binary operator {type(node.op).__name__}")
        if isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError(f"unsupported unary operator {type(node.op).__name__}")
        if isinstance(node, ast.Call):
            if node.keywords:
                raise ValueError("keyword arguments are not supported in expressions")
            if not isinstance(node.func, ast.Name):
                raise ValueError("only simple function names are supported in expressions")
            func_name = node.func.id
            if func_name not in allowed_funcs:
                raise ValueError(f"unsupported function {func_name!r} in expression")
            return allowed_funcs[func_name](*[eval_node(arg) for arg in node.args])
        raise ValueError(f"unsupported expression element {type(node).__name__}")

    return eval_node(ast.parse(expression, mode="eval"))
