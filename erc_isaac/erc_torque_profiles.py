from __future__ import annotations
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence
import torch
from erc_design import ERCDesignConfig, ERCDesigner, PiecewiseLinearTorqueProfile, SpringCatalog
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPRING_CATALOG = PROJECT_ROOT / 'configs' / 'springs.yaml'
DEFAULT_SPRING_ID = 'durovis_0.9x6.1x21'
FORMULA_TORQUE_A_BOUNDS_NM = (0.001, 0.4)
FORMULA_TORQUE_C_BOUNDS_NM = (0.0, 0.4)
FORMULA_TORQUE_D_BOUNDS_NM = (0.0, 0.399)
FORMULA_TORQUE_B_BOUNDS = (0.01, 0.99)

@dataclass(frozen=True)
class ERCTorqueFunctionBuild:
    torque_function: object
    function_string: str
    design_result: object
    requested_knots_deg_nm: tuple[tuple[float, float], ...]
    actual_parameters: tuple[float, ...]
    angle_offset_rad: float

@dataclass(frozen=True)
class FormulaTorqueParameters:
    a: float
    b: float
    c: float
    d: float
    end_angle_rad: float

def torque_table_to_function_string(table: torch.Tensor, fn_name: str, extrapolation: str, flat_boundary_torques_nm: tuple[float, float] | None) -> str:
    """Serialize a torque table as a torch-callable function string."""
    if fn_name not in {'torque_fn_left', 'torque_fn_right'}:
        raise ValueError(f'Unexpected torque function name: {fn_name}')
    if extrapolation not in {'raise', 'flat'}:
        raise ValueError(f'Unexpected ERC torque extrapolation mode: {extrapolation}')
    table_cpu = table.detach().cpu()
    angles = [float(f'{float(v):.17g}') for v in table_cpu[:, 0]]
    torques = [float(f'{float(v):.17g}') for v in table_cpu[:, 1]]
    if extrapolation == 'flat' and flat_boundary_torques_nm is None:
        raise ValueError("flat_boundary_torques_nm is required when extrapolation is 'flat'")
    if extrapolation == 'raise' and flat_boundary_torques_nm is not None:
        raise ValueError("flat_boundary_torques_nm must be None when extrapolation is 'raise'")
    out_of_domain_logic = f'raise ValueError("{fn_name} received an angle outside its ERC table domain")' if extrapolation == 'raise' else 'return torch.where(x < xp[0], _erc_flat_lower + x * 0.0, torch.where(x > xp[-1], _erc_flat_upper + x * 0.0, _erc_interp(x, xp, fp)))'
    in_domain_logic = '_erc_interp(x, xp, fp)' if extrapolation == 'raise' else 'torch.where(x < xp[0], _erc_flat_lower + x * 0.0, torch.where(x > xp[-1], _erc_flat_upper + x * 0.0, _erc_interp(x, xp, fp)))'
    flat_lower = None if flat_boundary_torques_nm is None else float(flat_boundary_torques_nm[0])
    flat_upper = None if flat_boundary_torques_nm is None else float(flat_boundary_torques_nm[1])
    return f'\n_erc_angles = {angles!r}\n_erc_torques = {torques!r}\n_erc_flat_lower = {flat_lower!r}\n_erc_flat_upper = {flat_upper!r}\n_erc_cache = {{}}\ndef _erc_interp(x, xp, fp):\n    idx = torch.searchsorted(xp, x, right=True) - 1\n    idx = torch.clamp(idx, 0, xp.numel() - 2)\n    x0 = xp[idx]\n    x1 = xp[idx + 1]\n    y0 = fp[idx]\n    y1 = fp[idx + 1]\n    return y0 + (x - x0) / (x1 - x0) * (y1 - y0)\ndef {fn_name}(x):\n    key = (str(x.device), x.dtype)\n    if key not in _erc_cache:\n        _erc_cache[key] = (\n            torch.tensor(_erc_angles, device=x.device, dtype=x.dtype),\n            torch.tensor(_erc_torques, device=x.device, dtype=x.dtype),\n        )\n    xp, fp = _erc_cache[key]\n    if torch.any(x < xp[0]) or torch.any(x > xp[-1]):\n        {out_of_domain_logic}\n    return {in_domain_logic}\n{fn_name}.erc_angle_min_rad = _erc_angles[0]\n{fn_name}.erc_angle_max_rad = _erc_angles[-1]\n'

def formula_torque_profile_knots_rad_nm(parameters: FormulaTorqueParameters, mirror: bool) -> tuple[tuple[float, float], ...]:
    a = float(parameters.a)
    b = float(parameters.b)
    c = float(parameters.c)
    d = float(parameters.d)
    end_angle = float(parameters.end_angle_rad)
    if a < FORMULA_TORQUE_A_BOUNDS_NM[0] or a > FORMULA_TORQUE_A_BOUNDS_NM[1]:
        raise ValueError(f'a must be in {FORMULA_TORQUE_A_BOUNDS_NM}, got {a}')
    if b < FORMULA_TORQUE_B_BOUNDS[0] or b > FORMULA_TORQUE_B_BOUNDS[1]:
        raise ValueError(f'b must be in {FORMULA_TORQUE_B_BOUNDS}, got {b}')
    if c < FORMULA_TORQUE_C_BOUNDS_NM[0] or c > FORMULA_TORQUE_C_BOUNDS_NM[1]:
        raise ValueError(f'c must be in {FORMULA_TORQUE_C_BOUNDS_NM}, got {c}')
    if d < FORMULA_TORQUE_D_BOUNDS_NM[0] or d > FORMULA_TORQUE_D_BOUNDS_NM[1]:
        raise ValueError(f'd must be in {FORMULA_TORQUE_D_BOUNDS_NM}, got {d}')
    if end_angle <= 0.0:
        raise ValueError(f'end_angle_rad must be positive, got {end_angle}')
    point_1 = (math.radians(-1.0), 0.4)
    point_2 = (0.0, 0.0)
    point_3 = (math.radians(-a / -0.4), -a)
    point_5 = (end_angle - math.radians(1.0) - math.radians(-d / 0.4), -d)
    point_4 = (point_3[0] + b * (point_5[0] - point_3[0]), -c)
    point_6 = (end_angle, -0.4)
    knots = (point_1, point_2, point_3, point_4, point_5, point_6)
    if any((right[0] <= left[0] for (left, right) in zip(knots, knots[1:]))):
        raise ValueError(f'formula torque-profile points must be strictly increasing in angle; got {[point[0] for point in knots]} radians')
    if mirror:
        knots = tuple(sorted(((-angle, -torque) for (angle, torque) in knots)))
    return tuple(((float(angle), float(torque)) for (angle, torque) in knots))

def zero_crossing_angle_rad(table: torch.Tensor) -> float:
    angles = table[:, 0]
    torques = table[:, 1]
    exact_zero_indices = torch.where(torques == 0.0)[0]
    exact_zero_angles = angles[exact_zero_indices]
    crossing_indices = torch.where(torques[:-1] * torques[1:] < 0.0)[0]
    crossing_angles = []
    for index_tensor in crossing_indices:
        index = int(index_tensor)
        x0 = angles[index]
        x1 = angles[index + 1]
        y0 = torques[index]
        y1 = torques[index + 1]
        crossing_angles.append(x0 - y0 * (x1 - x0) / (y1 - y0))
    if exact_zero_angles.numel() == 0 and len(crossing_angles) == 0:
        raise ValueError('ERC repaired torque table has no zero crossing')
    all_crossing_angles = torch.cat((exact_zero_angles, torch.stack(crossing_angles)) if crossing_angles else (exact_zero_angles,))
    closest_index = int(torch.argmin(torch.abs(all_crossing_angles)))
    return float(all_crossing_angles[closest_index])

def build_formula_erc_torque_function(parameters: FormulaTorqueParameters, fn_name: str, mirror: bool, spring_catalog_path: str | Path, spring_id: str, n_grid: int, safety_factor: float, max_torque_rmse_nm: float) -> ERCTorqueFunctionBuild:
    """Build a repaired ERC torque response from the formula-defined knots.

    The formula angles are radians because the simulator calls torque
    functions with joint angles in radians.
    """
    knots_rad_nm = formula_torque_profile_knots_rad_nm(parameters, mirror)
    erc_knots_rad_nm = tuple(((angle, -torque) for (angle, torque) in knots_rad_nm))
    knots = torch.tensor(erc_knots_rad_nm, dtype=torch.float64)
    profile = PiecewiseLinearTorqueProfile.from_xy(knots[:, 0], knots[:, 1])
    spring = SpringCatalog.from_yaml(spring_catalog_path)[spring_id]
    designer = ERCDesigner(spring, ERCDesignConfig(n_grid=n_grid, safety_factor=safety_factor, angle_convention='joint', joint_angle_limits_rad=(float(knots[0, 0]), float(knots[-1, 0])), energy_reference_angle_rad=0.0, invert_output_torque=True))
    result = designer.design(profile)
    if result.torque_rmse_nm > max_torque_rmse_nm:
        raise ValueError(f'ERC repair changed the requested torque profile too much: RMSE={result.torque_rmse_nm:.6g} Nm > {max_torque_rmse_nm:.6g} Nm')
    angle_offset_rad = zero_crossing_angle_rad(result.output_torque_table)
    shifted_output_torque_table = result.output_torque_table.clone()
    shifted_output_torque_table[:, 0] = shifted_output_torque_table[:, 0] - angle_offset_rad
    result = replace(result, output_torque_table=shifted_output_torque_table)
    function_string = torque_table_to_function_string(result.output_torque_table, fn_name, 'flat', (0.4, -0.4))
    namespace = {'torch': torch}
    exec(function_string, namespace)
    knots_deg_nm = tuple(((float(angle * 180.0 / math.pi), float(torque)) for (angle, torque) in knots_rad_nm))
    return ERCTorqueFunctionBuild(torque_function=namespace[fn_name], function_string=function_string, design_result=result, requested_knots_deg_nm=knots_deg_nm, actual_parameters=(float(parameters.a), float(parameters.b), float(parameters.c), float(parameters.d), float(parameters.end_angle_rad)), angle_offset_rad=angle_offset_rad)