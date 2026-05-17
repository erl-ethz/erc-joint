from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch

from .designer import ERCDesignConfig, ERCDesignResult, ERCDesigner
from .function_profiles import FunctionTorqueProfileConfig, build_function_profile
from .profiles import PiecewiseLinearTorqueProfile
from .springs import SpringCatalog


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPRING_CATALOG = PROJECT_ROOT / "configs" / "springs.yaml"
DEFAULT_SPRING_ID = "durovis_0.9x6.1x21"


def negate_profile_torque(profile: PiecewiseLinearTorqueProfile) -> PiecewiseLinearTorqueProfile:
    """Convert an Isaac-facing torque profile into the internal ERC sign convention.

    The support-function ERC workflow uses the opposite torque sign internally.
    Integration code should therefore negate simulator-facing torque profiles
    before passing them to ``ERCDesigner``.
    """

    knots = profile.to_tensor()
    knots[:, 1] = -knots[:, 1]
    return PiecewiseLinearTorqueProfile(knots)


def build_erc_design(
    knots_deg_nm: Sequence[tuple[float, float]],
    *,
    spring_catalog_path: str | Path = DEFAULT_SPRING_CATALOG,
    spring_id: str = DEFAULT_SPRING_ID,
    n_grid: int = 1000,
    safety_factor: float = 1.1,
    angle_limits_deg: tuple[float, float] = (-90.0, 90.0),
) -> ERCDesignResult:
    """Run the standard joint-angle ERC design pipeline from degree/Nm knots."""

    knots = torch.tensor(knots_deg_nm, dtype=torch.float64)
    profile = PiecewiseLinearTorqueProfile.from_xy(
        knots[:, 0] * math.pi / 180.0,
        knots[:, 1],
    )
    spring = SpringCatalog.from_yaml(spring_catalog_path)[spring_id]
    designer = ERCDesigner(
        spring,
        ERCDesignConfig(
            n_grid=n_grid,
            safety_factor=safety_factor,
            angle_convention="joint",
            joint_angle_limits_rad=(
                math.radians(angle_limits_deg[0]),
                math.radians(angle_limits_deg[1]),
            ),
            invert_output_torque=True,
        ),
    )
    return designer.design(profile)


def build_erc_torque_table(
    knots_deg_nm: Sequence[tuple[float, float]],
    *,
    spring_catalog_path: str | Path = DEFAULT_SPRING_CATALOG,
    spring_id: str = DEFAULT_SPRING_ID,
    n_grid: int = 1000,
    safety_factor: float = 1.1,
    angle_limits_deg: tuple[float, float] = (-90.0, 90.0),
) -> torch.Tensor:
    """Build an Isaac-ready torque table with columns ``[joint_angle_rad, torque_nm]``."""

    return build_erc_design(
        knots_deg_nm,
        spring_catalog_path=spring_catalog_path,
        spring_id=spring_id,
        n_grid=n_grid,
        safety_factor=safety_factor,
        angle_limits_deg=angle_limits_deg,
    ).output_torque_table


def build_erc_design_from_function(
    function_config: FunctionTorqueProfileConfig,
    *,
    spring_catalog_path: str | Path = DEFAULT_SPRING_CATALOG,
    spring_id: str = DEFAULT_SPRING_ID,
    n_grid: int = 1000,
    safety_factor: float = 1.1,
    joint_angle_limits_rad: tuple[float, float] | None = None,
    energy_reference_angle_rad: float = 0.0,
    max_torque_rmse_nm: float | None = None,
) -> ERCDesignResult:
    """Run ERC design from a function-defined joint torque profile.

    The input function is assumed to follow the simulator/Isaac convention.
    Its torque values are negated before entering the internal ERC workflow.
    """

    profile = negate_profile_torque(build_function_profile(function_config))
    if joint_angle_limits_rad is None:
        joint_angle_limits_rad = (
            function_config.angle_min_rad,
            function_config.angle_max_rad,
        )
    spring = SpringCatalog.from_yaml(spring_catalog_path)[spring_id]
    designer = ERCDesigner(
        spring,
        ERCDesignConfig(
            n_grid=n_grid,
            safety_factor=safety_factor,
            angle_convention="joint",
            joint_angle_limits_rad=joint_angle_limits_rad,
            energy_reference_angle_rad=energy_reference_angle_rad,
            invert_output_torque=True,
        ),
    )
    result = designer.design(profile)
    if max_torque_rmse_nm is not None and result.torque_rmse_nm > max_torque_rmse_nm:
        raise ValueError(
            "ERC repair changed the requested torque profile too much: "
            f"RMSE={result.torque_rmse_nm:.6g} Nm > {max_torque_rmse_nm:.6g} Nm"
        )
    return result


def build_erc_torque_table_from_function(
    function_config: FunctionTorqueProfileConfig,
    *,
    spring_catalog_path: str | Path = DEFAULT_SPRING_CATALOG,
    spring_id: str = DEFAULT_SPRING_ID,
    n_grid: int = 1000,
    safety_factor: float = 1.1,
    joint_angle_limits_rad: tuple[float, float] | None = None,
    energy_reference_angle_rad: float = 0.0,
    max_torque_rmse_nm: float | None = None,
) -> torch.Tensor:
    """Build an Isaac-ready torque table from a function-defined profile."""

    return build_erc_design_from_function(
        function_config,
        spring_catalog_path=spring_catalog_path,
        spring_id=spring_id,
        n_grid=n_grid,
        safety_factor=safety_factor,
        joint_angle_limits_rad=joint_angle_limits_rad,
        energy_reference_angle_rad=energy_reference_angle_rad,
        max_torque_rmse_nm=max_torque_rmse_nm,
    ).output_torque_table


def torque_table_to_function_string(
    table: torch.Tensor,
    *,
    fn_name: str = "erc_torque_fn",
    extrapolation: str = "raise",
    flat_boundary_torques_nm: tuple[float, float] | None = None,
) -> str:
    """Serialize a torque table as a small torch-callable function.

    This is useful when an Isaac Lab or USD pipeline expects an inline Python
    function instead of a tensor asset.
    """

    if extrapolation not in {"raise", "flat"}:
        raise ValueError("extrapolation must be 'raise' or 'flat'")
    if extrapolation == "flat" and flat_boundary_torques_nm is None:
        raise ValueError("flat_boundary_torques_nm is required when extrapolation='flat'")
    if extrapolation == "raise" and flat_boundary_torques_nm is not None:
        raise ValueError("flat_boundary_torques_nm must be None when extrapolation='raise'")

    table_cpu = table.detach().cpu()
    angles = [float(f"{float(value):.17g}") for value in table_cpu[:, 0]]
    torques = [float(f"{float(value):.17g}") for value in table_cpu[:, 1]]
    flat_lower = None if flat_boundary_torques_nm is None else float(flat_boundary_torques_nm[0])
    flat_upper = None if flat_boundary_torques_nm is None else float(flat_boundary_torques_nm[1])

    out_of_domain_logic = (
        f'raise ValueError("{fn_name} received an angle outside its ERC table domain")'
        if extrapolation == "raise"
        else "return torch.where(x < xp[0], _erc_flat_lower + x * 0.0, torch.where(x > xp[-1], _erc_flat_upper + x * 0.0, _erc_interp(x, xp, fp)))"
    )
    in_domain_logic = (
        "_erc_interp(x, xp, fp)"
        if extrapolation == "raise"
        else "torch.where(x < xp[0], _erc_flat_lower + x * 0.0, torch.where(x > xp[-1], _erc_flat_upper + x * 0.0, _erc_interp(x, xp, fp)))"
    )

    return f"""
_erc_angles = {angles!r}
_erc_torques = {torques!r}
_erc_flat_lower = {flat_lower!r}
_erc_flat_upper = {flat_upper!r}
_erc_cache = {{}}
def _erc_interp(x, xp, fp):
    idx = torch.searchsorted(xp, x, right=True) - 1
    idx = torch.clamp(idx, 0, xp.numel() - 2)
    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]
    return y0 + (x - x0) / (x1 - x0) * (y1 - y0)
def {fn_name}(x):
    key = (str(x.device), x.dtype)
    if key not in _erc_cache:
        _erc_cache[key] = (
            torch.tensor(_erc_angles, device=x.device, dtype=x.dtype),
            torch.tensor(_erc_torques, device=x.device, dtype=x.dtype),
        )
    xp, fp = _erc_cache[key]
    if torch.any(x < xp[0]) or torch.any(x > xp[-1]):
        {out_of_domain_logic}
    return {in_domain_logic}
{fn_name}.erc_angle_min_rad = _erc_angles[0]
{fn_name}.erc_angle_max_rad = _erc_angles[-1]
"""
