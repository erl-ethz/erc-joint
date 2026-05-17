"""Python ERC cam design utilities.

The package ports the support-function cam profile workflow from
``matlab/ERC_model.mlx`` into torch-native classes that can be called from an
Isaac Lab codesign loop.
"""

from .designer import ERCDesignConfig, ERCDesignResult, ERCDesigner
from .function_profiles import FunctionTorqueProfileConfig, build_function_profile
from .integration import (
    DEFAULT_SPRING_CATALOG,
    DEFAULT_SPRING_ID,
    build_erc_design,
    build_erc_design_from_function,
    build_erc_torque_table,
    build_erc_torque_table_from_function,
    negate_profile_torque,
    torque_table_to_function_string,
)
from .profiles import PiecewiseLinearTorqueProfile, ProfileParameterization
from .springs import Spring, SpringCatalog

__all__ = [
    "DEFAULT_SPRING_CATALOG",
    "DEFAULT_SPRING_ID",
    "ERCDesignConfig",
    "ERCDesignResult",
    "ERCDesigner",
    "FunctionTorqueProfileConfig",
    "PiecewiseLinearTorqueProfile",
    "ProfileParameterization",
    "Spring",
    "SpringCatalog",
    "build_erc_design",
    "build_erc_design_from_function",
    "build_function_profile",
    "build_erc_torque_table",
    "build_erc_torque_table_from_function",
    "negate_profile_torque",
    "torque_table_to_function_string",
]
