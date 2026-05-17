"""Airframe class for unified representation and manipulation of drone airframe parameters.

This module provides a single Airframe class that encapsulates all airframe-related
operations, including:
- Parameter storage in real units (meters, degrees)
- Conversion between different representations (normalized [0,1], actual values, dict)
- Generation of airframe properties JSON
- Computation of allocation matrices, torque parameters, max arm angles
- Validation of hover capability and arm deflection constraints
- Repair functions to fix invalid designs

Internal representation uses actual parameter values. All conversions are handled
through properties and factory methods.
"""
from __future__ import annotations
import numpy as np
import json
from pathlib import Path
from typing import Optional
from functools import cached_property
from collections.abc import Mapping
PARAM_BOUNDS = {'front_arm_length': (0.05, 0.15), 'front_prop_rotation_x': (-45, 45), 'front_prop_rotation_y': (-45, 45), 'front_prop_rotation_z': (-45, 45), 'back_arm_length': (0.05, 0.15), 'back_prop_rotation_x': (-45, 45), 'back_prop_rotation_y': (-45, 45), 'back_prop_rotation_z': (-45, 45), 'side_arm_seg2_length_left': (0.05, 0.15), 'side_arm_seg2_length_right': (0.05, 0.15), 'side_base_rotation_x_left': (-30, 30), 'side_base_rotation_y_left': (-45, 45), 'side_base_rotation_z_left': (-5, 5), 'side_base_rotation_x_right': (-30, 30), 'side_base_rotation_y_right': (-45, 45), 'side_base_rotation_z_right': (-5, 5), 'side_motor_rotation_x_left': (-45, 45), 'side_motor_rotation_y_left': (-45, 45), 'side_motor_rotation_z_left': (-45, 45), 'side_motor_rotation_x_right': (-45, 45), 'side_motor_rotation_y_right': (-45, 45), 'side_motor_rotation_z_right': (-45, 45)}
PARAM_NAMES = list(PARAM_BOUNDS.keys())
TORQUE_RESPONSE_MAX_ANGLE_DEG = 180.0
TORQUE_RESPONSE_RAMP_ANGLE_DEG = 2.0
TORQUE_RESPONSE_AMPLITUDE = 1.0
TORQUE_RESPONSE_BETWEEN_TORQUE = 0.0
ADD_ROD = True

class AirframeConstants(Mapping):
    arm_width = 0.01
    arm_height = 0.01
    arm_density = 200.0
    motor_mass = 0.01625
    motor_radius = 0.045
    motor_height = 0.035
    base_mass = 0.225
    base_box_dimensions = (0.07, 0.05, 0.08)
    side_seg1_len = 0.02
    rod_radius = 0.003
    motor_idx_list = [5, 5, 5, 5]
    motor_directions = [-1, 1, -1, 1]
    _keys = {'arm_width', 'arm_height', 'arm_density', 'motor_mass', 'motor_radius', 'motor_height', 'base_mass', 'base_box_dimensions', 'side_seg1_len', 'rod_radius', 'motor_idx_list', 'motor_directions', 'thrust_coefficient', 'cq', 'max_rps', 'min_rps'}

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)
AIRFRAME_CONSTANTS = AirframeConstants()

class Airframe:
    """Unified airframe representation with conversion and computation methods.

    The internal representation stores parameters in their actual units (meters for lengths,
    degrees for angles). Factory methods and properties provide access to different
    representations as needed.

    Attributes:
        _params: Dictionary mapping parameter names to actual values (meters, degrees)
    """

    def __init__(self, params: dict[str, float], skipp_validation: bool=False) -> None:
        """Initialize Airframe with actual parameter values.

        Args:
            params: Dictionary mapping parameter names to actual values

        Raises:
            ValueError: If required parameters are missing or out of bounds
        """
        if not skipp_validation:
            self._validate_params(params)
        self._params = params.copy()
        self._cached_json_data = None

    def _validate_params(self, params: dict[str, float]) -> None:
        """Validate that all required parameters are present and within bounds."""
        missing = [name for name in PARAM_NAMES if name not in params]
        if missing:
            raise ValueError(f'Missing required parameters: {missing}')
        for (name, value) in params.items():
            if name not in PARAM_BOUNDS:
                continue
            (min_val, max_val) = PARAM_BOUNDS[name]
            if not min_val <= value <= max_val:
                raise ValueError(f"Parameter '{name}' = {value} out of bounds [{min_val}, {max_val}]")

    @classmethod
    def from_x_0_1(cls, x_0_1: list | np.ndarray, skip_validation: bool=False) -> Airframe:
        """Create Airframe from normalized [0,1] parameter vector.

        Args:
            x_0_1: List or array of 31 normalized parameters in [0,1] range
            skip_validation: If True, skip parameter bounds validation

        Returns:
            Airframe instance
        """
        if len(x_0_1) != len(PARAM_NAMES):
            raise ValueError(f'Expected {len(PARAM_NAMES)} parameters, got {len(x_0_1)}')
        actual_params = cls._map_params_to_ranges(x_0_1)
        params = dict(zip(PARAM_NAMES, actual_params))
        return cls(params, skipp_validation=skip_validation)

    @classmethod
    def from_params_dict(cls, params: dict[str, float]) -> Airframe:
        """Create Airframe from parameter dictionary with actual values.

        Args:
            params: Dictionary mapping parameter names to actual values

        Returns:
            Airframe instance
        """
        return cls(params)

    @classmethod
    def from_json(cls, json_path: str | Path) -> Airframe:
        """Create Airframe from JSON file.

        Args:
            json_path: Path to airframe_data.json file

        Returns:
            Airframe instance
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        if 'parameters' in data:
            params = data['parameters']
        elif 'x_0_1' in data:
            return cls.from_x_0_1(data['x_0_1'])
        else:
            raise ValueError("JSON must contain 'parameters' or 'x_0_1' field")
        return cls(params)

    @classmethod
    def random(cls, rng: Optional[np.random.Generator]=None) -> Airframe:
        """Generate random airframe with parameters uniformly sampled from bounds.

        Args:
            rng: Optional numpy random generator for reproducibility

        Returns:
            Airframe instance
        """
        if rng is None:
            rng = np.random.default_rng()
        x_0_1 = rng.uniform(0, 1, len(PARAM_NAMES))
        return cls.from_x_0_1(x_0_1)

    @staticmethod
    def _map_params_to_ranges(x_0_1: list | np.ndarray) -> list[float]:
        """Convert normalized [0,1] parameters to actual values."""
        return [min_val + val * (max_val - min_val) for (val, (min_val, max_val)) in zip(x_0_1, PARAM_BOUNDS.values())]

    @staticmethod
    def _normalize_params(actual_values: list | np.ndarray) -> list[float]:
        """Convert actual parameter values to normalized [0,1] range."""
        return [(val - min_val) / (max_val - min_val) if max_val != min_val else 0.0 for (val, (min_val, max_val)) in zip(actual_values, PARAM_BOUNDS.values())]

    @property
    def params(self) -> dict[str, float]:
        """Get parameter dictionary with actual values (meters, degrees)."""
        return self._params.copy()

    @property
    def x_0_1(self) -> np.ndarray:
        """Get normalized [0,1] parameter vector."""
        actual_values = [self._params[name] for name in PARAM_NAMES]
        return np.array(self._normalize_params(actual_values))

    def with_updated_params(self, updates: dict[str, float]) -> Airframe:
        """Return new Airframe with updated parameters.

        Args:
            updates: Dict of parameter names and new values

        Returns:
            New Airframe instance with updated parameters
        """
        new_params = self._params.copy()
        new_params.update(updates)
        return Airframe(new_params)

    def _invalidate_cache(self) -> None:
        """Invalidate cached computations."""
        self._cached_json_data = None

    def compute_total_mass(self) -> float:
        """Compute total mass of the airframe."""
        arm_width = AIRFRAME_CONSTANTS['arm_width']
        arm_height = AIRFRAME_CONSTANTS['arm_height']
        arm_density = AIRFRAME_CONSTANTS['arm_density']
        motor_mass = AIRFRAME_CONSTANTS['motor_mass']
        base_mass = AIRFRAME_CONSTANTS['base_mass']
        side_seg1_len = AIRFRAME_CONSTANTS['side_seg1_len']
        front_arm_len = self._params['front_arm_length']
        back_arm_len = self._params['back_arm_length']
        side_seg2_len_left = self._params['side_arm_seg2_length_left']
        side_seg2_len_right = self._params['side_arm_seg2_length_right']
        front_arm_mass = front_arm_len * arm_width * arm_height * arm_density
        back_arm_mass = back_arm_len * arm_width * arm_height * arm_density
        side_seg1_mass = side_seg1_len * arm_width * arm_height * arm_density
        side_seg2_mass_left = side_seg2_len_left * arm_width * arm_height * arm_density
        side_seg2_mass_right = side_seg2_len_right * arm_width * arm_height * arm_density
        return base_mass + 4 * motor_mass + front_arm_mass + back_arm_mass + 2 * side_seg1_mass + side_seg2_mass_left + side_seg2_mass_right

    def compute_motor_positions_and_thrusts(self) -> dict:
        """Compute motor positions and thrust directions in body frame.

        Returns:
            Dictionary with:
                - motor_positions: (4, 3) array of motor positions
                - motor_thrust_directions: (4, 3) array of thrust directions
                - joint_1_position, joint_3_position: joint positions
                - joint_1_axis, joint_3_axis: joint rotation axes
        """
        import math

        def rpy_to_rotation_matrix(roll, pitch, yaw):
            (cr, sr) = (np.cos(roll), np.sin(roll))
            (cp, sp) = (np.cos(pitch), np.sin(pitch))
            (cy, sy) = (np.cos(yaw), np.sin(yaw))
            R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            return R_z @ R_y @ R_x
        params = self._params
        side_seg1_len = AIRFRAME_CONSTANTS['side_seg1_len']
        side_seg2_len_left = params['side_arm_seg2_length_left']
        side_seg2_len_right = params['side_arm_seg2_length_right']
        motor_positions = []
        motor_thrust_directions = []
        front_prop_rot_x = math.radians(params['front_prop_rotation_x'])
        front_prop_rot_y = math.radians(params['front_prop_rotation_y'])
        front_prop_rot_z = math.radians(params['front_prop_rotation_z'])
        motor_0_pos = np.array([0.035 + params['front_arm_length'], 0.0, 0.0])
        motor_0_thrust = rpy_to_rotation_matrix(front_prop_rot_x, front_prop_rot_y, front_prop_rot_z) @ np.array([0, 0, 1])
        motor_positions.append(motor_0_pos.tolist())
        motor_thrust_directions.append(motor_0_thrust.tolist())
        side_base_rot_x_left = math.radians(params['side_base_rotation_x_left'])
        side_base_rot_y_left = math.radians(params['side_base_rotation_y_left'])
        side_base_rot_z_left = math.radians(params['side_base_rotation_z_left'])
        side_motor_rot_x_left = math.radians(params['side_motor_rotation_x_left'])
        side_motor_rot_y_left = math.radians(params['side_motor_rotation_y_left'])
        side_motor_rot_z_left = math.radians(params['side_motor_rotation_z_left'])
        R_base_1 = rpy_to_rotation_matrix(side_base_rot_x_left, side_base_rot_y_left, side_base_rot_z_left + np.pi / 2)
        base_attachment_1 = np.array([0, 0.025, 0])
        joint_1_pos = base_attachment_1 + R_base_1 @ np.array([side_seg1_len, 0, 0])
        joint_1_axis = (R_base_1 @ np.array([0, 0, 1])).tolist()
        motor_1_pos = base_attachment_1 + R_base_1 @ np.array([side_seg1_len + side_seg2_len_left, 0, 0])
        R_motor_1 = R_base_1 @ rpy_to_rotation_matrix(side_motor_rot_x_left, side_motor_rot_y_left, side_motor_rot_z_left)
        motor_1_thrust = R_motor_1 @ np.array([0, 0, 1])
        motor_positions.append(motor_1_pos.tolist())
        motor_thrust_directions.append(motor_1_thrust.tolist())
        back_prop_rot_x = math.radians(params['back_prop_rotation_x'])
        back_prop_rot_y = math.radians(params['back_prop_rotation_y'])
        back_prop_rot_z = math.radians(params['back_prop_rotation_z'])
        R_back = rpy_to_rotation_matrix(0, 0, np.pi)
        pos_back = np.array([-0.035, 0, 0]) + R_back @ np.array([params['back_arm_length'], 0, 0])
        R_motor_2 = R_back @ rpy_to_rotation_matrix(back_prop_rot_x, back_prop_rot_y, back_prop_rot_z)
        motor_2_thrust = R_motor_2 @ np.array([0, 0, 1])
        motor_positions.append(pos_back.tolist())
        motor_thrust_directions.append(motor_2_thrust.tolist())
        side_base_rot_x_right = math.radians(params['side_base_rotation_x_right'])
        side_base_rot_y_right = math.radians(params['side_base_rotation_y_right'])
        side_base_rot_z_right = math.radians(params['side_base_rotation_z_right'])
        side_motor_rot_x_right = math.radians(params['side_motor_rotation_x_right'])
        side_motor_rot_y_right = math.radians(params['side_motor_rotation_y_right'])
        side_motor_rot_z_right = math.radians(params['side_motor_rotation_z_right'])
        R_base_3 = rpy_to_rotation_matrix(side_base_rot_x_right, side_base_rot_y_right, side_base_rot_z_right - np.pi / 2)
        base_attachment_3 = np.array([0, -0.025, 0])
        joint_3_pos = base_attachment_3 + R_base_3 @ np.array([side_seg1_len, 0, 0])
        joint_3_axis = (R_base_3 @ np.array([0, 0, 1])).tolist()
        motor_3_pos = base_attachment_3 + R_base_3 @ np.array([side_seg1_len + side_seg2_len_right, 0, 0])
        R_motor_3 = R_base_3 @ rpy_to_rotation_matrix(side_motor_rot_x_right, side_motor_rot_y_right, side_motor_rot_z_right)
        motor_3_thrust = R_motor_3 @ np.array([0, 0, 1])
        motor_positions.append(motor_3_pos.tolist())
        motor_thrust_directions.append(motor_3_thrust.tolist())
        return {'motor_positions': motor_positions, 'motor_thrust_directions': motor_thrust_directions, 'joint_1_position': joint_1_pos.tolist(), 'joint_3_position': joint_3_pos.tolist(), 'joint_1_axis': joint_1_axis, 'joint_3_axis': joint_3_axis}

    def save_json(self, path: str | Path) -> None:
        """Save airframe properties to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_json_dict(), f, indent=2)

    def __repr__(self) -> str:
        return f'Airframe(mass={self.compute_total_mass():.4f}kg, x_0_1={self.x_0_1[:5].round(3)}...)'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Airframe):
            return False
        return np.allclose(self.x_0_1, other.x_0_1, atol=1e-08)