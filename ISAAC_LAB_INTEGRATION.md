# ERC Python API Reference

This note documents the Python ERC workflow only. The original MATLAB workflow
now lives under [`matlab/`](matlab) and is documented separately in
[`matlab/README.md`](matlab/README.md).

The MATLAB and Python implementations are alternative entrypoints. Use this
document when you want the torch-native API and the Isaac Lab-facing wrappers.

## High-Level Helpers

The repository now also exposes a small integration layer for common
simulation-facing operations:

```python
from erc_design import (
    build_erc_design,
    build_erc_design_from_function,
    build_erc_torque_table,
    build_erc_torque_table_from_function,
)
```

`build_erc_design(...)`
  Accepts torque knots as `(angle_deg, torque_nm)` pairs, loads the selected
  spring from the YAML catalog, runs the standard joint-angle ERC design
  pipeline, and returns an `ERCDesignResult`.

`build_erc_torque_table(...)`
  Runs the same pipeline and returns only the output torque table with columns
  `[joint_angle_rad, torque_nm]`.

`build_erc_design_from_function(...)`
  Accepts a `FunctionTorqueProfileConfig`, discretizes the requested torque
  function, negates it into the internal ERC sign convention, and then runs
  the full ERC design pipeline including repair.

  Optional argument:
  `max_torque_rmse_nm`
    Raise an error if the repaired ERC torque deviates from the requested
    profile by more than this RMSE threshold.

`build_erc_torque_table_from_function(...)`
  Returns only the repaired Isaac-ready torque table derived from a
  function-defined torque law.

`torque_table_to_function_string(...)`
  Serializes a generated torque table into a torch-callable Python function
  string for downstream pipelines that expect inline code instead of a tensor.

## Function-Defined Profiles

For bistable or otherwise analytic profiles, the repository exposes:

```python
from erc_design import FunctionTorqueProfileConfig, build_function_profile
```

Supported function modes are:

```text
sin
cos
tan
expression
saturating_dual_stiffness
```

Example:

```python
from erc_design import (
    FunctionTorqueProfileConfig,
    build_erc_torque_table_from_function,
)

table = build_erc_torque_table_from_function(
    FunctionTorqueProfileConfig(
        name="expression",
        expression="-0.4 * sin(2*pi*theta) - 0.1 * tan(theta * 0.96)",
        angle_min_rad=-1.48,
        angle_max_rad=1.48,
        num_samples=400,
    ),
)
```

This is the preferred way to port a function-defined programmable-joint law
from Isaac Lab into a realizable ERC approximation, because the output is
always passed through ERC energy scaling and convexity repair before export.

Sign convention note:

- define the input function in the same sign convention you want in Isaac Lab,
- do not negate it manually,
- the integration helpers negate it internally before calling `ERCDesigner`,
- the exported `output_torque_table` is again Isaac-facing.

## External Joint-Response Simulation

The example script

```python
examples/erc_bistable_isaaclab_example.py
```

does not implement a separate local mock simulator. Instead, it prepares a
repaired ERC response and then launches the local test harness:

```python
erc_isaac/joint_response.py
```

Workflow:

1. Build the repaired ERC torque table from the configured bistable function.
2. Generate a `joint_response.py` YAML config with `joint_response.mode=piecewise`
   and the repaired ERC table as the piecewise response.
3. If the required yaw-only USD asset is missing, call:

```python
scripts/convert_concentrated_urdf_yaw_only.py
```

4. Launch the local `joint_response.py` simulation with the generated config.

The wrapper also writes a recommended `test.torque_z` based on the repaired ERC
peak passive torque so the applied load in the Isaac Lab test is scaled to a
reasonable level.

The wrapper exposes runtime options such as `--headless`, `--video`,
`--analytics`, `--analytics_path`, and `--analytics_fps`. It writes the final
video and analytics paths into the generated config as absolute paths under
this repository's `results/` tree, and the analytics replay uses the same
playback slow-down as the viewport video. Any remaining unknown CLI options
are also forwarded.

After asset generation, the wrapper also recolors the yaw-only USD locally so:

- `arm_segment_1_*` (non-folding / proximal) uses a distinct warm color,
- `arm_segment_2_*` (folding / distal) uses a distinct green color,
- `motor_*` uses a separate motor color.

## Local Morphy Example

The repository also exposes a second standalone example:

```text
examples/erc_profile_morphy_example.py
```

This local script implements:

- repaired point-wise ERC profile generation,
- fixed-base Morphy simulation,
- viewport video recording,
- analytics video export.

The local wrapper exposes the main runtime options directly:

- `--headless`
- `--video`
- `--analytics`
- `--analytics_path`
- `--analytics_fps`
- `--camera_eye`
- `--camera_target`
- `--results_dir`

All outputs are written locally under `results/morphy_erc_profile_example/`
unless `--results_dir` is overridden.

## SpringCatalog

```python
catalog = SpringCatalog.from_yaml(path)
spring = catalog[spring_id]
```

Arguments:

```text
path
  YAML file containing a top-level `springs:` list.

spring_id
  String ID of the spring entry to use.
```

Expected YAML fields per spring:

```yaml
springs:
  - id: example_spring
    description: Optional human-readable description
    material: Optional material name
    wire_diameter_m: 0.001
    external_diameter_m: 0.006
    free_length_m: 0.021
    max_length_m: 0.040
    max_extension_m: 0.0125
    tension_at_rest_n: 0.8
    maximum_load_n: 21.88
    spring_constant_n_per_m: 1750.0
    count: 1
```

Required fields:

```text
id
max_length_m
max_extension_m
tension_at_rest_n
spring_constant_n_per_m
```

Optional fields:

```text
description
material
wire_diameter_m
external_diameter_m
free_length_m
maximum_load_n
count
```

All values are SI units.

## PiecewiseLinearTorqueProfile

```python
profile = PiecewiseLinearTorqueProfile.from_xy(
    angles_rad,
    torques_nm,
)
```

Arguments:

```text
angles_rad
  1D tensor-like sequence of strictly increasing profile knot angles.

torques_nm
  1D tensor-like sequence of torque values at those knots.
```

Output:

```text
PiecewiseLinearTorqueProfile
  Stores knots as a torch tensor shaped `(n_points, 2)` with columns
  `[angle_rad, torque_nm]`.
```

Notes:

```text
The angle convention is interpreted later by ERCDesignConfig.
Use joint-angle knots when angle_convention="joint".
Use cam-angle knots when angle_convention="cam".
```

## ProfileParameterization

```python
param = ProfileParameterization(
    n_segments=5,
    angle_bounds_rad=None,
    fixed_start_torque_nm=None,
    fixed_end_torque_nm=None,
    sort_angles=True,
)

profile = param.decode(parameters)
```

Constructor arguments:

```text
n_segments
  Number of piecewise-linear segments. Default is 5, giving 6 knots.

angle_bounds_rad
  Optional `(min_angle, max_angle)` tuple. If provided, endpoint angles are
  fixed and the parameter vector contains only interior angles.

fixed_start_torque_nm
  Optional fixed torque at the first knot.

fixed_end_torque_nm
  Optional fixed torque at the last knot.

sort_angles
  If true, angle parameters are sorted before profile construction.
```

Parameter-vector layout:

```text
[all angle parameters, all free torque parameters]
```

Parameter counts for the default 5-segment profile:

```text
Free x-y profile:
  6 angles + 6 torques = 12 parameters

Fixed endpoint torques:
  6 angles + 4 torques = 10 parameters

Fixed angle bounds and fixed endpoint torques:
  4 interior angles + 4 interior torques = 8 parameters
```

Output:

```text
PiecewiseLinearTorqueProfile
```

## ERCDesignConfig

```python
config = ERCDesignConfig(
    n_grid=1000,
    safety_factor=1.1,
    convexity_eps_m=1e-3,
    energy_scale_eps=1e-6,
    angle_convention="joint",
    joint_angle_limits_rad=(-math.pi / 2.0, math.pi / 2.0),
    angle_limit_tolerance_rad=1e-7,
    energy_reference_angle_rad=0.0,
    invert_output_torque=True,
)
```

Arguments:

```text
n_grid
  Number of samples used for energy integration, repair, cam generation, and
  output table generation.

safety_factor
  Spring safety factor applied to usable extension/energy.

convexity_eps_m
  Minimum allowed curvature radius target used by the repair solve.

energy_scale_eps
  Small margin subtracted when the energy profile must be scaled down.

angle_convention
  "joint" if profile angles are joint angles.
  "cam" if profile angles are single-cam angles.

joint_angle_limits_rad
  Hard joint-angle limits. Default is `[-pi/2, pi/2]`.
  In cam convention, profile input angles are checked against half of this
  interval.

angle_limit_tolerance_rad
  Numerical tolerance for accepting boundary angles.

energy_reference_angle_rad
  Joint angle at which integrated energy is shifted to zero.

invert_output_torque
  If true, the returned output torque table is sign-inverted for downstream use.
```

## ERCDesigner

```python
designer = ERCDesigner(spring, config)
result = designer.design(profile, repair=True)
```

Constructor arguments:

```text
spring
  Spring object loaded from SpringCatalog.

config
  ERCDesignConfig. If omitted, defaults are used.
```

`design(...)` arguments:

```text
profile
  PiecewiseLinearTorqueProfile.

repair
  If true, profiles with negative `r + r''` are repaired using the finite-
  difference support-function repair.
```

Output:

```text
ERCDesignResult
```

## ERCDesignResult

Main fields:

```text
input_profile
  Original PiecewiseLinearTorqueProfile.

output_torque_table
  Torch tensor shaped `(n_grid, 2)`.
  Columns are `[joint_angle_rad, torque_nm]`.
  Torque is sign-inverted if `invert_output_torque=True`.

target_torque_nm
  Requested torque sampled on the internal grid, before energy scaling and
  repair.

scaled_torque_nm
  Torque after energy scaling.

repaired_torque_nm
  Torque after convexity repair. This is before optional output sign inversion.

cam_xy_m
  Pre-repair cam profile coordinates in meters.

repaired_cam_xy_m
  Final admissible cam profile coordinates in meters.

support_radius_m
  Pre-repair support function radius.

repaired_support_radius_m
  Final support function radius.

curvature_radius_m
  Pre-repair `r + r''`.

repaired_curvature_radius_m
  Final `r + r''`.

alpha
  Energy scaling factor. `1.0` means no energy scaling.

was_scaled
  True if the profile exceeded spring energy limits and was scaled.

was_repaired
  True if convexity repair was applied.

torque_rmse_nm
  RMSE between requested sampled torque and final repaired torque before output
  sign inversion.
```

Export methods:

```python
result.export_xyz(path, repaired=True, n_points=None)
result.export_sldcrv(path, repaired=True, n_points=None)
```

Arguments:

```text
path
  Output file path.

repaired
  If true, export the final repaired cam profile.
  If false, export the pre-repair cam profile.

n_points
  Optional number of points to export. If omitted, the curve is resampled to
  twice the internal grid length.
```

Output format:

```text
x_mm y_mm z_mm
```

The design computations use SI units and radians. Export files are converted to
millimeters for CAD import.

## Torque Table To Callable String

If a downstream system needs a torch-callable code string rather than a tensor,
the output table can be serialized with:

```python
def torque_table_to_function_string(table, fn_name: str) -> str:
    table = table.detach().cpu()
    angles = [float(v) for v in table[:, 0]]
    torques = [float(v) for v in table[:, 1]]
    if fn_name not in {"torque_fn_left", "torque_fn_right"}:
        raise ValueError(fn_name)

    return f"""
_erc_angles = {angles!r}
_erc_torques = {torques!r}
_erc_cache = {{}}
def {fn_name}(x):
    key = (str(x.device), x.dtype)
    if key not in _erc_cache:
        _erc_cache[key] = (
            torch.tensor(_erc_angles, device=x.device, dtype=x.dtype),
            torch.tensor(_erc_torques, device=x.device, dtype=x.dtype),
        )
    xp, fp = _erc_cache[key]
    q = torch.clamp(x, xp[0], xp[-1])
    idx = torch.searchsorted(xp, q, right=True) - 1
    idx = torch.clamp(idx, 0, xp.numel() - 2)
    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]
    return y0 + (q - x0) / (x1 - x0) * (y1 - y0)
"""
```

Inputs:

```text
table
  `ERCDesignResult.output_torque_table`.

fn_name
  Function name to define in the string.
```

Output:

```text
Python code string defining a torch function with the given name.
```
