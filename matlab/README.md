# MATLAB ERC Workflow

This directory contains the original MATLAB ERC implementation. It is an alternative to the Python package in the repository root, not a dependency of it.

Use this workflow if you want the original lab design path based on scripts and live scripts rather than the newer Python API.

If you want to read the mathematical derivation directly on GitHub, use
[`ERC_model.md`](ERC_model.md). If you want the original interactive live-script
version, use [`ERC_model.mlx`](ERC_model.mlx).

## Directory Contents

- [`Step1_SpringSelector.m`](Step1_SpringSelector.m): spring sizing and target-response preparation.
- [`Step2_ERCdesigner.m`](Step2_ERCdesigner.m): support-function design and torque reconstruction.
- [`Step3_ERCmodeller.m`](Step3_ERCmodeller.m): CAD-oriented export and AutoCAD script generation.
- [`repair_support_function.m`](repair_support_function.m): support-function repair utility.
- [`ERC_model.mlx`](ERC_model.mlx): original live-script formulation and derivation.
- [`ERC_model.md`](ERC_model.md): GitHub-readable markdown version of the derivation.
- [`ERC_demo.m`](ERC_demo.m): sensorized ERC real-time visualization.
- [`arduino/ERC_sensorised.ino`](arduino/ERC_sensorised.ino): Arduino sketch used with the sensorized demo.
- [`ERCscr.m`](ERCscr.m), [`ERCscr_sensorised_concave.m`](ERCscr_sensorised_concave.m), [`ERCscr_sensorised_convex_potentiometer.m`](ERCscr_sensorised_convex_potentiometer.m): AutoCAD script helpers.

## Recommended Usage

Run MATLAB from this `matlab/` directory, or add this directory to the MATLAB path before running the scripts. That keeps relative script calls and output paths consistent.

Typical sequence:

1. Run [`Step1_SpringSelector.m`](Step1_SpringSelector.m).
2. Run [`Step2_ERCdesigner.m`](Step2_ERCdesigner.m).
3. Run [`Step3_ERCmodeller.m`](Step3_ERCmodeller.m).

Outputs are intended to be written under [`results/`](results).

## Sensorized Variant

The sensorized ERC files remain in this subtree as part of the MATLAB-oriented workflow:

- use [`ERC_demo.m`](ERC_demo.m) together with [`arduino/ERC_sensorised.ino`](arduino/ERC_sensorised.ino),
- use the `ERCscr_sensorised_*` scripts when generating the sensorized CAD variant.

## Relation To The Python Workflow

The root-level Python workflow provides:

- a torch-native API,
- function-defined and point-wise ERC helpers,
- a repair-aware integration layer,
- Isaac Lab-facing wrappers and export utilities.

If that is what you need, return to the root [`README.md`](../README.md) and [`ISAAC_LAB_INTEGRATION.md`](../ISAAC_LAB_INTEGRATION.md).
