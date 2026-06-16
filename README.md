# Offline RTHS Code for MDOF Interface Coupling

This repository contains the numerical and experimental code used for the
offline real-time hybrid simulation (offline RTHS) study with
multi-degree-of-freedom (MDOF) interface coupling. The terminology follows the
manuscript: numerical substructure, experimental substructure, interface
compatibility, interface equilibrium, adaptive relaxation factor iteration
(ARFI), and Aitken's acceleration.

## Repository Structure

- `/Code/Numerical_Code/`
  Numerical simulations, spectral-radius analysis, convergence comparisons,
  and manuscript figure scripts for the cantilever-beam benchmark.
- `/Code/Experiment_Code/`
  Robotic-arm-based experimental-substructure execution, satellite-body
  coupling scripts, force/moment post-processing, and hybrid-test plotting.
- `Data/`
  Stored numerical and experimental results.
- `/Code/Experiment_Code/dobot_api.py`
  Vendor-provided Dobot robotic-arm TCP/IP API. This file is intentionally kept
  unchanged to preserve hardware communication behavior.

## Numerical Workflow

The main numerical benchmark is implemented in
`Numerical_Code/adaptive_relaxation_coupling.py`. It couples a
displacement-driven Dirichlet substructure and a force-driven Neumann
substructure by fixed-point iteration with relaxation. When adaptive relaxation
is enabled, Aitken's acceleration updates the relaxation factor during the
offline RTHS iterations.

Supporting modules include:

- `displacement_driven_beam.py`: time integration of the displacement-driven
  Dirichlet beam substructure.
- `force_driven_beam.py`: time integration of the force-driven Neumann beam
  substructure with boundary constraints.
- `dirichlet_dynamic_stiffness.py` and
  `neumann_dynamic_stiffness.py`: condensed interface dynamic stiffness
  matrices for spectral analysis.
- `fig3_*`, `fig4_*`, `fig5_6_*`, and `fig7_*`: manuscript figure and
  post-processing scripts.

## Experimental Workflow

The robotic-arm-based hybrid-test workflow is centered on
`Experiment_Code/aitken_satellite.py`. It commands interface displacement and
rotation histories, acquires force and moment responses from the force sensor,
transforms tool-frame measurements into the user frame, and stores iteration
results for later analysis.

Typical support scripts are:

- `main_satellite.py`: robotic-arm connection and response parsing utilities.
- `plot_sci_satellite_hybrid_results.py`: publication-quality plots from saved
  satellite hybrid-test data.


## Basic Usage

Run numerical scripts from inside `Numerical_Code/` when they rely on local
imports:

```bash
python adaptive_relaxation_coupling.py
python fig3_spectral_radius.py
python fig4_method_comparison.py
python fig5_6_error_history.py
python fig7_compare_with_numerical_solution.py
```

Run experimental scripts from inside `Experiment_Code/` after confirming the
robotic arm, TCP/IP mode, user frame, tool frame, load, eccentricity, and force
sensor are configured correctly:

```bash
python main_satellite.py
```

Hardware execution should be performed only after checking the inverse
kinematics preview, commanded trajectory continuity, and the safe working
envelope of the robotic arm.
