# MATLAB Code for "Joint Position-Orientation Deployment Design of UAV-Borne Linear-Array Angle-of-Arrival Sensors for Target UAV Localization"

This repository contains the MATLAB implementation of the deployment design algorithm for UAV-borne linear‑array angle‑of‑arrival (AOA) sensors. The code reproduces the simulation results presented in the paper, including optimal positioning and orientation of the sensing UAVs to localize a target UAV.

## Requirements

- MATLAB R2022b or later (older versions may work but have not been tested)
- Required toolboxes:
  - Optimization Toolbox
  - Statistics and Machine Learning Toolbox (for random target generation)
  - Parallel Computing Toolbox (optional, to speed up Monte Carlo runs)
  - cvx
  - SDPT3
  - Sedumi
