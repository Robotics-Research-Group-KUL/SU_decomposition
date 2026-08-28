[![KU Leuven](https://img.shields.io/badge/KU%20Leuven-research-1E64C8)](https://www.kuleuven.be/)
[![CI](https://github.com/arnoverduyn/SU_decomposition/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/arnoverduyn/SU_decomposition/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/sulib.svg?cacheSeconds=300)](https://pypi.org/project/sulib/)
[![Python](https://img.shields.io/pypi/pyversions/sulib?cacheSeconds=300)](https://pypi.org/project/sulib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# SU Decomposition

This repository contains a Python implementation of the **SU-decomposition**, which can be used to compute a **coordinate-invariant local representation** for rigid-body motion trajectories and force trajectories. This representation is referred to as the **Dual-Upper-Triangular Invariant Representation (DUTIR)**.

The implementation is intended for trajectory analysis, identification, and generalization across coordinate systems. A detailed description of the method is provided in the corresponding paper:

> **A Coordinate-Invariant Local Representation of Motion and Force Trajectories for Identification and Generalization Across Coordinate Systems**

A preprint is available on [arXiv](https://arxiv.org/abs/2604.10241).

## Code highlights

The main functionalities are:
* `sulib.su_decomp.SU()`: Compute the SU-decomposition of a *6x3* matrix.
* `sulib.su_decomp.screw_trajectory_to_dutir()`: Compute the DUTIR from a *6xN* screw trajectory, where *N* represents the number of trajectory samples. The DUTIR is computed by applying the SU-decomposition to successive and overlapping windows of screw triplets.
* `sulib.su_decomp.pose_trajectory_to_dutir()`: Compute the DUTIR from a *4x4xN* rigid-body pose trajectory, where *N* represents the number of trajectory samples. The pose trajectory is first converted into a screw trajectory, after which `sulib.su_decomp.screw_trajectory_to_dutir()` is applied.
  
## Duality between motion and force
The function `sulib.su_decomp.compute_dutir_from_screw_traj()` takes screw trajectories as input. For motion, these screw trajectories correspond to **twist trajectories**. For force/torque data, they correspond to **wrench trajectories**.

Conceptually:

```text
Rigid-body pose trajectory
          │
          ▼
    Twist trajectory                         Wrench trajectory
          │                                         │
          ▼                                         ▼
   SU-decomposition                          SU-decomposition
          │                                         │
          ▼                                         ▼
DUTIR of rigid-body motion               DUTIR of force/torque data
```

Hence, the functions `sulib.su_decomp.screw_trajectory_to_dutir()` and `sulib.su_decomp.SU()` can be applied directly to either twist or wrench data:

## Installation (public user)

Install the latest published version from PyPI:

```bash
pip install sulib
```

**Example**: compute the DUTIR of a synthetically generated trajectory
```python
import numpy
import sulib

# Generate pose trajectory data of a precession motion
T, dt = sulib.su_decomp.generate_synthetic_pose_trajectory(trajectory_type="rotation_3D")

# Calculate the DUTIR of this pose trajectory
dutir, twist_trajectory = sulib.su_decomp.pose_trajectory_to_dutir(T, dt, L=0.3, twist_type="body")

# Print the first three samples of the twist_trajectory as an intermediate result
print("Three twist samples:")
print(numpy.round(twist_trajectory[:,:3], 5))
print(" ")

# Print the first DUTIR sample
print("First DUTIR sample:")
print(numpy.round(dutir[:,:,0], 5))
```
Output:
```python
Three twist samples:
[[ 2.35575  2.35575  2.35575]
 [ 0.05608  0.1681   0.27975]
 [ 2.35575  2.35041  2.33975]
 [ 0.      -0.      -0.     ]
 [ 0.23557  0.23504  0.23398]
 [-0.00561 -0.01681 -0.02798]]
 
First DUTIR sample:
[[ 3.332    3.33012  3.32446]
 [-0.       0.11214  0.22409]
 [ 0.       0.      -0.00377]
 [ 0.      -0.      -0.     ]
 [ 0.       0.      -0.     ]
 [ 0.      -0.      -0.     ]]
```

## Installation (developer)

Clone the repository and install the required dependencies and sulib package:

```bash
git clone https://github.com/arnoverduyn/SU_decomposition.git
cd SU_decomposition
pip install -r requirements.txt
pip install -e .
```

The core implementation is located in `src/sulib`.

## Repository structure

```text
SU_decomposition/
├── data/
│   └── demos/
│       ├── contour_following/
│       ├── peg_on_hole_alignment/
│       ├── pouring/
│       └── pouring_objects/
│
├── figures/
│
├── notebooks/
│   ├── notebook.ipynb
│
├── scripts/
│   ├── example_public.py
│   ├── example_SU_calculation_force.py
│   └── example_SU_calculation_motion.py
│
├── src/
│   └── sulib/
│       ├── _data_handling.py
│       ├── _plotting.py
│       ├── _preprocessing.py
│       ├── _robotics.py
│       └── su_decomp.py
│
├── tests/
│   └── test_su_decomp.py
│
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

### Package modules (public)

* `sulib.su_decomp` — core SU-decomposition and DUTIR computation.

### Package modules (internal)

* `sulib._robotics` — robotics-related operations for rigid-body trajectories.
* `sulib._preprocessing` — trajectory preprocessing utilities.
* `sulib._data_handling` — loading and handling trajectory data.
* `sulib._plotting` — visualization utilities.

## Examples

The `notebooks/` directory contains numerical examples and demonstrations.

Additional example scripts are provided in `scripts/`:

## Reference

If you use this implementation in academic work, please cite the associated paper:

> **A Coordinate-Invariant Local Representation of Motion and Force Trajectories for Identification and Generalization Across Coordinate Systems**

Preprint: [arXiv:2604.10241](https://arxiv.org/abs/2604.10241)

## License

This project is distributed under the license specified in [`LICENSE`](LICENSE).

