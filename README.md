# DentaVision 3D Registration & Modeling

3D point cloud registration and mesh reconstruction pipeline for the DentaVision intraoral scanner project.

## Pipeline

1. Point cloud preprocessing
2. Pairwise registration
3. FPFH + Fast Global Registration
4. Point-to-Plane ICP refinement
5. Multiway registration and pose graph optimization
6. Point cloud fusion
7. Surface reconstruction
8. Mesh cleanup and topology analysis
9. Mesh simplification
10. PLY and STL export

## Selected Methods

### Registration

FPFH + Fast Global Registration + Point-to-Plane ICP

### Surface Reconstruction

Ball Pivoting Algorithm

## Environment

- Python 3.12
- Open3D 0.19
- Poetry

## Installation

    poetry install

## Usage

Pairwise registration:

    poetry run python -m registration3d.main

Controlled registration experiments:

    poetry run python -m registration3d.experiment_main

Multiway registration:

    poetry run python -m registration3d.multiway_main

Mesh reconstruction:

    poetry run python -m registration3d.reconstruction.mesh_main

Final mesh export:

    poetry run python -m registration3d.reconstruction.final_mesh_export

## Documentation

The complete project documentation can be found in the `docs/` directory.
