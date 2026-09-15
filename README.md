# DentaVision 3D Registration & Modeling

3D point-cloud registration and surface-reconstruction pipeline for the DentaVision intraoral-scanner project.

The project evaluates multiple pairwise registration methods, performs sequential multiway alignment of overlapping dental fragments, fuses the registered point clouds, reconstructs a surface mesh, and evaluates the resulting geometry.

## Pipeline

1. Point-cloud preprocessing
2. Pairwise registration
3. Global feature matching with FPFH
4. Global alignment using RANSAC or Fast Global Registration (FGR)
5. Point-to-Plane ICP refinement
6. Sequential multiway registration
7. Point-cloud fusion
8. Surface reconstruction
9. Mesh cleanup and topology analysis
10. Mesh simplification
11. PLY and STL export

## Selected Methods

### Registration

The final pairwise registration method selected from the evaluation is:

**FPFH + RANSAC + Point-to-Plane ICP**

Five registration approaches are implemented and evaluated:

- Point-to-Point ICP
- Point-to-Plane ICP
- Generalized ICP
- FPFH + RANSAC + Point-to-Plane ICP
- FPFH + FGR + Point-to-Plane ICP

The final realistic multiway validation uses sequential registration of adjacent fragments with the selected RANSAC-based method.

### Surface Reconstruction

The selected surface-reconstruction method is:

**Ball Pivoting Algorithm (BPA)**

Poisson Surface Reconstruction is retained as a comparison method. BPA was selected because it preserved the sampled dental surface more faithfully and avoided the unsupported surfaces observed in the Poisson reconstructions.

## Validation

The registration pipeline is evaluated in two stages.

### Controlled synthetic experiments

Synthetic dental-like point clouds are used to study the influence of:

- overlap
- rotation
- translation
- Gaussian noise

Ground-truth transformations allow translation and rotation errors to be measured directly.

### Five-model dental validation

A second validation uses five distinct dental model sets consisting of:

- 5 upper-jaw meshes
- 5 lower-jaw meshes
- 10 jaw meshes in total
- 4 overlapping fragments per jaw
- 30 adjacent fragment pairs

The source meshes are STL reference models that are sampled virtually into point clouds. They are therefore more realistic geometrically than the synthetic benchmark, but they are not raw point clouds captured directly by an intraoral scanner.

The final RANSAC-based registration method successfully registered all 30 adjacent fragment pairs in the noise-free five-model benchmark.

## Project Structure

```text
3-D-Registrierung/
├── archive/          # superseded and historical experiment scripts
├── assets/           # report figures and visual material
├── data/             # input models and generated test datasets
├── diagnostics/      # diagnostic scripts
├── docs/             # project documentation
├── experiments/      # current benchmark and validation scripts
├── output/           # generated CSV files, meshes and point clouds
├── src/
│   └── registration3d/
│       ├── algorithms/       # pairwise registration algorithms
│       ├── reconstruction/   # mesh reconstruction and processing
│       └── ...               # preprocessing, evaluation and multiway logic
├── visualization/    # scripts used to generate report figures
├── pyproject.toml
├── poetry.lock
└── README.md
```

## Environment

- Python 3.12
- Open3D 0.19.0
- NumPy
- SciPy
- Pandas
- Matplotlib
- Poetry

## Installation

From the project directory:

```powershell
poetry install
```

## Core Package Usage

Pairwise registration demonstration:

```powershell
poetry run python -m registration3d.main
```

Controlled synthetic registration experiments:

```powershell
poetry run python -m registration3d.experiment_main
```

Synthetic multiway registration:

```powershell
poetry run python -m registration3d.multiway_main
```

Synthetic mesh reconstruction:

```powershell
poetry run python -m registration3d.reconstruction.mesh_main
```

Final synthetic mesh export:

```powershell
poetry run python -m registration3d.reconstruction.final_mesh_export
```

## Five-Model Experiments

The following scripts are the current realistic-model validation pipeline.

Generate the five-model fragment dataset:

```powershell
poetry run python experiments/create_realistic_model_suite_data.py
```

Validate model dimensions and generated dataset structure:

```powershell
poetry run python experiments/validate_realistic_model_suite.py
```

Run the pairwise benchmark:

```powershell
poetry run python experiments/realistic_model_suite_pairwise_benchmark.py
```

Run the noise-robustness benchmark:

```powershell
poetry run python experiments/realistic_model_suite_noise_benchmark.py
```

Run sequential multiway registration with the selected RANSAC method:

```powershell
poetry run python experiments/realistic_model_suite_multiway_ransac.py
```

Run the BPA parameter study:

```powershell
poetry run python experiments/realistic_model_suite_bpa_parameter_study.py
```

Compare BPA and Poisson reconstruction:

```powershell
poetry run python experiments/realistic_model_suite_mesh_validation.py
```

## Report Figures

Model-suite report figures can be generated with:

```powershell
poetry run python visualization/create_model_suite_report_plots.py
```

Additional visualization scripts are located in `visualization/`.

## Main Outputs

Important generated results include:

```text
output/controlled_experiments/
output/realistic_model_suite_pairwise/
output/realistic_model_suite_noise/
output/realistic_model_suite_multiway_ransac/
output/realistic_model_suite_bpa_parameter_study/
output/realistic_model_suite_mesh_validation/
output/mesh/
```

The realistic benchmark directories contain the raw measurements and aggregated CSV summaries used in the report.

## Notes on Reproducibility

- Controlled experiments use fixed random seeds where applicable.
- The synthetic registration benchmark uses repeated runs for each experimental condition.
- The realistic noise benchmark applies repeated perturbations to the same 30 base fragment pairs.
- Registration parameters are kept constant across the five-model validation unless explicitly varied by an experiment.
- Distance values for the realistic STL datasets are reported in **model units**, because the physical unit encoded by the reference meshes is not assumed.

## Documentation

The written project documentation and report material are stored in `docs/`.

Historical scripts that were superseded by the five-model validation are kept in `archive/` so that the final project structure remains clear without discarding earlier development stages.
