from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd

from registration3d.realistic_dataset_registry import (
    DENTAL_MODELS,
    JAW_NAMES,
    get_stl_path,
)


# ============================================================
# CONFIGURATION
# ============================================================

REFERENCE_SAMPLE_POINTS = 100_000
RECONSTRUCTION_SAMPLE_POINTS = 100_000

NORMAL_RADIUS_FACTOR = 3.0
NORMAL_MAX_NN = 50
NORMAL_ORIENTATION_K = 30

RANDOM_SEED = 42


BPA_CONFIGURATIONS = {
    "A_small": (
        1.0,
        1.5,
        2.0,
    ),

    "B_baseline": (
        1.5,
        2.0,
        3.0,
    ),

    "C_medium": (
        2.0,
        3.0,
        4.0,
    ),

    "D_large": (
        2.5,
        4.0,
        6.0,
    ),
}


# ============================================================
# BASIC LOADERS
# ============================================================

def load_cloud(
    path: Path,
) -> o3d.geometry.PointCloud:

    if not path.exists():
        raise FileNotFoundError(
            f"Point cloud not found:\n{path}"
        )

    cloud = (
        o3d.io.read_point_cloud(
            str(path)
        )
    )

    if cloud.is_empty():
        raise RuntimeError(
            f"Could not load point cloud:\n{path}"
        )

    return cloud


def load_mesh(
    path: Path,
) -> o3d.geometry.TriangleMesh:

    if not path.exists():
        raise FileNotFoundError(
            f"Mesh not found:\n{path}"
        )

    mesh = (
        o3d.io.read_triangle_mesh(
            str(path)
        )
    )

    if mesh.is_empty():
        raise RuntimeError(
            f"Could not load mesh:\n{path}"
        )

    return mesh


# ============================================================
# POINT-CLOUD PREPARATION
# ============================================================

def mean_nearest_neighbor_distance(
    cloud: o3d.geometry.PointCloud,
) -> float:

    distances = (
        cloud
        .compute_nearest_neighbor_distance()
    )

    values = np.asarray(
        distances,
        dtype=float,
    )

    if len(values) == 0:
        raise RuntimeError(
            "Could not calculate nearest-neighbor distances."
        )

    return float(
        np.mean(
            values
        )
    )


def prepare_cloud_for_bpa(
    cloud: o3d.geometry.PointCloud,
) -> tuple[
    o3d.geometry.PointCloud,
    float,
]:

    prepared = copy.deepcopy(
        cloud
    )

    mean_nn = (
        mean_nearest_neighbor_distance(
            prepared
        )
    )

    normal_radius = (
        mean_nn
        * NORMAL_RADIUS_FACTOR
    )

    prepared.estimate_normals(
        search_param=(
            o3d.geometry
            .KDTreeSearchParamHybrid(
                radius=normal_radius,
                max_nn=NORMAL_MAX_NN,
            )
        )
    )

    prepared.orient_normals_consistent_tangent_plane(
        NORMAL_ORIENTATION_K
    )

    prepared.normalize_normals()

    return (
        prepared,
        mean_nn,
    )


# ============================================================
# MESH CLEANUP
# ============================================================

def cleanup_mesh(
    mesh: o3d.geometry.TriangleMesh,
) -> None:

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()


# ============================================================
# BPA
# ============================================================

def reconstruct_bpa(
    prepared_cloud: o3d.geometry.PointCloud,
    mean_nn: float,
    radius_factors: tuple[
        float,
        float,
        float,
    ],
) -> tuple[
    o3d.geometry.TriangleMesh,
    list[
        float
    ],
]:

    radii = [
        mean_nn
        * factor

        for factor
        in radius_factors
    ]

    mesh = (
        o3d.geometry.TriangleMesh
        .create_from_point_cloud_ball_pivoting(
            prepared_cloud,
            o3d.utility.DoubleVector(
                radii
            ),
        )
    )

    cleanup_mesh(
        mesh
    )

    return (
        mesh,
        radii,
    )


# ============================================================
# COMPONENT METRICS
# ============================================================

def connected_component_metrics(
    mesh: o3d.geometry.TriangleMesh,
) -> tuple[
    int,
    float,
]:

    triangle_count = len(
        mesh.triangles
    )

    if triangle_count == 0:
        return (
            0,
            0.0,
        )

    (
        labels,
        cluster_counts,
        _,
    ) = (
        mesh
        .cluster_connected_triangles()
    )

    cluster_counts = np.asarray(
        cluster_counts,
        dtype=int,
    )

    component_count = len(
        cluster_counts
    )

    if component_count == 0:
        return (
            0,
            0.0,
        )

    largest_component_fraction = (
        float(
            np.max(
                cluster_counts
            )
        )
        / triangle_count
    )

    return (
        component_count,
        largest_component_fraction,
    )


# ============================================================
# BOUNDARY EDGES
# ============================================================

def count_boundary_edges(
    mesh: o3d.geometry.TriangleMesh,
) -> int:

    triangles = np.asarray(
        mesh.triangles,
        dtype=np.int64,
    )

    if len(triangles) == 0:
        return 0

    edges = np.vstack(
        [
            triangles[
                :,
                [0, 1],
            ],

            triangles[
                :,
                [1, 2],
            ],

            triangles[
                :,
                [2, 0],
            ],
        ]
    )

    edges = np.sort(
        edges,
        axis=1,
    )

    _, counts = np.unique(
        edges,
        axis=0,
        return_counts=True,
    )

    return int(
        np.sum(
            counts == 1
        )
    )


# ============================================================
# SURFACE SAMPLING
# ============================================================

def sample_mesh(
    mesh: o3d.geometry.TriangleMesh,
    number_of_points: int,
    seed: int,
) -> o3d.geometry.PointCloud:

    o3d.utility.random.seed(
        seed
    )

    return (
        mesh
        .sample_points_uniformly(
            number_of_points=(
                number_of_points
            )
        )
    )


# ============================================================
# GEOMETRIC DISTANCES
# ============================================================

def distance_statistics(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> dict[
    str,
    float,
]:

    distances = np.asarray(
        source.compute_point_cloud_distance(
            target
        ),
        dtype=float,
    )

    if len(distances) == 0:

        return {
            "mean":
                float(
                    "nan"
                ),

            "median":
                float(
                    "nan"
                ),

            "rmse":
                float(
                    "nan"
                ),

            "p95":
                float(
                    "nan"
                ),

            "max":
                float(
                    "nan"
                ),
        }

    return {
        "mean":
            float(
                np.mean(
                    distances
                )
            ),

        "median":
            float(
                np.median(
                    distances
                )
            ),

        "rmse":
            float(
                np.sqrt(
                    np.mean(
                        distances ** 2
                    )
                )
            ),

        "p95":
            float(
                np.percentile(
                    distances,
                    95.0,
                )
            ),

        "max":
            float(
                np.max(
                    distances
                )
            ),
    }


def evaluate_mesh(
    reference_samples:
        o3d.geometry.PointCloud,

    reconstructed_mesh:
        o3d.geometry.TriangleMesh,

    reference_surface_area: float,

) -> dict:

    reconstructed_samples = (
        sample_mesh(
            mesh=reconstructed_mesh,
            number_of_points=(
                RECONSTRUCTION_SAMPLE_POINTS
            ),
            seed=RANDOM_SEED,
        )
    )

    # --------------------------------------------------------
    # Reference -> Reconstruction
    # --------------------------------------------------------

    ref_to_rec = (
        distance_statistics(
            source=(
                reference_samples
            ),
            target=(
                reconstructed_samples
            ),
        )
    )

    # --------------------------------------------------------
    # Reconstruction -> Reference
    # --------------------------------------------------------

    rec_to_ref = (
        distance_statistics(
            source=(
                reconstructed_samples
            ),
            target=(
                reference_samples
            ),
        )
    )

    # --------------------------------------------------------
    # Symmetric measures
    # --------------------------------------------------------

    symmetric_mean = (
        0.5
        * (
            ref_to_rec[
                "mean"
            ]
            +
            rec_to_ref[
                "mean"
            ]
        )
    )

    symmetric_rmse = float(
        np.sqrt(
            0.5
            * (
                ref_to_rec[
                    "rmse"
                ] ** 2
                +
                rec_to_ref[
                    "rmse"
                ] ** 2
            )
        )
    )

    # --------------------------------------------------------
    # Structural metrics
    # --------------------------------------------------------

    (
        component_count,
        largest_component_fraction,
    ) = (
        connected_component_metrics(
            reconstructed_mesh
        )
    )

    boundary_edges = (
        count_boundary_edges(
            reconstructed_mesh
        )
    )

    reconstructed_area = float(
        reconstructed_mesh
        .get_surface_area()
    )

    if reference_surface_area > 0:

        area_ratio = (
            reconstructed_area
            / reference_surface_area
        )

    else:

        area_ratio = float(
            "nan"
        )

    # Self-intersection can be somewhat expensive,
    # but the meshes are small enough for this study.
    self_intersecting = bool(
        reconstructed_mesh
        .is_self_intersecting()
    )

    return {
        "Vertices":
            len(
                reconstructed_mesh.vertices
            ),

        "Triangles":
            len(
                reconstructed_mesh.triangles
            ),

        "Connected Components":
            component_count,

        "Largest Component Fraction":
            largest_component_fraction,

        "Boundary Edges":
            boundary_edges,

        "Self Intersecting":
            self_intersecting,

        "Reference Surface Area":
            reference_surface_area,

        "Reconstructed Surface Area":
            reconstructed_area,

        "Area Ratio":
            area_ratio,

        "Reference to Reconstruction Mean":
            ref_to_rec[
                "mean"
            ],

        "Reference to Reconstruction RMSE":
            ref_to_rec[
                "rmse"
            ],

        "Reference to Reconstruction P95":
            ref_to_rec[
                "p95"
            ],

        "Reconstruction to Reference Mean":
            rec_to_ref[
                "mean"
            ],

        "Reconstruction to Reference RMSE":
            rec_to_ref[
                "rmse"
            ],

        "Reconstruction to Reference P95":
            rec_to_ref[
                "p95"
            ],

        "Symmetric Mean":
            symmetric_mean,

        "Symmetric RMSE":
            symmetric_rmse,
    }


# ============================================================
# ONE JAW
# ============================================================

def run_one_jaw(
    project_root: Path,
    model_id: str,
    jaw: str,
    source_stl_path: Path,
) -> list[
    dict
]:

    combined_cloud_path = (
        project_root
        / "output"
        / "realistic_model_suite_multiway_ransac"
        / model_id
        / jaw
        / "combined_cloud.ply"
    )

    output_directory = (
        project_root
        / "output"
        / "realistic_model_suite_bpa_parameter_study"
        / model_id
        / jaw
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 100)

    print(
        f"{model_id} / "
        f"{jaw.upper()}"
    )

    print("=" * 100)

    # --------------------------------------------------------
    # Reference STL
    # --------------------------------------------------------

    reference_mesh = (
        load_mesh(
            source_stl_path
        )
    )

    reference_surface_area = float(
        reference_mesh
        .get_surface_area()
    )

    reference_samples = (
        sample_mesh(
            mesh=(
                reference_mesh
            ),
            number_of_points=(
                REFERENCE_SAMPLE_POINTS
            ),
            seed=RANDOM_SEED,
        )
    )

    # --------------------------------------------------------
    # Registered cloud
    # --------------------------------------------------------

    combined_cloud = (
        load_cloud(
            combined_cloud_path
        )
    )

    (
        prepared_cloud,
        mean_nn,
    ) = (
        prepare_cloud_for_bpa(
            combined_cloud
        )
    )

    print(
        f"Combined cloud: "
        f"{len(combined_cloud.points):,} points"
    )

    print(
        f"Mean NN distance: "
        f"{mean_nn:.6f}"
    )

    print(
        f"Reference area: "
        f"{reference_surface_area:.6f}"
    )

    rows = []

    # ========================================================
    # FOUR BPA CONFIGURATIONS
    # ========================================================

    for (
        configuration,
        radius_factors,
    ) in BPA_CONFIGURATIONS.items():

        print()
        print("-" * 100)

        print(
            f"{configuration}: "
            f"{radius_factors}"
        )

        print("-" * 100)

        mesh, radii = (
            reconstruct_bpa(
                prepared_cloud=(
                    prepared_cloud
                ),
                mean_nn=(
                    mean_nn
                ),
                radius_factors=(
                    radius_factors
                ),
            )
        )

        if mesh.is_empty():

            print(
                "WARNING: empty mesh."
            )

            continue

        metrics = (
            evaluate_mesh(
                reference_samples=(
                    reference_samples
                ),
                reconstructed_mesh=(
                    mesh
                ),
                reference_surface_area=(
                    reference_surface_area
                ),
            )
        )

        mesh_path = (
            output_directory
            / (
                f"{configuration}.ply"
            )
        )

        o3d.io.write_triangle_mesh(
            str(
                mesh_path
            ),
            mesh,
        )

        row = {
            "Model":
                model_id,

            "Jaw":
                jaw,

            "Configuration":
                configuration,

            "Radius Factor 1":
                radius_factors[0],

            "Radius Factor 2":
                radius_factors[1],

            "Radius Factor 3":
                radius_factors[2],

            "Radius 1":
                radii[0],

            "Radius 2":
                radii[1],

            "Radius 3":
                radii[2],

            "Mean NN Distance":
                mean_nn,
        }

        row.update(
            metrics
        )

        rows.append(
            row
        )

        print(
            f"Vertices:      "
            f"{metrics['Vertices']:,}"
        )

        print(
            f"Triangles:     "
            f"{metrics['Triangles']:,}"
        )

        print(
            f"Components:    "
            f"{metrics['Connected Components']}"
        )

        print(
            f"Largest comp.: "
            f"{metrics['Largest Component Fraction'] * 100:.3f}%"
        )

        print(
            f"Boundary edges:"
            f" {metrics['Boundary Edges']:,}"
        )

        print(
            f"Area ratio:    "
            f"{metrics['Area Ratio']:.6f}"
        )

        print(
            f"Sym. RMSE:     "
            f"{metrics['Symmetric RMSE']:.6f}"
        )

        print(
            f"Self intersect:"
            f" {metrics['Self Intersecting']}"
        )

    jaw_results = pd.DataFrame(
        rows
    )

    jaw_results.to_csv(
        output_directory
        / "bpa_results.csv",
        index=False,
    )

    return rows


# ============================================================
# AGGREGATE SUMMARY
# ============================================================

def create_configuration_summary(
    raw: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        raw.groupby(
            "Configuration",
            as_index=False,
        )
        .agg(
            Jaw_Meshes=(
                "Jaw",
                "size",
            ),

            Symmetric_RMSE_Mean=(
                "Symmetric RMSE",
                "mean",
            ),

            Symmetric_RMSE_Median=(
                "Symmetric RMSE",
                "median",
            ),

            Symmetric_RMSE_Max=(
                "Symmetric RMSE",
                "max",
            ),

            Boundary_Edges_Mean=(
                "Boundary Edges",
                "mean",
            ),

            Boundary_Edges_Median=(
                "Boundary Edges",
                "median",
            ),

            Connected_Components_Mean=(
                "Connected Components",
                "mean",
            ),

            Largest_Component_Mean=(
                "Largest Component Fraction",
                "mean",
            ),

            Area_Ratio_Mean=(
                "Area Ratio",
                "mean",
            ),

            Area_Ratio_Median=(
                "Area Ratio",
                "median",
            ),

            Self_Intersecting_Count=(
                "Self Intersecting",
                "sum",
            ),
        )
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    aggregate_directory = (
        project_root
        / "output"
        / "realistic_model_suite_bpa_parameter_study"
    )

    aggregate_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []

    print()
    print("=" * 100)
    print(
        "REALISTIC FIVE-MODEL BPA PARAMETER STUDY"
    )
    print("=" * 100)

    print()
    print(
        f"Models:      "
        f"{len(DENTAL_MODELS)}"
    )

    print(
        f"Jaw meshes:  "
        f"{len(DENTAL_MODELS) * len(JAW_NAMES)}"
    )

    print(
        f"BPA configs: "
        f"{len(BPA_CONFIGURATIONS)}"
    )

    print(
        f"Total meshes:"
        f" "
        f"{len(DENTAL_MODELS) * len(JAW_NAMES) * len(BPA_CONFIGURATIONS)}"
    )

    # ========================================================
    # ALL MODEL SETS
    # ========================================================

    for model in (
        DENTAL_MODELS
    ):

        for jaw in (
            JAW_NAMES
        ):

            source_stl_path = (
                get_stl_path(
                    project_root=(
                        project_root
                    ),
                    model=model,
                    jaw=jaw,
                )
            )

            rows = (
                run_one_jaw(
                    project_root=(
                        project_root
                    ),
                    model_id=(
                        model.model_id
                    ),
                    jaw=jaw,
                    source_stl_path=(
                        source_stl_path
                    ),
                )
            )

            all_rows.extend(
                rows
            )

    # ========================================================
    # SAVE AGGREGATE DATA
    # ========================================================

    raw = pd.DataFrame(
        all_rows
    )

    raw_path = (
        aggregate_directory
        / "bpa_parameter_raw.csv"
    )

    raw.to_csv(
        raw_path,
        index=False,
    )

    summary = (
        create_configuration_summary(
            raw
        )
    )

    summary_path = (
        aggregate_directory
        / "bpa_parameter_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Per-model summary
    # --------------------------------------------------------

    by_model = (
        raw.groupby(
            [
                "Model",
                "Configuration",
            ],
            as_index=False,
        )
        .agg(
            Symmetric_RMSE_Mean=(
                "Symmetric RMSE",
                "mean",
            ),

            Boundary_Edges_Mean=(
                "Boundary Edges",
                "mean",
            ),

            Area_Ratio_Mean=(
                "Area Ratio",
                "mean",
            ),

            Self_Intersecting_Count=(
                "Self Intersecting",
                "sum",
            ),
        )
    )

    by_model.to_csv(
        aggregate_directory
        / "bpa_parameter_by_model.csv",
        index=False,
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print("=" * 100)
    print(
        "AGGREGATE BPA RESULTS"
    )
    print("=" * 100)
    print()

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("=" * 100)
    print(
        "SYMMETRIC RMSE BY MODEL"
    )
    print("=" * 100)
    print()

    rmse_pivot = (
        by_model.pivot(
            index="Model",
            columns="Configuration",
            values="Symmetric_RMSE_Mean",
        )
    )

    print(
        rmse_pivot.to_string(
            float_format=lambda value: (
                f"{value:.6f}"
            )
        )
    )

    print()
    print("=" * 100)
    print(
        "BOUNDARY EDGES BY MODEL"
    )
    print("=" * 100)
    print()

    boundary_pivot = (
        by_model.pivot(
            index="Model",
            columns="Configuration",
            values="Boundary_Edges_Mean",
        )
    )

    print(
        boundary_pivot.to_string(
            float_format=lambda value: (
                f"{value:.1f}"
            )
        )
    )

    print()
    print(
        f"Raw results:\n"
        f"{raw_path}"
    )

    print()
    print(
        f"Summary:\n"
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()