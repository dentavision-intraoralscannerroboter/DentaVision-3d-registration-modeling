from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd

from registration3d.realistic_dataset_registry import (
    DENTAL_MODELS,
    JAW_NAMES,
    get_stl_path,
)


REFERENCE_SAMPLE_POINTS = 100_000
RECONSTRUCTION_SAMPLE_POINTS = 100_000

POISSON_DEPTH = 9
POISSON_DENSITY_QUANTILE = 0.02

RANDOM_SEED = 42


def load_cloud(
    path: Path,
) -> o3d.geometry.PointCloud:

    cloud = o3d.io.read_point_cloud(
        str(path)
    )

    if cloud.is_empty():
        raise RuntimeError(
            f"Could not load cloud:\n{path}"
        )

    return cloud


def load_mesh(
    path: Path,
) -> o3d.geometry.TriangleMesh:

    mesh = o3d.io.read_triangle_mesh(
        str(path)
    )

    if mesh.is_empty():
        raise RuntimeError(
            f"Could not load mesh:\n{path}"
        )

    return mesh


def sample_mesh(
    mesh: o3d.geometry.TriangleMesh,
    points: int,
) -> o3d.geometry.PointCloud:

    o3d.utility.random.seed(
        RANDOM_SEED
    )

    return mesh.sample_points_uniformly(
        number_of_points=points
    )


def distance_stats(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> dict:

    distances = np.asarray(
        source.compute_point_cloud_distance(
            target
        )
    )

    return {
        "mean":
            float(
                np.mean(
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
    }


def connected_components(
    mesh: o3d.geometry.TriangleMesh,
) -> tuple[int, float]:

    if len(
        mesh.triangles
    ) == 0:
        return (
            0,
            0.0,
        )

    _, counts, _ = (
        mesh.cluster_connected_triangles()
    )

    counts = np.asarray(
        counts
    )

    return (
        len(
            counts
        ),

        float(
            np.max(
                counts
            )
            / len(
                mesh.triangles
            )
        ),
    )


def count_boundary_edges(
    mesh: o3d.geometry.TriangleMesh,
) -> int:

    triangles = np.asarray(
        mesh.triangles,
        dtype=np.int64,
    )

    edges = np.vstack(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
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


def evaluate(
    reference_mesh:
        o3d.geometry.TriangleMesh,

    reconstruction:
        o3d.geometry.TriangleMesh,

) -> dict:

    reference_samples = (
        sample_mesh(
            reference_mesh,
            REFERENCE_SAMPLE_POINTS,
        )
    )

    reconstruction_samples = (
        sample_mesh(
            reconstruction,
            RECONSTRUCTION_SAMPLE_POINTS,
        )
    )

    ref_to_rec = (
        distance_stats(
            reference_samples,
            reconstruction_samples,
        )
    )

    rec_to_ref = (
        distance_stats(
            reconstruction_samples,
            reference_samples,
        )
    )

    symmetric_mean = (
        0.5
        * (
            ref_to_rec["mean"]
            +
            rec_to_ref["mean"]
        )
    )

    symmetric_rmse = float(
        np.sqrt(
            0.5
            * (
                ref_to_rec["rmse"] ** 2
                +
                rec_to_ref["rmse"] ** 2
            )
        )
    )

    components, largest_fraction = (
        connected_components(
            reconstruction
        )
    )

    reference_area = float(
        reference_mesh
        .get_surface_area()
    )

    reconstruction_area = float(
        reconstruction
        .get_surface_area()
    )

    return {
        "Vertices":
            len(
                reconstruction.vertices
            ),

        "Triangles":
            len(
                reconstruction.triangles
            ),

        "Connected Components":
            components,

        "Largest Component Fraction":
            largest_fraction,

        "Boundary Edges":
            count_boundary_edges(
                reconstruction
            ),

        "Self Intersecting":
            bool(
                reconstruction
                .is_self_intersecting()
            ),

        "Area Ratio":
            (
                reconstruction_area
                / reference_area
            ),

        "Reference to Reconstruction Mean":
            ref_to_rec["mean"],

        "Reference to Reconstruction RMSE":
            ref_to_rec["rmse"],

        "Reconstruction to Reference Mean":
            rec_to_ref["mean"],

        "Reconstruction to Reference RMSE":
            rec_to_ref["rmse"],

        "Symmetric Mean":
            symmetric_mean,

        "Symmetric RMSE":
            symmetric_rmse,
    }


def reconstruct_poisson(
    cloud: o3d.geometry.PointCloud,
) -> o3d.geometry.TriangleMesh:

    working = o3d.geometry.PointCloud(
        cloud
    )

    distances = np.asarray(
        working
        .compute_nearest_neighbor_distance()
    )

    mean_nn = float(
        np.mean(
            distances
        )
    )

    working.estimate_normals(
        search_param=(
            o3d.geometry
            .KDTreeSearchParamHybrid(
                radius=(
                    mean_nn
                    * 3.0
                ),
                max_nn=50,
            )
        )
    )

    working.orient_normals_consistent_tangent_plane(
        30
    )

    mesh, densities = (
        o3d.geometry.TriangleMesh
        .create_from_point_cloud_poisson(
            working,
            depth=POISSON_DEPTH,
        )
    )

    densities = np.asarray(
        densities
    )

    threshold = np.quantile(
        densities,
        POISSON_DENSITY_QUANTILE,
    )

    remove_mask = (
        densities
        < threshold
    )

    mesh.remove_vertices_by_mask(
        remove_mask
    )

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()

    return mesh


def main() -> None:

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    output_root = (
        project_root
        / "output"
        / "realistic_model_suite_mesh_validation"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for model in DENTAL_MODELS:

        for jaw in JAW_NAMES:

            print()
            print("=" * 100)
            print(
                f"{model.model_id} / "
                f"{jaw.upper()}"
            )
            print("=" * 100)

            reference_mesh = (
                load_mesh(
                    get_stl_path(
                        project_root=project_root,
                        model=model,
                        jaw=jaw,
                    )
                )
            )

            combined_path = (
                project_root
                / "output"
                / "realistic_model_suite_multiway_ransac"
                / model.model_id
                / jaw
                / "combined_cloud.ply"
            )

            combined_cloud = (
                load_cloud(
                    combined_path
                )
            )

            # ------------------------------------------------
            # Final BPA = configuration B
            # ------------------------------------------------

            bpa_path = (
                project_root
                / "output"
                / "realistic_model_suite_bpa_parameter_study"
                / model.model_id
                / jaw
                / "B_baseline.ply"
            )

            bpa_mesh = (
                load_mesh(
                    bpa_path
                )
            )

            # ------------------------------------------------
            # Poisson
            # ------------------------------------------------

            print(
                "Reconstructing Poisson..."
            )

            poisson_mesh = (
                reconstruct_poisson(
                    combined_cloud
                )
            )

            jaw_output = (
                output_root
                / model.model_id
                / jaw
            )

            jaw_output.mkdir(
                parents=True,
                exist_ok=True,
            )

            poisson_path = (
                jaw_output
                / "poisson.ply"
            )

            o3d.io.write_triangle_mesh(
                str(
                    poisson_path
                ),
                poisson_mesh,
            )

            # ------------------------------------------------
            # Evaluate both
            # ------------------------------------------------

            for method, mesh in (
                (
                    "Poisson",
                    poisson_mesh,
                ),
                (
                    "BPA",
                    bpa_mesh,
                ),
            ):

                print(
                    f"Evaluating {method}..."
                )

                metrics = (
                    evaluate(
                        reference_mesh=(
                            reference_mesh
                        ),
                        reconstruction=(
                            mesh
                        ),
                    )
                )

                row = {
                    "Model":
                        model.model_id,

                    "Jaw":
                        jaw,

                    "Method":
                        method,
                }

                row.update(
                    metrics
                )

                rows.append(
                    row
                )

                print(
                    f"  Sym RMSE: "
                    f"{metrics['Symmetric RMSE']:.6f}"
                )

                print(
                    f"  Components: "
                    f"{metrics['Connected Components']}"
                )

                print(
                    f"  Boundary: "
                    f"{metrics['Boundary Edges']}"
                )

                print(
                    f"  Self-intersecting: "
                    f"{metrics['Self Intersecting']}"
                )

    raw = pd.DataFrame(
        rows
    )

    raw_path = (
        output_root
        / "mesh_validation_raw.csv"
    )

    raw.to_csv(
        raw_path,
        index=False,
    )

    summary = (
        raw.groupby(
            "Method",
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

            Ref_to_Rec_RMSE_Mean=(
                "Reference to Reconstruction RMSE",
                "mean",
            ),

            Rec_to_Ref_RMSE_Mean=(
                "Reconstruction to Reference RMSE",
                "mean",
            ),

            Boundary_Edges_Mean=(
                "Boundary Edges",
                "mean",
            ),

            Components_Mean=(
                "Connected Components",
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

    summary_path = (
        output_root
        / "mesh_validation_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    by_model = (
        raw.groupby(
            [
                "Model",
                "Method",
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

            Self_Intersecting_Count=(
                "Self Intersecting",
                "sum",
            ),
        )
    )

    by_model_path = (
        output_root
        / "mesh_validation_by_model.csv"
    )

    by_model.to_csv(
        by_model_path,
        index=False,
    )

    print()
    print("=" * 100)
    print(
        "AGGREGATE MESH VALIDATION"
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

    pivot = (
        by_model.pivot(
            index="Model",
            columns="Method",
            values="Symmetric_RMSE_Mean",
        )
    )

    print(
        pivot.to_string(
            float_format=lambda value: (
                f"{value:.6f}"
            )
        )
    )

    print()
    print(
        f"Raw:\n{raw_path}"
    )

    print()
    print(
        f"Summary:\n{summary_path}"
    )


if __name__ == "__main__":
    main()