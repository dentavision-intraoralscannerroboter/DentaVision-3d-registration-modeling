from __future__ import annotations

from collections import Counter
from pathlib import Path
import math

import numpy as np
import open3d as o3d
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/realistic_jaw")

MULTIWAY_DIR = Path(
    "output/realistic_multiway_ransac"
)

OUTPUT_DIR = Path(
    "output/realistic_bpa_parameter_study"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


JAW_CONFIG = {
    "upper": {
        "reference_mesh": (
            DATA_DIR / "upper_jaw.stl"
        ),
        "combined_cloud": (
            MULTIWAY_DIR
            / "upper"
            / "combined_cloud.ply"
        ),
    },
    "lower": {
        "reference_mesh": (
            DATA_DIR / "lower_jaw.stl"
        ),
        "combined_cloud": (
            MULTIWAY_DIR
            / "lower"
            / "combined_cloud.ply"
        ),
    },
}


# ============================================================
# BPA CONFIGURATIONS
# ============================================================

BPA_CONFIGS = {
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


NORMAL_RADIUS_FACTOR = 3.0
NORMAL_MAX_NN = 50

EVALUATION_SAMPLE_POINTS = 100_000

SEED = 42


# ============================================================
# LOADING
# ============================================================

def load_cloud(
    path: Path,
) -> o3d.geometry.PointCloud:

    cloud = o3d.io.read_point_cloud(
        str(path)
    )

    if cloud.is_empty():
        raise RuntimeError(
            f"Could not load cloud: {path}"
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
            f"Could not load mesh: {path}"
        )

    return mesh


# ============================================================
# POINT CLOUD PREPARATION
# ============================================================

def mean_nearest_neighbor_distance(
    cloud: o3d.geometry.PointCloud,
) -> float:

    distances = np.asarray(
        cloud.compute_nearest_neighbor_distance()
    )

    return float(
        np.mean(distances)
    )


def prepare_cloud(
    cloud: o3d.geometry.PointCloud,
):

    prepared = o3d.geometry.PointCloud(
        cloud
    )

    mean_distance = (
        mean_nearest_neighbor_distance(
            prepared
        )
    )

    normal_radius = (
        mean_distance
        * NORMAL_RADIUS_FACTOR
    )

    prepared.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=NORMAL_MAX_NN,
        )
    )

    try:

        prepared.orient_normals_consistent_tangent_plane(
            30
        )

    except RuntimeError:

        print(
            "Warning: consistent normal "
            "orientation failed."
        )

    return (
        prepared,
        mean_distance,
    )


# ============================================================
# CLEANUP
# ============================================================

def cleanup_mesh(
    mesh: o3d.geometry.TriangleMesh,
):

    mesh.remove_duplicated_vertices()

    mesh.remove_duplicated_triangles()

    mesh.remove_degenerate_triangles()

    mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()

    return mesh


# ============================================================
# BPA
# ============================================================

def reconstruct_bpa(
    cloud: o3d.geometry.PointCloud,
    mean_distance: float,
    radius_factors,
):

    radii = [
        mean_distance * factor
        for factor in radius_factors
    ]

    mesh = (
        o3d.geometry.TriangleMesh
        .create_from_point_cloud_ball_pivoting(
            cloud,
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
# EDGE ANALYSIS
# ============================================================

def edge_statistics(
    mesh: o3d.geometry.TriangleMesh,
):

    triangles = np.asarray(
        mesh.triangles
    )

    edge_counter = Counter()

    for triangle in triangles:

        a, b, c = map(
            int,
            triangle
        )

        edges = [
            tuple(
                sorted(
                    (
                        a,
                        b,
                    )
                )
            ),
            tuple(
                sorted(
                    (
                        b,
                        c,
                    )
                )
            ),
            tuple(
                sorted(
                    (
                        c,
                        a,
                    )
                )
            ),
        ]

        for edge in edges:

            edge_counter[
                edge
            ] += 1

    boundary_edges = sum(
        1
        for count
        in edge_counter.values()
        if count == 1
    )

    manifold_edges = sum(
        1
        for count
        in edge_counter.values()
        if count == 2
    )

    non_manifold_edges = sum(
        1
        for count
        in edge_counter.values()
        if count > 2
    )

    return {
        "boundary_edges":
            boundary_edges,
        "manifold_edges":
            manifold_edges,
        "non_manifold_edges":
            non_manifold_edges,
    }


# ============================================================
# COMPONENT ANALYSIS
# ============================================================

def component_statistics(
    mesh: o3d.geometry.TriangleMesh,
):

    _, cluster_counts, _ = (
        mesh.cluster_connected_triangles()
    )

    counts = np.asarray(
        cluster_counts
    )

    number_components = (
        len(counts)
    )

    if number_components == 0:

        return {
            "components": 0,
            "largest_component_triangles": 0,
            "largest_component_ratio": 0.0,
        }

    largest = int(
        counts.max()
    )

    total_triangles = len(
        mesh.triangles
    )

    ratio = (
        largest
        / total_triangles
        if total_triangles > 0
        else 0.0
    )

    return {
        "components":
            int(number_components),

        "largest_component_triangles":
            largest,

        "largest_component_ratio":
            float(ratio),
    }


# ============================================================
# DISTANCE EVALUATION
# ============================================================

def point_cloud_distance_stats(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
):

    distances = np.asarray(
        source.compute_point_cloud_distance(
            target
        )
    )

    return {
        "mean":
            float(
                np.mean(distances)
            ),

        "median":
            float(
                np.median(distances)
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
                    95
                )
            ),

        "max":
            float(
                np.max(distances)
            ),
    }


def evaluate_geometry(
    reconstruction:
        o3d.geometry.TriangleMesh,

    reference:
        o3d.geometry.TriangleMesh,
):

    o3d.utility.random.seed(
        SEED
    )

    reference_points = (
        reference
        .sample_points_uniformly(
            number_of_points=
                EVALUATION_SAMPLE_POINTS
        )
    )

    o3d.utility.random.seed(
        SEED
    )

    reconstruction_points = (
        reconstruction
        .sample_points_uniformly(
            number_of_points=
                EVALUATION_SAMPLE_POINTS
        )
    )

    ref_to_rec = (
        point_cloud_distance_stats(
            reference_points,
            reconstruction_points,
        )
    )

    rec_to_ref = (
        point_cloud_distance_stats(
            reconstruction_points,
            reference_points,
        )
    )

    symmetric_mean = (
        ref_to_rec["mean"]
        + rec_to_ref["mean"]
    ) / 2.0

    symmetric_rmse = math.sqrt(
        (
            ref_to_rec["rmse"] ** 2
            + rec_to_ref["rmse"] ** 2
        )
        / 2.0
    )

    return {
        "reference_to_reconstruction":
            ref_to_rec,

        "reconstruction_to_reference":
            rec_to_ref,

        "symmetric_mean":
            symmetric_mean,

        "symmetric_rmse":
            symmetric_rmse,
    }


# ============================================================
# SURFACE AREA
# ============================================================

def surface_area_metrics(
    reconstructed:
        o3d.geometry.TriangleMesh,

    reference:
        o3d.geometry.TriangleMesh,
):

    reconstruction_area = float(
        reconstructed.get_surface_area()
    )

    reference_area = float(
        reference.get_surface_area()
    )

    area_ratio = (
        reconstruction_area
        / reference_area
        if reference_area > 0
        else 0.0
    )

    return {
        "reconstruction_area":
            reconstruction_area,

        "reference_area":
            reference_area,

        "surface_area_ratio":
            area_ratio,
    }


# ============================================================
# PROCESS ONE JAW
# ============================================================

def process_jaw(
    jaw_name: str,
    reference_path: Path,
    cloud_path: Path,
):

    print()
    print("=" * 90)
    print(
        f"{jaw_name.upper()} JAW – "
        "REALISTIC BPA PARAMETER STUDY"
    )
    print("=" * 90)

    reference_mesh = load_mesh(
        reference_path
    )

    reference_mesh.compute_vertex_normals()

    combined_cloud = load_cloud(
        cloud_path
    )

    prepared_cloud, mean_distance = (
        prepare_cloud(
            combined_cloud
        )
    )

    print(
        f"Combined cloud points: "
        f"{len(combined_cloud.points):,}"
    )

    print(
        f"Mean nearest-neighbor distance: "
        f"{mean_distance:.6f}"
    )

    print(
        f"Reference surface area: "
        f"{reference_mesh.get_surface_area():.6f}"
    )

    jaw_output = (
        OUTPUT_DIR
        / jaw_name
    )

    jaw_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for (
        config_name,
        radius_factors,
    ) in BPA_CONFIGS.items():

        print()
        print("-" * 90)

        print(
            f"{config_name}: "
            f"{radius_factors}"
        )

        mesh, radii = (
            reconstruct_bpa(
                prepared_cloud,
                mean_distance,
                radius_factors,
            )
        )

        mesh_path = (
            jaw_output
            / f"{config_name}.ply"
        )

        o3d.io.write_triangle_mesh(
            str(mesh_path),
            mesh,
        )

        geometry = (
            evaluate_geometry(
                mesh,
                reference_mesh,
            )
        )

        edges = (
            edge_statistics(
                mesh
            )
        )

        components = (
            component_statistics(
                mesh
            )
        )

        area = (
            surface_area_metrics(
                mesh,
                reference_mesh,
            )
        )

        ref_rec = (
            geometry[
                "reference_to_reconstruction"
            ]
        )

        rec_ref = (
            geometry[
                "reconstruction_to_reference"
            ]
        )

        self_intersecting = bool(
            mesh.is_self_intersecting()
        )

        print(
            "Radii: "
            + ", ".join(
                f"{radius:.6f}"
                for radius in radii
            )
        )

        print(
            f"Vertices: "
            f"{len(mesh.vertices):,}"
        )

        print(
            f"Triangles: "
            f"{len(mesh.triangles):,}"
        )

        print(
            f"Components: "
            f"{components['components']}"
        )

        print(
            f"Largest component: "
            f"{components['largest_component_ratio'] * 100:.3f}%"
        )

        print(
            f"Boundary edges: "
            f"{edges['boundary_edges']:,}"
        )

        print(
            f"Non-manifold edges: "
            f"{edges['non_manifold_edges']:,}"
        )

        print(
            f"Surface area ratio: "
            f"{area['surface_area_ratio']:.4f}"
        )

        print(
            f"Ref -> Reconstruction RMSE: "
            f"{ref_rec['rmse']:.6f}"
        )

        print(
            f"Ref -> Reconstruction P95: "
            f"{ref_rec['p95']:.6f}"
        )

        print(
            f"Reconstruction -> Ref RMSE: "
            f"{rec_ref['rmse']:.6f}"
        )

        print(
            f"Symmetric RMSE: "
            f"{geometry['symmetric_rmse']:.6f}"
        )

        print(
            f"Self intersections: "
            f"{self_intersecting}"
        )

        rows.append(
            {
                "jaw":
                    jaw_name,

                "configuration":
                    config_name,

                "radius_factors":
                    str(
                        radius_factors
                    ),

                "radius_1":
                    radii[0],

                "radius_2":
                    radii[1],

                "radius_3":
                    radii[2],

                "vertices":
                    len(
                        mesh.vertices
                    ),

                "triangles":
                    len(
                        mesh.triangles
                    ),

                "components":
                    components[
                        "components"
                    ],

                "largest_component_ratio":
                    components[
                        "largest_component_ratio"
                    ],

                "boundary_edges":
                    edges[
                        "boundary_edges"
                    ],

                "non_manifold_edges":
                    edges[
                        "non_manifold_edges"
                    ],

                "surface_area":
                    area[
                        "reconstruction_area"
                    ],

                "reference_surface_area":
                    area[
                        "reference_area"
                    ],

                "surface_area_ratio":
                    area[
                        "surface_area_ratio"
                    ],

                "ref_to_rec_rmse":
                    ref_rec[
                        "rmse"
                    ],

                "ref_to_rec_p95":
                    ref_rec[
                        "p95"
                    ],

                "rec_to_ref_rmse":
                    rec_ref[
                        "rmse"
                    ],

                "symmetric_mean":
                    geometry[
                        "symmetric_mean"
                    ],

                "symmetric_rmse":
                    geometry[
                        "symmetric_rmse"
                    ],

                "self_intersecting":
                    self_intersecting,
            }
        )

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    all_rows = []

    for (
        jaw_name,
        config,
    ) in JAW_CONFIG.items():

        rows = process_jaw(
            jaw_name=jaw_name,

            reference_path=(
                config[
                    "reference_mesh"
                ]
            ),

            cloud_path=(
                config[
                    "combined_cloud"
                ]
            ),
        )

        all_rows.extend(
            rows
        )

    dataframe = pd.DataFrame(
        all_rows
    )

    csv_path = (
        OUTPUT_DIR
        / "realistic_bpa_parameter_results.csv"
    )

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    print()
    print("=" * 100)
    print("GESAMTVERGLEICH")
    print("=" * 100)

    columns = [
        "jaw",
        "configuration",
        "triangles",
        "components",
        "boundary_edges",
        "surface_area_ratio",
        "ref_to_rec_rmse",
        "ref_to_rec_p95",
        "rec_to_ref_rmse",
        "symmetric_rmse",
        "self_intersecting",
    ]

    print(
        dataframe[
            columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()
    print(
        f"Saved: {csv_path}"
    )


if __name__ == "__main__":
    main()