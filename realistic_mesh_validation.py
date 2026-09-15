from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import open3d as o3d
import pandas as pd


# ---------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------

DATA_DIR = Path("data/realistic_jaw")
MULTIWAY_DIR = Path("output/realistic_multiway_ransac")
OUTPUT_DIR = Path("output/realistic_mesh_validation")

JAW_CONFIG = {
    "upper": {
        "reference_mesh": DATA_DIR / "upper_jaw.stl",
        "combined_cloud": (
            MULTIWAY_DIR / "upper" / "combined_cloud.ply"
        ),
    },
    "lower": {
        "reference_mesh": DATA_DIR / "lower_jaw.stl",
        "combined_cloud": (
            MULTIWAY_DIR / "lower" / "combined_cloud.ply"
        ),
    },
}

NORMAL_RADIUS_FACTOR = 3.0
NORMAL_MAX_NN = 50

POISSON_DEPTH = 9
POISSON_DENSITY_QUANTILE = 0.02

BPA_RADIUS_FACTORS = (
    1.0,
    1.5,
    2.0,
)

EVALUATION_SAMPLE_POINTS = 100_000

SEED = 42


# ---------------------------------------------------------------------
# Laden / Vorbereitung
# ---------------------------------------------------------------------

def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))

    if mesh.is_empty():
        raise RuntimeError(
            f"Mesh konnte nicht geladen werden: {path}"
        )

    return mesh


def load_cloud(path: Path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))

    if cloud.is_empty():
        raise RuntimeError(
            f"Punktwolke konnte nicht geladen werden: {path}"
        )

    return cloud


def mean_nearest_neighbor_distance(
    cloud: o3d.geometry.PointCloud,
) -> float:

    distances = cloud.compute_nearest_neighbor_distance()

    if len(distances) == 0:
        raise RuntimeError(
            "Keine Nearest-Neighbor-Distanzen verfügbar."
        )

    return float(np.mean(distances))


def prepare_cloud(
    cloud: o3d.geometry.PointCloud,
) -> tuple[o3d.geometry.PointCloud, float]:

    prepared = o3d.geometry.PointCloud(cloud)

    mean_distance = mean_nearest_neighbor_distance(
        prepared
    )

    normal_radius = (
        mean_distance * NORMAL_RADIUS_FACTOR
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
            "WARNUNG: Konsistente "
            "Normalenorientierung nicht vollständig möglich."
        )

    return prepared, mean_distance


# ---------------------------------------------------------------------
# Mesh Cleanup
# ---------------------------------------------------------------------

def basic_cleanup(
    mesh: o3d.geometry.TriangleMesh,
) -> o3d.geometry.TriangleMesh:

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    return mesh


# ---------------------------------------------------------------------
# Rekonstruktion
# ---------------------------------------------------------------------

def reconstruct_poisson(
    cloud: o3d.geometry.PointCloud,
) -> o3d.geometry.TriangleMesh:

    print(
        f"Poisson Reconstruction, depth={POISSON_DEPTH}"
    )

    mesh, densities = (
        o3d.geometry.TriangleMesh
        .create_from_point_cloud_poisson(
            cloud,
            depth=POISSON_DEPTH,
        )
    )

    densities = np.asarray(densities)

    threshold = np.quantile(
        densities,
        POISSON_DENSITY_QUANTILE,
    )

    remove_mask = (
        densities < threshold
    )

    mesh.remove_vertices_by_mask(
        remove_mask
    )

    basic_cleanup(mesh)

    mesh.compute_vertex_normals()

    return mesh


def reconstruct_bpa(
    cloud: o3d.geometry.PointCloud,
    mean_distance: float,
) -> o3d.geometry.TriangleMesh:

    radii = [
        mean_distance * factor
        for factor in BPA_RADIUS_FACTORS
    ]

    print(
        "Ball Pivoting radii: "
        + ", ".join(
            f"{radius:.6f}"
            for radius in radii
        )
    )

    mesh = (
        o3d.geometry.TriangleMesh
        .create_from_point_cloud_ball_pivoting(
            cloud,
            o3d.utility.DoubleVector(radii),
        )
    )

    basic_cleanup(mesh)

    mesh.compute_vertex_normals()

    return mesh


# ---------------------------------------------------------------------
# Distanzmetriken
# ---------------------------------------------------------------------

def distance_statistics(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> dict:

    distances = np.asarray(
        source.compute_point_cloud_distance(
            target
        )
    )

    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "rmse": float(
            np.sqrt(
                np.mean(
                    distances ** 2
                )
            )
        ),
        "p95": float(
            np.percentile(
                distances,
                95
            )
        ),
        "max": float(np.max(distances)),
    }


def mesh_distance_evaluation(
    reconstructed_mesh: o3d.geometry.TriangleMesh,
    reference_mesh: o3d.geometry.TriangleMesh,
) -> dict:

    # Gleiche Sampleanzahl und Seed für reproduzierbaren Vergleich.

    o3d.utility.random.seed(SEED)

    reference_sample = (
        reference_mesh.sample_points_uniformly(
            number_of_points=EVALUATION_SAMPLE_POINTS
        )
    )

    o3d.utility.random.seed(SEED)

    reconstruction_sample = (
        reconstructed_mesh.sample_points_uniformly(
            number_of_points=EVALUATION_SAMPLE_POINTS
        )
    )

    reference_to_reconstruction = distance_statistics(
        reference_sample,
        reconstruction_sample,
    )

    reconstruction_to_reference = distance_statistics(
        reconstruction_sample,
        reference_sample,
    )

    symmetric_mean = (
        reference_to_reconstruction["mean"]
        + reconstruction_to_reference["mean"]
    ) / 2.0

    symmetric_rmse = math.sqrt(
        (
            reference_to_reconstruction["rmse"] ** 2
            + reconstruction_to_reference["rmse"] ** 2
        )
        / 2.0
    )

    return {
        "reference_to_reconstruction":
            reference_to_reconstruction,
        "reconstruction_to_reference":
            reconstruction_to_reference,
        "symmetric_mean": float(
            symmetric_mean
        ),
        "symmetric_rmse": float(
            symmetric_rmse
        ),
    }


# ---------------------------------------------------------------------
# Topologie
# ---------------------------------------------------------------------

def topology_metrics(
    mesh: o3d.geometry.TriangleMesh,
) -> dict:

    triangles = len(mesh.triangles)
    vertices = len(mesh.vertices)

    triangle_clusters, cluster_n_triangles, _ = (
        mesh.cluster_connected_triangles()
    )

    cluster_n_triangles = np.asarray(
        cluster_n_triangles
    )

    number_components = int(
        len(cluster_n_triangles)
    )

    if number_components > 0:
        largest_component = int(
            cluster_n_triangles.max()
        )
    else:
        largest_component = 0

    largest_ratio = (
        largest_component / triangles
        if triangles > 0
        else 0.0
    )

    return {
        "vertices": vertices,
        "triangles": triangles,
        "connected_components": number_components,
        "largest_component_triangles":
            largest_component,
        "largest_component_ratio":
            largest_ratio,
        "edge_manifold_boundary_allowed":
            bool(
                mesh.is_edge_manifold(
                    allow_boundary_edges=True
                )
            ),
        "edge_manifold_closed":
            bool(
                mesh.is_edge_manifold(
                    allow_boundary_edges=False
                )
            ),
        "vertex_manifold":
            bool(mesh.is_vertex_manifold()),
        "self_intersecting":
            bool(mesh.is_self_intersecting()),
        "watertight":
            bool(mesh.is_watertight()),
        "orientable":
            bool(mesh.is_orientable()),
    }


# ---------------------------------------------------------------------
# Ein Kiefer
# ---------------------------------------------------------------------

def process_jaw(
    jaw_name: str,
    reference_mesh_path: Path,
    combined_cloud_path: Path,
) -> list[dict]:

    print()
    print("=" * 80)
    print(
        f"{jaw_name.upper()} JAW – "
        "MESH VALIDATION"
    )
    print("=" * 80)

    jaw_output = (
        OUTPUT_DIR / jaw_name
    )

    jaw_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_mesh = load_mesh(
        reference_mesh_path
    )

    reference_mesh.compute_vertex_normals()

    combined_cloud = load_cloud(
        combined_cloud_path
    )

    prepared_cloud, mean_distance = (
        prepare_cloud(
            combined_cloud
        )
    )

    print(
        f"Combined Cloud: "
        f"{len(combined_cloud.points):,} Punkte"
    )

    print(
        f"Mean NN Distance: "
        f"{mean_distance:.6f}"
    )

    reconstructors = {
        "Poisson": lambda: reconstruct_poisson(
            prepared_cloud
        ),
        "Ball Pivoting": lambda: reconstruct_bpa(
            prepared_cloud,
            mean_distance,
        ),
    }

    rows = []

    for method_name, reconstruct in reconstructors.items():

        print()
        print("-" * 80)
        print(method_name.upper())
        print("-" * 80)

        mesh = reconstruct()

        output_mesh_path = (
            jaw_output
            / (
                "poisson_mesh.ply"
                if method_name == "Poisson"
                else "ball_pivoting_mesh.ply"
            )
        )

        o3d.io.write_triangle_mesh(
            str(output_mesh_path),
            mesh,
        )

        topology = topology_metrics(mesh)

        geometry = mesh_distance_evaluation(
            reconstructed_mesh=mesh,
            reference_mesh=reference_mesh,
        )

        ref_to_rec = (
            geometry[
                "reference_to_reconstruction"
            ]
        )

        rec_to_ref = (
            geometry[
                "reconstruction_to_reference"
            ]
        )

        print(
            f"Vertices:   "
            f"{topology['vertices']:,}"
        )

        print(
            f"Triangles:  "
            f"{topology['triangles']:,}"
        )

        print(
            f"Components: "
            f"{topology['connected_components']}"
        )

        print(
            f"Self intersections: "
            f"{topology['self_intersecting']}"
        )

        print(
            f"Reference -> Reconstruction RMSE: "
            f"{ref_to_rec['rmse']:.6f}"
        )

        print(
            f"Reconstruction -> Reference RMSE: "
            f"{rec_to_ref['rmse']:.6f}"
        )

        print(
            f"Symmetric RMSE: "
            f"{geometry['symmetric_rmse']:.6f}"
        )

        print(
            f"Symmetric Mean: "
            f"{geometry['symmetric_mean']:.6f}"
        )

        evaluation = {
            "jaw": jaw_name,
            "method": method_name,
            "mean_nearest_neighbor_distance":
                mean_distance,
            "topology": topology,
            "geometry": geometry,
        }

        json_path = (
            jaw_output
            / (
                f"{method_name.lower().replace(' ', '_')}"
                "_evaluation.json"
            )
        )

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                evaluation,
                file,
                indent=2,
            )

        rows.append(
            {
                "jaw": jaw_name,
                "method": method_name,
                "vertices":
                    topology["vertices"],
                "triangles":
                    topology["triangles"],
                "connected_components":
                    topology[
                        "connected_components"
                    ],
                "largest_component_ratio":
                    topology[
                        "largest_component_ratio"
                    ],
                "self_intersecting":
                    topology[
                        "self_intersecting"
                    ],
                "watertight":
                    topology["watertight"],
                "orientable":
                    topology["orientable"],
                "reference_to_reconstruction_rmse":
                    ref_to_rec["rmse"],
                "reconstruction_to_reference_rmse":
                    rec_to_ref["rmse"],
                "symmetric_mean":
                    geometry["symmetric_mean"],
                "symmetric_rmse":
                    geometry["symmetric_rmse"],
            }
        )

    return rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for jaw_name, config in JAW_CONFIG.items():

        rows.extend(
            process_jaw(
                jaw_name=jaw_name,
                reference_mesh_path=(
                    config["reference_mesh"]
                ),
                combined_cloud_path=(
                    config["combined_cloud"]
                ),
            )
        )

    dataframe = pd.DataFrame(rows)

    output_csv = (
        OUTPUT_DIR
        / "realistic_mesh_comparison.csv"
    )

    dataframe.to_csv(
        output_csv,
        index=False,
    )

    print()
    print("=" * 80)
    print("GESAMTVERGLEICH")
    print("=" * 80)

    display_columns = [
        "jaw",
        "method",
        "triangles",
        "connected_components",
        "self_intersecting",
        "symmetric_mean",
        "symmetric_rmse",
    ]

    print(
        dataframe[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print(
        f"Gespeichert: {output_csv}"
    )


if __name__ == "__main__":
    main()