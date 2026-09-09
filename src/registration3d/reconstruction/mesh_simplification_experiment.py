from pathlib import Path
import csv
import time

import open3d as o3d

from registration3d.data_loader import (
    PointCloudLoader,
)
from registration3d.reconstruction.mesh_config import (
    MeshConfig,
)
from registration3d.reconstruction.mesh_evaluator import (
    MeshEvaluator,
)


TARGET_TRIANGLE_COUNTS = (
    30_000,
    20_000,
    10_000,
    5_000,
)


def evaluate_mesh(
    name: str,
    mesh: o3d.geometry.TriangleMesh,
    reference_cloud: o3d.geometry.PointCloud,
    evaluator: MeshEvaluator,
) -> tuple:

    print()
    print("=" * 80)
    print(f"EVALUATION: {name}")
    print("=" * 80)

    # Gleicher Seed für alle Varianten,
    # damit die Sampling-basierte Evaluation
    # möglichst vergleichbar bleibt.
    o3d.utility.random.seed(
        42
    )

    evaluation = (
        evaluator.evaluate(
            mesh=mesh,
            reference_cloud=reference_cloud,
        )
    )

    return evaluation


def main() -> None:

    # =========================================================
    # Projektpfade
    # =========================================================

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    input_mesh_path = (
        project_root
        / "output"
        / "mesh"
        / "ball_pivoting"
        / "final_candidate_mesh.ply"
    )

    reference_cloud_path = (
        project_root
        / "output"
        / "multiway"
        / "fgr_baseline"
        / "combined_cloud.ply"
    )

    output_directory = (
        project_root
        / "output"
        / "mesh"
        / "simplification_experiments"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # Input prüfen
    # =========================================================

    if not input_mesh_path.exists():

        raise FileNotFoundError(
            "Finales BPA-Kandidaten-Mesh fehlt:\n"
            f"{input_mesh_path}"
        )

    if not reference_cloud_path.exists():

        raise FileNotFoundError(
            "Referenzpunktwolke fehlt:\n"
            f"{reference_cloud_path}"
        )

    # =========================================================
    # Daten laden
    # =========================================================

    print()
    print("=" * 80)
    print("MESH SIMPLIFICATION EXPERIMENT")
    print("=" * 80)

    print()
    print(
        f"Mesh: {input_mesh_path}"
    )

    original_mesh = (
        o3d.io.read_triangle_mesh(
            str(input_mesh_path)
        )
    )

    if len(original_mesh.vertices) == 0:

        raise ValueError(
            "Mesh enthält keine Vertices."
        )

    if len(original_mesh.triangles) == 0:

        raise ValueError(
            "Mesh enthält keine Dreiecke."
        )

    original_mesh.compute_triangle_normals()
    original_mesh.compute_vertex_normals()

    reference_cloud = (
        PointCloudLoader.load(
            reference_cloud_path
        )
    )

    print()
    print(
        f"Original Vertices: "
        f"{len(original_mesh.vertices):,}"
    )

    print(
        f"Original Dreiecke: "
        f"{len(original_mesh.triangles):,}"
    )

    # =========================================================
    # Evaluation Config
    # =========================================================

    config = (
        MeshConfig(
            normal_radius_factor=3.0,
            normal_max_nn=50,

            poisson_depth=9,
            poisson_density_quantile=0.02,

            ball_pivoting_radius_factors=(
                1.0,
                1.5,
                2.0,
            ),

            minimum_component_triangles=250,

            target_triangle_count=20_000,

            evaluation_sample_points=100_000,
        )
    )

    evaluator = (
        MeshEvaluator(
            config
        )
    )

    results = []

    original_triangle_count = (
        len(original_mesh.triangles)
    )

    # =========================================================
    # Originalmesh als Referenzwert evaluieren
    # =========================================================

    original_evaluation = (
        evaluate_mesh(
            name="Original",
            mesh=original_mesh,
            reference_cloud=reference_cloud,
            evaluator=evaluator,
        )
    )

    results.append(
        {
            "Variant": "Original",
            "Target Triangles": (
                original_triangle_count
            ),
            "Vertices": (
                original_evaluation.vertices
            ),
            "Triangles": (
                original_evaluation.triangles
            ),
            "Triangle Ratio": 1.0,
            "Reduction Percent": 0.0,
            "Runtime [s]": 0.0,
            "Connected Components": (
                original_evaluation
                .connected_components
            ),
            "Edge Manifold": (
                original_evaluation
                .edge_manifold_with_boundary
            ),
            "Vertex Manifold": (
                original_evaluation
                .vertex_manifold
            ),
            "Self Intersecting": (
                original_evaluation
                .self_intersecting
            ),
            "Watertight": (
                original_evaluation
                .watertight
            ),
            "Orientable": (
                original_evaluation
                .orientable
            ),
            "Cloud to Mesh RMSE": (
                original_evaluation
                .cloud_to_mesh_rmse
            ),
            "Mesh to Cloud RMSE": (
                original_evaluation
                .mesh_to_cloud_rmse
            ),
            "Symmetric Mean": (
                original_evaluation
                .symmetric_mean_distance
            ),
            "Symmetric RMSE": (
                original_evaluation
                .symmetric_rmse
            ),
        }
    )

    # =========================================================
    # Vereinfachungsvarianten
    # =========================================================

    for target_count in TARGET_TRIANGLE_COUNTS:

        print()
        print()
        print("#" * 80)
        print(
            f"SIMPLIFICATION TARGET: "
            f"{target_count:,}"
        )
        print("#" * 80)

        start_time = (
            time.perf_counter()
        )

        simplified_mesh = (
            original_mesh
            .simplify_quadric_decimation(
                target_number_of_triangles=(
                    target_count
                )
            )
        )

        runtime = (
            time.perf_counter()
            - start_time
        )

        # -----------------------------------------------------
        # Einfaches Cleanup nach Decimation
        # -----------------------------------------------------

        simplified_mesh.remove_duplicated_vertices()

        simplified_mesh.remove_duplicated_triangles()

        simplified_mesh.remove_degenerate_triangles()

        simplified_mesh.remove_unreferenced_vertices()

        simplified_mesh.compute_triangle_normals()

        simplified_mesh.compute_vertex_normals()

        actual_triangle_count = (
            len(
                simplified_mesh.triangles
            )
        )

        triangle_ratio = (
            actual_triangle_count
            / original_triangle_count
        )

        reduction_percent = (
            (
                1.0
                - triangle_ratio
            )
            * 100.0
        )

        print()
        print(
            f"Vertices: "
            f"{len(simplified_mesh.vertices):,}"
        )

        print(
            f"Dreiecke: "
            f"{actual_triangle_count:,}"
        )

        print(
            f"Reduktion: "
            f"{reduction_percent:.2f} %"
        )

        print(
            f"Laufzeit: "
            f"{runtime:.6f} s"
        )

        # -----------------------------------------------------
        # Mesh speichern
        # -----------------------------------------------------

        variant_directory = (
            output_directory
            / f"target_{target_count}"
        )

        variant_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        mesh_path = (
            variant_directory
            / "simplified_mesh.ply"
        )

        success = (
            o3d.io.write_triangle_mesh(
                str(mesh_path),
                simplified_mesh,
            )
        )

        if not success:

            raise RuntimeError(
                "Vereinfachtes Mesh konnte "
                "nicht gespeichert werden."
            )

        # -----------------------------------------------------
        # Evaluation
        # -----------------------------------------------------

        evaluation = (
            evaluate_mesh(
                name=(
                    f"Target {target_count:,}"
                ),
                mesh=simplified_mesh,
                reference_cloud=(
                    reference_cloud
                ),
                evaluator=evaluator,
            )
        )

        evaluation_path = (
            variant_directory
            / "evaluation.json"
        )

        evaluator.save_json(
            result=evaluation,
            path=evaluation_path,
        )

        # -----------------------------------------------------
        # Ergebnis speichern
        # -----------------------------------------------------

        results.append(
            {
                "Variant": (
                    f"Target {target_count}"
                ),
                "Target Triangles": (
                    target_count
                ),
                "Vertices": (
                    evaluation.vertices
                ),
                "Triangles": (
                    evaluation.triangles
                ),
                "Triangle Ratio": (
                    triangle_ratio
                ),
                "Reduction Percent": (
                    reduction_percent
                ),
                "Runtime [s]": (
                    runtime
                ),
                "Connected Components": (
                    evaluation
                    .connected_components
                ),
                "Edge Manifold": (
                    evaluation
                    .edge_manifold_with_boundary
                ),
                "Vertex Manifold": (
                    evaluation
                    .vertex_manifold
                ),
                "Self Intersecting": (
                    evaluation
                    .self_intersecting
                ),
                "Watertight": (
                    evaluation
                    .watertight
                ),
                "Orientable": (
                    evaluation
                    .orientable
                ),
                "Cloud to Mesh RMSE": (
                    evaluation
                    .cloud_to_mesh_rmse
                ),
                "Mesh to Cloud RMSE": (
                    evaluation
                    .mesh_to_cloud_rmse
                ),
                "Symmetric Mean": (
                    evaluation
                    .symmetric_mean_distance
                ),
                "Symmetric RMSE": (
                    evaluation
                    .symmetric_rmse
                ),
            }
        )

    # =========================================================
    # CSV speichern
    # =========================================================

    csv_path = (
        output_directory
        / "simplification_results.csv"
    )

    fieldnames = list(
        results[0].keys()
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = (
            csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # =========================================================
    # Ergebnistabelle
    # =========================================================

    print()
    print()
    print("=" * 122)
    print("MESH SIMPLIFICATION RESULTS")
    print("=" * 122)

    print()

    print(
        f"{'Variant':<17}"
        f"{'Triangles':>12}"
        f"{'Reduction':>12}"
        f"{'Components':>12}"
        f"{'C->M RMSE':>12}"
        f"{'M->C RMSE':>12}"
        f"{'Sym RMSE':>12}"
        f"{'SelfInt':>10}"
        f"{'V-Manifold':>12}"
    )

    print(
        "-" * 111
    )

    for result in results:

        print(
            f"{result['Variant']:<17}"
            f"{result['Triangles']:>12,}"
            f"{result['Reduction Percent']:>11.2f}%"
            f"{result['Connected Components']:>12}"
            f"{result['Cloud to Mesh RMSE']:>12.6f}"
            f"{result['Mesh to Cloud RMSE']:>12.6f}"
            f"{result['Symmetric RMSE']:>12.6f}"
            f"{str(result['Self Intersecting']):>10}"
            f"{str(result['Vertex Manifold']):>12}"
        )

    print()
    print(
        f"CSV gespeichert:"
    )

    print(
        f"  {csv_path}"
    )

    print()
    print("=" * 80)
    print("SIMPLIFICATION EXPERIMENT ABGESCHLOSSEN")
    print("=" * 80)


if __name__ == "__main__":
    main()