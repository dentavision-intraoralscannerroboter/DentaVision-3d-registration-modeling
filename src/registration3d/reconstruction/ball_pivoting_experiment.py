from pathlib import Path
import csv

import open3d as o3d

from registration3d.data_loader import (
    PointCloudLoader,
)
from registration3d.reconstruction.mesh_cleanup import (
    MeshCleaner,
)
from registration3d.reconstruction.mesh_config import (
    MeshConfig,
)
from registration3d.reconstruction.mesh_evaluator import (
    MeshEvaluator,
)
from registration3d.reconstruction.mesh_reconstruction import (
    MeshReconstructor,
)


# =============================================================
# Zu untersuchende BPA-Konfigurationen
#
# Alle Werte sind Faktoren des mittleren Punktabstands.
# =============================================================

RADIUS_CONFIGURATIONS = {
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


def main() -> None:

    # =========================================================
    # Projektpfade
    # =========================================================

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    input_path = (
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
        / "ball_pivoting_experiments"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # Input prüfen
    # =========================================================

    if not input_path.exists():

        raise FileNotFoundError(
            "Registrierte Gesamtpunktwolke "
            "wurde nicht gefunden:\n"
            f"{input_path}"
        )

    # =========================================================
    # Punktwolke laden
    # =========================================================

    print()
    print("=" * 80)
    print("BALL PIVOTING PARAMETER EXPERIMENT")
    print("=" * 80)

    print()
    print(
        f"Eingabe: "
        f"{input_path}"
    )

    cloud = (
        PointCloudLoader.load(
            input_path
        )
    )

    # =========================================================
    # Gemeinsame Basis-Konfiguration
    #
    # Wichtig:
    # Für alle BPA-Versuche verwenden wir dasselbe
    # Preprocessing.
    # =========================================================

    base_config = (
        MeshConfig(
            normal_radius_factor=3.0,
            normal_max_nn=50,

            poisson_depth=9,
            poisson_density_quantile=0.02,

            ball_pivoting_radius_factors=(
                1.5,
                2.0,
                3.0,
            ),

            minimum_component_triangles=100,

            target_triangle_count=100_000,

            evaluation_sample_points=100_000,
        )
    )

    base_reconstructor = (
        MeshReconstructor(
            base_config
        )
    )

    # =========================================================
    # Punktwolke genau EINMAL vorbereiten
    #
    # Dadurch unterscheiden sich die Experimente wirklich nur
    # durch die Ball-Pivoting-Radien.
    # =========================================================

    prepared_cloud = (
        base_reconstructor
        .prepare_point_cloud(
            cloud
        )
    )

    mean_point_distance = (
        base_reconstructor
        .estimate_mean_point_distance(
            prepared_cloud
        )
    )

    print()
    print(
        f"Mittlerer Punktabstand: "
        f"{mean_point_distance:.6f}"
    )

    # =========================================================
    # Ergebnisse sammeln
    # =========================================================

    results = []

    # =========================================================
    # Alle Radius-Konfigurationen testen
    # =========================================================

    for (
        configuration_name,
        radius_factors,
    ) in RADIUS_CONFIGURATIONS.items():

        print()
        print()
        print("#" * 80)
        print(
            f"TEST: {configuration_name}"
        )
        print("#" * 80)

        print(
            f"Radius Factors: "
            f"{radius_factors}"
        )

        # -----------------------------------------------------
        # Eigene Config für diesen Lauf
        # -----------------------------------------------------

        config = (
            MeshConfig(
                normal_radius_factor=(
                    base_config
                    .normal_radius_factor
                ),

                normal_max_nn=(
                    base_config
                    .normal_max_nn
                ),

                poisson_depth=(
                    base_config
                    .poisson_depth
                ),

                poisson_density_quantile=(
                    base_config
                    .poisson_density_quantile
                ),

                ball_pivoting_radius_factors=(
                    radius_factors
                ),

                minimum_component_triangles=(
                    base_config
                    .minimum_component_triangles
                ),

                target_triangle_count=(
                    base_config
                    .target_triangle_count
                ),

                evaluation_sample_points=(
                    base_config
                    .evaluation_sample_points
                ),
            )
        )

        reconstructor = (
            MeshReconstructor(
                config
            )
        )

        cleaner = (
            MeshCleaner(
                config
            )
        )

        evaluator = (
            MeshEvaluator(
                config
            )
        )

        # -----------------------------------------------------
        # Tatsächliche Radien berechnen
        # -----------------------------------------------------

        actual_radii = tuple(
            mean_point_distance
            * factor

            for factor
            in radius_factors
        )

        print()
        print(
            "Tatsächliche Radien:"
        )

        for index, radius in enumerate(
            actual_radii,
            start=1,
        ):

            print(
                f"  Radius {index}: "
                f"{radius:.6f}"
            )

        # -----------------------------------------------------
        # Unterordner für diese Konfiguration
        # -----------------------------------------------------

        configuration_directory = (
            output_directory
            / configuration_name
        )

        configuration_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # =====================================================
        # Ball Pivoting
        # =====================================================

        raw_mesh = (
            reconstructor
            .reconstruct_ball_pivoting(
                prepared_cloud
            )
        )

        raw_mesh_path = (
            configuration_directory
            / "raw_mesh.ply"
        )

        success = (
            o3d.io.write_triangle_mesh(
                str(
                    raw_mesh_path
                ),
                raw_mesh,
            )
        )

        if not success:

            raise RuntimeError(
                "Raw-Mesh konnte für "
                f"{configuration_name} "
                "nicht gespeichert werden."
            )

        # =====================================================
        # Basis-Cleanup
        # =====================================================

        cleanup_result = (
            cleaner.cleanup_basic(
                raw_mesh
            )
        )

        cleaned_mesh = (
            cleanup_result.mesh
        )

        cleaned_mesh_path = (
            configuration_directory
            / "cleaned_mesh.ply"
        )

        success = (
            o3d.io.write_triangle_mesh(
                str(
                    cleaned_mesh_path
                ),
                cleaned_mesh,
            )
        )

        if not success:

            raise RuntimeError(
                "Cleaned-Mesh konnte für "
                f"{configuration_name} "
                "nicht gespeichert werden."
            )

        # =====================================================
        # Evaluation
        #
        # Gleicher Seed für die Mesh-Sampling-Evaluation,
        # damit der Vergleich möglichst reproduzierbar bleibt.
        # =====================================================

        o3d.utility.random.seed(
            42
        )

        evaluation = (
            evaluator.evaluate(
                mesh=(
                    cleaned_mesh
                ),
                reference_cloud=(
                    cloud
                ),
            )
        )

        evaluation_path = (
            configuration_directory
            / "mesh_evaluation.json"
        )

        evaluator.save_json(
            result=(
                evaluation
            ),
            path=(
                evaluation_path
            ),
        )

        # =====================================================
        # Ergebnisse für CSV sammeln
        # =====================================================

        results.append(
            {
                "Configuration": (
                    configuration_name
                ),

                "Radius Factor 1": (
                    radius_factors[0]
                ),

                "Radius Factor 2": (
                    radius_factors[1]
                ),

                "Radius Factor 3": (
                    radius_factors[2]
                ),

                "Radius 1": (
                    actual_radii[0]
                ),

                "Radius 2": (
                    actual_radii[1]
                ),

                "Radius 3": (
                    actual_radii[2]
                ),

                "Vertices": (
                    evaluation.vertices
                ),

                "Triangles": (
                    evaluation.triangles
                ),

                "Connected Components": (
                    evaluation
                    .connected_components
                ),

                "Largest Component Ratio": (
                    evaluation
                    .largest_component_ratio
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

                "Cloud to Mesh Mean": (
                    evaluation
                    .cloud_to_mesh_mean
                ),

                "Cloud to Mesh RMSE": (
                    evaluation
                    .cloud_to_mesh_rmse
                ),

                "Cloud to Mesh P95": (
                    evaluation
                    .cloud_to_mesh_p95
                ),

                "Mesh to Cloud Mean": (
                    evaluation
                    .mesh_to_cloud_mean
                ),

                "Mesh to Cloud RMSE": (
                    evaluation
                    .mesh_to_cloud_rmse
                ),

                "Mesh to Cloud P95": (
                    evaluation
                    .mesh_to_cloud_p95
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
        / "ball_pivoting_parameter_results.csv"
    )

    if not results:

        raise RuntimeError(
            "Keine Experimentergebnisse vorhanden."
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
    # Vergleichstabelle ausgeben
    # =========================================================

    print()
    print()
    print("=" * 110)
    print(
        "BALL PIVOTING PARAMETERVERGLEICH"
    )
    print("=" * 110)

    print()

    print(
        f"{'Config':<15}"
        f"{'Triangles':>12}"
        f"{'Components':>13}"
        f"{'Largest %':>12}"
        f"{'C->M RMSE':>12}"
        f"{'M->C RMSE':>12}"
        f"{'Sym RMSE':>12}"
        f"{'SelfInt':>10}"
    )

    print(
        "-" * 98
    )

    for result in results:

        print(
            f"{result['Configuration']:<15}"
            f"{result['Triangles']:>12,}"
            f"{result['Connected Components']:>13}"
            f"{result['Largest Component Ratio'] * 100:>11.2f}%"
            f"{result['Cloud to Mesh RMSE']:>12.6f}"
            f"{result['Mesh to Cloud RMSE']:>12.6f}"
            f"{result['Symmetric RMSE']:>12.6f}"
            f"{str(result['Self Intersecting']):>10}"
        )

    # =========================================================
    # Beste Konfiguration nach Symmetric RMSE
    #
    # Das ist zunächst nur eine numerische Rangfolge.
    # Die finale Auswahl erfolgt NICHT ausschließlich anhand
    # dieser einen Kennzahl, sondern zusammen mit Topologie
    # und visueller Kontrolle.
    # =========================================================

    best_result = min(
        results,
        key=lambda result: (
            result[
                "Symmetric RMSE"
            ]
        ),
    )

    print()
    print("=" * 110)
    print(
        "NUMERISCH BESTE KONFIGURATION"
    )
    print("=" * 110)

    print()
    print(
        f"Konfiguration: "
        f"{best_result['Configuration']}"
    )

    print(
        f"Symmetric RMSE: "
        f"{best_result['Symmetric RMSE']:.6f}"
    )

    print(
        f"Connected Components: "
        f"{best_result['Connected Components']}"
    )

    print(
        f"Largest Component: "
        f"{best_result['Largest Component Ratio'] * 100:.2f} %"
    )

    print(
        f"Self Intersecting: "
        f"{best_result['Self Intersecting']}"
    )

    print()
    print(
        "Wichtig: Die endgültige Auswahl erfolgt "
        "nicht nur anhand des RMSE."
    )

    print(
        "Topologie und visuelle Qualität müssen "
        "ebenfalls berücksichtigt werden."
    )

    print()
    print(
        "CSV gespeichert:"
    )

    print(
        f"  {csv_path}"
    )


if __name__ == "__main__":
    main()