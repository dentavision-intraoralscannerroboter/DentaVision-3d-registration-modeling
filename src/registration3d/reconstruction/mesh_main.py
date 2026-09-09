from pathlib import Path

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

    poisson_output_directory = (
        project_root
        / "output"
        / "mesh"
        / "poisson"
    )

    ball_pivoting_output_directory = (
        project_root
        / "output"
        / "mesh"
        / "ball_pivoting"
    )

    poisson_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    ball_pivoting_output_directory.mkdir(
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
            f"{input_path}\n\n"
            "Bitte zuerst die Multiway-Registrierung "
            "ausführen."
        )

    # =========================================================
    # Punktwolke laden
    # =========================================================

    print()
    print("=" * 80)
    print("3-D MESH RECONSTRUCTION")
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
    # Konfiguration
    # =========================================================

    config = (
        MeshConfig(
            normal_radius_factor=3.0,
            normal_max_nn=50,

            poisson_depth=9,
            poisson_density_quantile=0.02,

            # Ergebnis BPA-Parameterstudie
            ball_pivoting_radius_factors=(
                1.0,
                1.5,
                2.0,
            ),

            # Ergebnis Komponentenanalyse
            minimum_component_triangles=250,

            target_triangle_count=100_000,

            evaluation_sample_points=100_000,
        )
    )

    print()
    print("=" * 80)
    print("MESH-KONFIGURATION")
    print("=" * 80)

    print(
        f"Poisson Depth: "
        f"{config.poisson_depth}"
    )

    print(
        f"Poisson Density Quantile: "
        f"{config.poisson_density_quantile}"
    )

    print(
        f"Ball Pivoting Radius Factors: "
        f"{config.ball_pivoting_radius_factors}"
    )

    print(
        f"Minimum Component Triangles: "
        f"{config.minimum_component_triangles}"
    )

    print(
        f"Evaluation Sample Points: "
        f"{config.evaluation_sample_points:,}"
    )

    # =========================================================
    # Objekte erzeugen
    # =========================================================

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

    # =========================================================
    # Punktwolke vorbereiten
    # =========================================================

    prepared_cloud = (
        reconstructor
        .prepare_point_cloud(
            cloud
        )
    )

    # =========================================================
    # Vorbereitete Punktwolke speichern
    # =========================================================

    prepared_cloud_path = (
        poisson_output_directory
        / "prepared_cloud.ply"
    )

    success = (
        o3d.io.write_point_cloud(
            str(
                prepared_cloud_path
            ),
            prepared_cloud,
        )
    )

    if not success:

        raise RuntimeError(
            "Vorbereitete Punktwolke konnte "
            "nicht gespeichert werden."
        )

    # =========================================================
    # POISSON
    # =========================================================

    (
        poisson_raw_mesh,
        poisson_densities,
    ) = (
        reconstructor
        .reconstruct_poisson(
            prepared_cloud
        )
    )

    poisson_raw_path = (
        poisson_output_directory
        / "raw_mesh.ply"
    )

    o3d.io.write_triangle_mesh(
        str(
            poisson_raw_path
        ),
        poisson_raw_mesh,
    )

    # =========================================================
    # Poisson Density Cleanup
    # =========================================================

    poisson_filtered_mesh = (
        reconstructor
        .remove_low_density_vertices(
            mesh=(
                poisson_raw_mesh
            ),
            densities=(
                poisson_densities
            ),
        )
    )

    poisson_filtered_path = (
        poisson_output_directory
        / "filtered_mesh.ply"
    )

    o3d.io.write_triangle_mesh(
        str(
            poisson_filtered_path
        ),
        poisson_filtered_mesh,
    )

    # =========================================================
    # Poisson Evaluation
    # =========================================================

    poisson_evaluation = (
        evaluator.evaluate(
            mesh=(
                poisson_filtered_mesh
            ),
            reference_cloud=(
                cloud
            ),
        )
    )

    poisson_evaluation_path = (
        poisson_output_directory
        / "mesh_evaluation.json"
    )

    evaluator.save_json(
        result=(
            poisson_evaluation
        ),
        path=(
            poisson_evaluation_path
        ),
    )

    # =========================================================
    # BALL PIVOTING
    # =========================================================

    ball_pivoting_raw_mesh = (
        reconstructor
        .reconstruct_ball_pivoting(
            prepared_cloud
        )
    )

    ball_pivoting_raw_path = (
        ball_pivoting_output_directory
        / "raw_mesh.ply"
    )

    success = (
        o3d.io.write_triangle_mesh(
            str(
                ball_pivoting_raw_path
            ),
            ball_pivoting_raw_mesh,
        )
    )

    if not success:

        raise RuntimeError(
            "Ball-Pivoting-Rohmesh konnte "
            "nicht gespeichert werden."
        )

    print()
    print(
        "Ball-Pivoting-Rohmesh gespeichert:"
    )

    print(
        f"  {ball_pivoting_raw_path}"
    )

    # =========================================================
    # BPA Basic Cleanup
    # =========================================================

    cleanup_result = (
        cleaner.cleanup_basic(
            ball_pivoting_raw_mesh
        )
    )

    ball_pivoting_basic_cleaned_mesh = (
        cleanup_result.mesh
    )

    basic_cleaned_path = (
        ball_pivoting_output_directory
        / "basic_cleaned_mesh.ply"
    )

    o3d.io.write_triangle_mesh(
        str(
            basic_cleaned_path
        ),
        ball_pivoting_basic_cleaned_mesh,
    )

    # =========================================================
    # BPA Component Cleanup
    #
    # Entfernt auf diesem synthetischen Datensatz:
    #
    # 81 kleine Komponenten
    # 389 Dreiecke
    #
    # Hauptkomponente:
    # ca. 40.020 Dreiecke
    # =========================================================

    ball_pivoting_final_mesh = (
        cleaner.remove_small_components(
            mesh=(
                ball_pivoting_basic_cleaned_mesh
            ),
            minimum_triangles=(
                config
                .minimum_component_triangles
            ),
        )
    )

    # =========================================================
    # Finalen BPA-Kandidaten speichern
    # =========================================================

    final_candidate_path = (
        ball_pivoting_output_directory
        / "final_candidate_mesh.ply"
    )

    success = (
        o3d.io.write_triangle_mesh(
            str(
                final_candidate_path
            ),
            ball_pivoting_final_mesh,
        )
    )

    if not success:

        raise RuntimeError(
            "Finaler Ball-Pivoting-Kandidat "
            "konnte nicht gespeichert werden."
        )

    print()
    print(
        "Finaler Ball-Pivoting-Kandidat gespeichert:"
    )

    print(
        f"  {final_candidate_path}"
    )

    print(
        f"  Vertices: "
        f"{len(ball_pivoting_final_mesh.vertices):,}"
    )

    print(
        f"  Dreiecke: "
        f"{len(ball_pivoting_final_mesh.triangles):,}"
    )

    # =========================================================
    # Finale BPA Evaluation
    # =========================================================

    ball_pivoting_evaluation = (
        evaluator.evaluate(
            mesh=(
                ball_pivoting_final_mesh
            ),
            reference_cloud=(
                cloud
            ),
        )
    )

    ball_pivoting_evaluation_path = (
        ball_pivoting_output_directory
        / "final_candidate_evaluation.json"
    )

    evaluator.save_json(
        result=(
            ball_pivoting_evaluation
        ),
        path=(
            ball_pivoting_evaluation_path
        ),
    )

    # =========================================================
    # Vergleich
    # =========================================================

    print()
    print("=" * 90)
    print(
        "FINALER VERGLEICH: "
        "POISSON VS. BALL PIVOTING"
    )
    print("=" * 90)

    print()

    print(
        f"{'Kennzahl':<35}"
        f"{'Poisson':>18}"
        f"{'Ball Pivoting':>20}"
    )

    print(
        "-" * 73
    )

    print(
        f"{'Vertices':<35}"
        f"{poisson_evaluation.vertices:>18,}"
        f"{ball_pivoting_evaluation.vertices:>20,}"
    )

    print(
        f"{'Dreiecke':<35}"
        f"{poisson_evaluation.triangles:>18,}"
        f"{ball_pivoting_evaluation.triangles:>20,}"
    )

    print(
        f"{'Connected Components':<35}"
        f"{poisson_evaluation.connected_components:>18}"
        f"{ball_pivoting_evaluation.connected_components:>20}"
    )

    print(
        f"{'Edge Manifold':<35}"
        f"{str(poisson_evaluation.edge_manifold_with_boundary):>18}"
        f"{str(ball_pivoting_evaluation.edge_manifold_with_boundary):>20}"
    )

    print(
        f"{'Vertex Manifold':<35}"
        f"{str(poisson_evaluation.vertex_manifold):>18}"
        f"{str(ball_pivoting_evaluation.vertex_manifold):>20}"
    )

    print(
        f"{'Self Intersecting':<35}"
        f"{str(poisson_evaluation.self_intersecting):>18}"
        f"{str(ball_pivoting_evaluation.self_intersecting):>20}"
    )

    print(
        f"{'Watertight':<35}"
        f"{str(poisson_evaluation.watertight):>18}"
        f"{str(ball_pivoting_evaluation.watertight):>20}"
    )

    print(
        f"{'Orientable':<35}"
        f"{str(poisson_evaluation.orientable):>18}"
        f"{str(ball_pivoting_evaluation.orientable):>20}"
    )

    print(
        f"{'Cloud -> Mesh RMSE':<35}"
        f"{poisson_evaluation.cloud_to_mesh_rmse:>18.6f}"
        f"{ball_pivoting_evaluation.cloud_to_mesh_rmse:>20.6f}"
    )

    print(
        f"{'Mesh -> Cloud RMSE':<35}"
        f"{poisson_evaluation.mesh_to_cloud_rmse:>18.6f}"
        f"{ball_pivoting_evaluation.mesh_to_cloud_rmse:>20.6f}"
    )

    print(
        f"{'Symmetric Mean':<35}"
        f"{poisson_evaluation.symmetric_mean_distance:>18.6f}"
        f"{ball_pivoting_evaluation.symmetric_mean_distance:>20.6f}"
    )

    print(
        f"{'Symmetric RMSE':<35}"
        f"{poisson_evaluation.symmetric_rmse:>18.6f}"
        f"{ball_pivoting_evaluation.symmetric_rmse:>20.6f}"
    )

    # =========================================================
    # Abschluss
    # =========================================================

    print()
    print("=" * 80)
    print(
        "MESH-REKONSTRUKTION ABGESCHLOSSEN"
    )
    print("=" * 80)

    print()
    print(
        "Finaler BPA-Kandidat:"
    )

    print(
        f"  {final_candidate_path}"
    )

    print(
        f"  Vertices: "
        f"{len(ball_pivoting_final_mesh.vertices):,}"
    )

    print(
        f"  Dreiecke: "
        f"{len(ball_pivoting_final_mesh.triangles):,}"
    )

    print(
        f"  Connected Components: "
        f"{ball_pivoting_evaluation.connected_components}"
    )

    print(
        f"  Symmetric RMSE: "
        f"{ball_pivoting_evaluation.symmetric_rmse:.6f}"
    )

    # =========================================================
    # Visualisierung
    # =========================================================

    print()
    print(
        "Öffne finalen "
        "Ball-Pivoting-Kandidaten ..."
    )

    ball_pivoting_final_mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries(
        [
            ball_pivoting_final_mesh
        ],

        window_name=(
            "Final Ball Pivoting Dental Mesh"
        ),

        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()