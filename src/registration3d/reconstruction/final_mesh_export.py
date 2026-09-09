from pathlib import Path
import csv
import shutil

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
        / "simplification_experiments"
        / "target_30000"
        / "simplified_mesh.ply"
    )

    original_mesh_path = (
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
        / "final"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # Dateien prüfen
    # =========================================================

    if not input_mesh_path.exists():

        raise FileNotFoundError(
            "Das ausgewählte 30k-Mesh wurde "
            "nicht gefunden:\n"
            f"{input_mesh_path}"
        )

    if not original_mesh_path.exists():

        raise FileNotFoundError(
            "Originales BPA-Mesh wurde "
            "nicht gefunden:\n"
            f"{original_mesh_path}"
        )

    if not reference_cloud_path.exists():

        raise FileNotFoundError(
            "Referenzpunktwolke wurde "
            "nicht gefunden:\n"
            f"{reference_cloud_path}"
        )

    # =========================================================
    # Start
    # =========================================================

    print()
    print("=" * 80)
    print("FINAL 3-D MODEL EXPORT")
    print("=" * 80)

    print()
    print(
        "Ausgewählte Vereinfachung:"
    )

    print(
        "  Target = 30.000 Dreiecke"
    )

    print()
    print(
        f"Eingabe:\n"
        f"  {input_mesh_path}"
    )

    # =========================================================
    # Mesh laden
    # =========================================================

    mesh = (
        o3d.io.read_triangle_mesh(
            str(input_mesh_path)
        )
    )

    if len(mesh.vertices) == 0:

        raise ValueError(
            "Das Mesh enthält keine Vertices."
        )

    if len(mesh.triangles) == 0:

        raise ValueError(
            "Das Mesh enthält keine Dreiecke."
        )

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    print()
    print(
        f"Geladen:"
    )

    print(
        f"  Vertices: "
        f"{len(mesh.vertices):,}"
    )

    print(
        f"  Dreiecke: "
        f"{len(mesh.triangles):,}"
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

            ball_pivoting_radius_factors=(
                1.0,
                1.5,
                2.0,
            ),

            minimum_component_triangles=250,

            # Ergebnis der Simplification-Studie
            target_triangle_count=30_000,

            evaluation_sample_points=100_000,
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
    # Basic Cleanup
    # =========================================================

    cleanup_result = (
        cleaner.cleanup_basic(
            mesh
        )
    )

    cleaned_mesh = (
        cleanup_result.mesh
    )

    # =========================================================
    # Winzige Komponenten entfernen
    #
    # Im 30k-Mesh:
    #
    # Hauptkomponente = 29.998 Dreiecke
    # zweite Komponente = 1 Dreieck
    #
    # Wir verwenden deshalb bewusst nur 10 Dreiecke als
    # Schwellenwert und NICHT die 250 aus der BPA-Rekonstruktion.
    # =========================================================

    final_mesh = (
        cleaner.remove_small_components(
            mesh=cleaned_mesh,
            minimum_triangles=10,
        )
    )

    final_mesh.compute_triangle_normals()
    final_mesh.compute_vertex_normals()

    print()
    print("=" * 80)
    print("FINALER MESH-KANDIDAT")
    print("=" * 80)

    print()
    print(
        f"Vertices: "
        f"{len(final_mesh.vertices):,}"
    )

    print(
        f"Dreiecke: "
        f"{len(final_mesh.triangles):,}"
    )

    # =========================================================
    # Referenzpunktwolke laden
    # =========================================================

    reference_cloud = (
        PointCloudLoader.load(
            reference_cloud_path
        )
    )

    # =========================================================
    # Finale Evaluation
    # =========================================================

    print()
    print("=" * 80)
    print("FINALE QUALITÄTSPRÜFUNG")
    print("=" * 80)

    o3d.utility.random.seed(
        42
    )

    final_evaluation = (
        evaluator.evaluate(
            mesh=final_mesh,
            reference_cloud=reference_cloud,
        )
    )

    evaluation_path = (
        output_directory
        / "final_model_evaluation.json"
    )

    evaluator.save_json(
        result=final_evaluation,
        path=evaluation_path,
    )

    # =========================================================
    # PLY Export
    # =========================================================

    ply_path = (
        output_directory
        / "dental_model_final.ply"
    )

    success = (
        o3d.io.write_triangle_mesh(
            str(ply_path),
            final_mesh,
            write_ascii=False,
        )
    )

    if not success:

        raise RuntimeError(
            "PLY-Datei konnte nicht "
            "gespeichert werden."
        )

    print()
    print(
        "PLY exportiert:"
    )

    print(
        f"  {ply_path}"
    )

    # =========================================================
    # STL Export
    # =========================================================

    stl_path = (
        output_directory
        / "dental_model_final.stl"
    )

    success = (
        o3d.io.write_triangle_mesh(
            str(stl_path),
            final_mesh,
            write_ascii=False,
        )
    )

    if not success:

        raise RuntimeError(
            "STL-Datei konnte nicht "
            "gespeichert werden."
        )

    print()
    print(
        "STL exportiert:"
    )

    print(
        f"  {stl_path}"
    )

    # =========================================================
    # Originales Full-Resolution-Mesh ebenfalls archivieren
    #
    # Dieses bleibt als nicht vereinfachte Referenz erhalten.
    # =========================================================

    full_resolution_path = (
        output_directory
        / "dental_model_full_resolution.ply"
    )

    shutil.copy2(
        original_mesh_path,
        full_resolution_path,
    )

    print()
    print(
        "Full-Resolution-Referenz gespeichert:"
    )

    print(
        f"  {full_resolution_path}"
    )

    # =========================================================
    # Exportdateien erneut einlesen
    #
    # Damit prüfen wir, ob die Dateien tatsächlich als
    # TriangleMesh gelesen werden können.
    # =========================================================

    print()
    print("=" * 80)
    print("EXPORT-INTEGRITÄT")
    print("=" * 80)

    exported_ply = (
        o3d.io.read_triangle_mesh(
            str(ply_path)
        )
    )

    exported_stl = (
        o3d.io.read_triangle_mesh(
            str(stl_path)
        )
    )

    ply_valid = (
        len(exported_ply.vertices) > 0
        and
        len(exported_ply.triangles) > 0
    )

    stl_valid = (
        len(exported_stl.vertices) > 0
        and
        len(exported_stl.triangles) > 0
    )

    print()
    print(
        "PLY wieder eingelesen:"
    )

    print(
        f"  gültig: {ply_valid}"
    )

    print(
        f"  Vertices: "
        f"{len(exported_ply.vertices):,}"
    )

    print(
        f"  Dreiecke: "
        f"{len(exported_ply.triangles):,}"
    )

    print()
    print(
        "STL wieder eingelesen:"
    )

    print(
        f"  gültig: {stl_valid}"
    )

    print(
        f"  Vertices: "
        f"{len(exported_stl.vertices):,}"
    )

    print(
        f"  Dreiecke: "
        f"{len(exported_stl.triangles):,}"
    )

    if not ply_valid:

        raise RuntimeError(
            "Der PLY-Export konnte nicht "
            "korrekt wieder eingelesen werden."
        )

    if not stl_valid:

        raise RuntimeError(
            "Der STL-Export konnte nicht "
            "korrekt wieder eingelesen werden."
        )

    # =========================================================
    # Vergleich Original vs. finales Modell
    # =========================================================

    original_mesh = (
        o3d.io.read_triangle_mesh(
            str(original_mesh_path)
        )
    )

    o3d.utility.random.seed(
        42
    )

    original_evaluation = (
        evaluator.evaluate(
            mesh=original_mesh,
            reference_cloud=reference_cloud,
        )
    )

    original_triangles = (
        len(original_mesh.triangles)
    )

    final_triangles = (
        len(final_mesh.triangles)
    )

    reduction_percent = (
        (
            1.0
            - (
                final_triangles
                / original_triangles
            )
        )
        * 100.0
    )

    symmetric_rmse_change = (
        final_evaluation.symmetric_rmse
        - original_evaluation.symmetric_rmse
    )

    symmetric_rmse_change_percent = (
        (
            symmetric_rmse_change
            / original_evaluation.symmetric_rmse
        )
        * 100.0
    )

    # =========================================================
    # Zusammenfassung als CSV
    # =========================================================

    summary_path = (
        output_directory
        / "final_model_summary.csv"
    )

    rows = [
        {
            "Variant": "Original BPA",
            "Vertices": (
                original_evaluation.vertices
            ),
            "Triangles": (
                original_evaluation.triangles
            ),
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
            "Symmetric RMSE": (
                original_evaluation
                .symmetric_rmse
            ),
        },
        {
            "Variant": "Final Simplified",
            "Vertices": (
                final_evaluation.vertices
            ),
            "Triangles": (
                final_evaluation.triangles
            ),
            "Connected Components": (
                final_evaluation
                .connected_components
            ),
            "Edge Manifold": (
                final_evaluation
                .edge_manifold_with_boundary
            ),
            "Vertex Manifold": (
                final_evaluation
                .vertex_manifold
            ),
            "Self Intersecting": (
                final_evaluation
                .self_intersecting
            ),
            "Watertight": (
                final_evaluation
                .watertight
            ),
            "Orientable": (
                final_evaluation
                .orientable
            ),
            "Cloud to Mesh RMSE": (
                final_evaluation
                .cloud_to_mesh_rmse
            ),
            "Mesh to Cloud RMSE": (
                final_evaluation
                .mesh_to_cloud_rmse
            ),
            "Symmetric RMSE": (
                final_evaluation
                .symmetric_rmse
            ),
        },
    ]

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = (
            csv.DictWriter(
                file,
                fieldnames=list(
                    rows[0].keys()
                ),
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # =========================================================
    # Finale Zusammenfassung
    # =========================================================

    print()
    print("=" * 90)
    print("FINALES 3-D-MODELL")
    print("=" * 90)

    print()
    print(
        f"{'Kennzahl':<35}"
        f"{'Original BPA':>18}"
        f"{'Final':>18}"
    )

    print(
        "-" * 71
    )

    print(
        f"{'Vertices':<35}"
        f"{original_evaluation.vertices:>18,}"
        f"{final_evaluation.vertices:>18,}"
    )

    print(
        f"{'Dreiecke':<35}"
        f"{original_evaluation.triangles:>18,}"
        f"{final_evaluation.triangles:>18,}"
    )

    print(
        f"{'Connected Components':<35}"
        f"{original_evaluation.connected_components:>18}"
        f"{final_evaluation.connected_components:>18}"
    )

    print(
        f"{'Edge Manifold':<35}"
        f"{str(original_evaluation.edge_manifold_with_boundary):>18}"
        f"{str(final_evaluation.edge_manifold_with_boundary):>18}"
    )

    print(
        f"{'Vertex Manifold':<35}"
        f"{str(original_evaluation.vertex_manifold):>18}"
        f"{str(final_evaluation.vertex_manifold):>18}"
    )

    print(
        f"{'Self Intersecting':<35}"
        f"{str(original_evaluation.self_intersecting):>18}"
        f"{str(final_evaluation.self_intersecting):>18}"
    )

    print(
        f"{'Watertight':<35}"
        f"{str(original_evaluation.watertight):>18}"
        f"{str(final_evaluation.watertight):>18}"
    )

    print(
        f"{'Orientable':<35}"
        f"{str(original_evaluation.orientable):>18}"
        f"{str(final_evaluation.orientable):>18}"
    )

    print(
        f"{'Cloud -> Mesh RMSE':<35}"
        f"{original_evaluation.cloud_to_mesh_rmse:>18.6f}"
        f"{final_evaluation.cloud_to_mesh_rmse:>18.6f}"
    )

    print(
        f"{'Mesh -> Cloud RMSE':<35}"
        f"{original_evaluation.mesh_to_cloud_rmse:>18.6f}"
        f"{final_evaluation.mesh_to_cloud_rmse:>18.6f}"
    )

    print(
        f"{'Symmetric RMSE':<35}"
        f"{original_evaluation.symmetric_rmse:>18.6f}"
        f"{final_evaluation.symmetric_rmse:>18.6f}"
    )

    print()
    print(
        f"Dreiecksreduktion: "
        f"{reduction_percent:.2f} %"
    )

    print(
        f"Änderung Symmetric RMSE: "
        f"{symmetric_rmse_change:+.6f} "
        f"({symmetric_rmse_change_percent:+.3f} %)"
    )

    print()
    print(
        "Exportdateien:"
    )

    print(
        f"  PLY: {ply_path}"
    )

    print(
        f"  STL: {stl_path}"
    )

    print(
        f"  Full Resolution: "
        f"{full_resolution_path}"
    )

    print(
        f"  Evaluation: "
        f"{evaluation_path}"
    )

    print(
        f"  Zusammenfassung: "
        f"{summary_path}"
    )

    print()
    print("=" * 90)
    print("3-D-MODELL-PIPELINE ABGESCHLOSSEN")
    print("=" * 90)

    # =========================================================
    # Finale Visualisierung
    # =========================================================

    print()
    print(
        "Öffne finales vereinfachtes Modell ..."
    )

    final_mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries(
        [
            final_mesh
        ],
        window_name=(
            "Final Simplified Dental Model"
        ),
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()