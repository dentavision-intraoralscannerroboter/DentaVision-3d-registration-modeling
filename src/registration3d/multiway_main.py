from pathlib import Path

import numpy as np
import open3d as o3d

from registration3d.algorithms.fgr_fpfh_icp import (
    FgrFPFHICP,
)
from registration3d.config import (
    RegistrationConfig,
)
from registration3d.data_loader import (
    PointCloudLoader,
)
from registration3d.multiway_config import (
    MultiwayConfig,
)
from registration3d.multiway_evaluator import (
    MultiwayEvaluator,
)
from registration3d.multiway_registration import (
    MultiwayRegistration,
)


def main() -> None:

    # =========================================================
    # Projektpfade
    # =========================================================

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    data_directory = (
        project_root
        / "data"
        / "multiway_test"
    )

    output_directory = (
        project_root
        / "output"
        / "multiway"
        / "fgr_baseline"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # Fragmente finden
    # =========================================================

    fragment_paths = sorted(
        data_directory.glob(
            "fragment_*.ply"
        )
    )

    if len(fragment_paths) < 2:
        raise RuntimeError(
            "Keine Multiway-Testfragmente gefunden."
        )

    print()
    print(
        f"{len(fragment_paths)} Fragmente gefunden."
    )

    # =========================================================
    # Fragmente laden
    # =========================================================

    clouds = [
        PointCloudLoader.load(path)
        for path in fragment_paths
    ]

    # =========================================================
    # Ground Truth
    #
    # Nur für synthetische Evaluation.
    # Wird NICHT von der Registrierung verwendet.
    # =========================================================

    ground_truth_path = (
        data_directory
        / "ground_truth_poses.npy"
    )

    ground_truth_poses = None

    if ground_truth_path.exists():

        ground_truth_poses = np.load(
            ground_truth_path
        )

        if (
            ground_truth_poses.ndim != 3
            or ground_truth_poses.shape[1:] != (4, 4)
        ):
            raise ValueError(
                "ground_truth_poses.npy muss "
                "die Form (N, 4, 4) haben."
            )

        if (
            len(ground_truth_poses)
            != len(clouds)
        ):
            raise ValueError(
                "Anzahl Ground-Truth-Posen "
                "stimmt nicht mit der Anzahl "
                "der Fragmente überein."
            )

    # =========================================================
    # Registration Config
    # =========================================================

    registration_config = (
        RegistrationConfig(
            voxel_size=0.5,

            normal_radius_factor=2.0,
            max_nn=30,

            correspondence_distance_factor=2.0,
            max_iterations=100,

            fpfh_radius_factor=5.0,
            fpfh_max_nn=100,

            ransac_distance_factor=1.5,
            ransac_n=3,
            ransac_max_iterations=100_000,
            ransac_confidence=0.999,

            fgr_distance_factor=0.5,

            refinement_distance_factor=0.4,

            evaluation_distance_factor=2.0,
        )
    )

    # =========================================================
    # Multiway Config
    # =========================================================
    #
    # Für die wissenschaftliche Baseline bleiben Loop Closures
    # zunächst deaktiviert.
    #
    # Damit bewerten wir ausschließlich:
    #
    # FPFH
    #   ↓
    # Fast Global Registration
    #   ↓
    # Point-to-Plane ICP Refinement
    #   ↓
    # Odometry / Pose Graph
    #
    # Die später untersuchte Cycle-Consistency ist eine
    # zusätzliche Robustheitserweiterung und gehört nicht
    # zur Baseline.
    # =========================================================

    multiway_config = (
        MultiwayConfig(
            odometry_min_fitness=0.15,
            odometry_max_rmse=0.80,

            use_loop_closures=False,

            loop_closure_max_gap=2,

            loop_closure_min_fitness=0.30,
            loop_closure_max_rmse=0.60,

            loop_closure_cycle_translation_threshold=1.0,
            loop_closure_cycle_rotation_threshold_degrees=5.0,

            edge_prune_threshold=0.25,

            preference_loop_closure=0.10,

            reference_node=0,

            merge_voxel_size=0.25,
        )
    )

    # =========================================================
    # Ausgewählter Hauptalgorithmus
    # =========================================================

    pairwise_algorithm = (
        FgrFPFHICP(
            registration_config
        )
    )

    # Für reproduzierbaren synthetischen Test
    o3d.utility.random.seed(
        42
    )

    # =========================================================
    # Konfiguration
    # =========================================================

    print()
    print("=" * 80)
    print("FINAL FGR MULTIWAY BASELINE")
    print("=" * 80)

    print(
        f"Pairwise Algorithmus: "
        f"{pairwise_algorithm.name}"
    )

    print(
        f"Voxel Size: "
        f"{registration_config.voxel_size}"
    )

    print(
        f"Loop Closures: "
        f"{multiway_config.use_loop_closures}"
    )

    # =========================================================
    # Multiway Registration
    # =========================================================

    multiway = (
        MultiwayRegistration(
            registration_config=(
                registration_config
            ),

            multiway_config=(
                multiway_config
            ),

            pairwise_algorithm=(
                pairwise_algorithm
            ),
        )
    )

    result = (
        multiway.register(
            clouds
        )
    )

    # =========================================================
    # Gesamtwolke speichern
    # =========================================================

    combined_path = (
        output_directory
        / "combined_cloud.ply"
    )

    success = (
        o3d.io.write_point_cloud(
            str(combined_path),
            result.combined_cloud,
        )
    )

    if not success:
        raise RuntimeError(
            "Gesamtpunktwolke konnte "
            "nicht gespeichert werden."
        )

    print()
    print(
        f"Gesamtwolke gespeichert: "
        f"{combined_path}"
    )

    # =========================================================
    # Einzelne registrierte Fragmente speichern
    # =========================================================

    for index, cloud in enumerate(
        result.transformed_clouds
    ):

        fragment_output_path = (
            output_directory
            / (
                f"registered_fragment_"
                f"{index:02d}.ply"
            )
        )

        o3d.io.write_point_cloud(
            str(fragment_output_path),
            cloud,
        )

    # =========================================================
    # Ground-Truth-Auswertung
    #
    # Nur verfügbar, wenn synthetische Ground Truth existiert.
    # Bei echten Dentalfragmenten wird dieser Teil automatisch
    # übersprungen.
    # =========================================================

    if ground_truth_poses is not None:

        estimated_poses = [
            np.asarray(
                node.pose
            ).copy()

            for node
            in result.pose_graph.nodes
        ]

        evaluation_table = (
            MultiwayEvaluator.evaluate_poses(
                estimated_poses=(
                    estimated_poses
                ),

                ground_truth_poses=(
                    ground_truth_poses
                ),
            )
        )

        print()
        print("=" * 80)
        print("POSE-GRAPH-AUSWERTUNG")
        print("=" * 80)

        print(
            evaluation_table.to_string(
                index=False,

                float_format=lambda value: (
                    f"{value:.6f}"
                ),
            )
        )

        evaluation_path = (
            output_directory
            / "pose_errors.csv"
        )

        MultiwayEvaluator.save_csv(
            evaluation_table,
            evaluation_path,
        )

        print()
        print(
            f"Pose-Fehler gespeichert: "
            f"{evaluation_path}"
        )

    else:

        print()
        print(
            "Keine Ground Truth vorhanden."
        )

        print(
            "Pose-Fehler-Evaluation wird "
            "übersprungen."
        )

    # =========================================================
    # Visualisierung
    # =========================================================

    print()
    print(
        "Öffne registrierte Gesamtwolke ..."
    )

    o3d.visualization.draw_geometries(
        [
            result.combined_cloud
        ],

        window_name=(
            "FPFH + FGR + ICP Multiway Registration"
        ),
    )


if __name__ == "__main__":
    main()