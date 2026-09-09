from pathlib import Path

import numpy as np

from registration3d.algorithms.fgr_fpfh_icp import (
    FgrFPFHICP,
)
from registration3d.algorithms.generalized_icp import (
    GeneralizedICP,
)
from registration3d.algorithms.point_to_plane_icp import (
    PointToPlaneICP,
)
from registration3d.algorithms.point_to_point_icp import (
    PointToPointICP,
)
from registration3d.algorithms.ransac_fpfh_icp import (
    RansacFPFHICP,
)
from registration3d.config import RegistrationConfig
from registration3d.data_loader import PointCloudLoader
from registration3d.evaluator import RegistrationEvaluator
from registration3d.preprocessing import (
    PointCloudPreprocessor,
)
from registration3d.visualization import (
    RegistrationVisualizer,
)


def main() -> None:

    # ---------------------------------------------------------
    # Projektpfad
    # ---------------------------------------------------------

    project_root = Path(__file__).resolve().parents[2]

    source_path = (
        project_root
        / "data"
        / "cloud_01.ply"
    )

    target_path = (
        project_root
        / "data"
        / "cloud_02.ply"
    )

    ground_truth_path = (
        project_root
        / "data"
        / "ground_truth.npy"
    )

    # ---------------------------------------------------------
    # 1. Punktwolken laden
    # ---------------------------------------------------------

    source = PointCloudLoader.load(
        source_path
    )

    target = PointCloudLoader.load(
        target_path
    )

    print()
    print("Original:")

    print(
        f"Source: "
        f"{len(source.points):,} Punkte"
    )

    print(
        f"Target: "
        f"{len(target.points):,} Punkte"
    )

    # ---------------------------------------------------------
    # 2. Ground Truth laden
    # ---------------------------------------------------------

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground-Truth-Datei nicht gefunden: "
            f"{ground_truth_path}"
        )

    ground_truth = np.load(
        ground_truth_path
    )

    if ground_truth.shape != (4, 4):
        raise ValueError(
            "Ground Truth muss eine 4x4-Transformationsmatrix sein."
        )

    print()
    print("Ground-Truth-Transformation:")
    print(ground_truth)

    # ---------------------------------------------------------
    # 3. Konfiguration
    # ---------------------------------------------------------

    config = RegistrationConfig(
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

    # ---------------------------------------------------------
    # 4. Preprocessing
    # ---------------------------------------------------------

    preprocessor = PointCloudPreprocessor(
        config
    )

    source_processed = (
        preprocessor.preprocess(source)
    )

    target_processed = (
        preprocessor.preprocess(target)
    )

    print()
    print("Nach Preprocessing:")

    print(
        f"Source: "
        f"{len(source_processed.points):,} Punkte"
    )

    print(
        f"Target: "
        f"{len(target_processed.points):,} Punkte"
    )

    print(
        f"Source hat Normalen: "
        f"{source_processed.has_normals()}"
    )

    print(
        f"Target hat Normalen: "
        f"{target_processed.has_normals()}"
    )

    # ---------------------------------------------------------
    # 5. Ausgangslage visualisieren
    # ---------------------------------------------------------

    RegistrationVisualizer.show_pair(
        source_processed,
        target_processed,
        window_name="Before Registration",
    )

    # ---------------------------------------------------------
    # 6. Registrierungsalgorithmen
    # ---------------------------------------------------------

    algorithms = [
        PointToPointICP(config),
        PointToPlaneICP(config),
        GeneralizedICP(config),
        RansacFPFHICP(config),
        FgrFPFHICP(config),
    ]

    # ---------------------------------------------------------
    # 7. Algorithmen ausführen
    # ---------------------------------------------------------

    results = []

    for algorithm in algorithms:

        print()
        print(
            f"Starte {algorithm.name} ..."
        )

        # ---------------------------------------------
        # Registrierung
        # ---------------------------------------------

        result = algorithm.register(
            source_processed,
            target_processed,
        )

        # ---------------------------------------------
        # Einheitliche Evaluation + Ground Truth
        # ---------------------------------------------

        result = (
            RegistrationEvaluator
            .evaluate_result(
                result,
                source_processed,
                target_processed,
                config,
                ground_truth,
            )
        )

        results.append(
            result
        )

        RegistrationEvaluator.print_result(
            result
        )

    # ---------------------------------------------------------
    # 8. Registrierte Ergebnisse visualisieren
    # ---------------------------------------------------------

    for result in results:

        RegistrationVisualizer.show_result(
            source_processed,
            target_processed,
            result,
        )

    # ---------------------------------------------------------
    # 9. Vergleichstabelle
    # ---------------------------------------------------------

    RegistrationEvaluator.print_comparison_table(
        results
    )

    # ---------------------------------------------------------
    # 10. Ergebnisse als CSV speichern
    # ---------------------------------------------------------

    RegistrationEvaluator.save_results_csv(
        results,
        project_root
        / "output"
        / "registration_results.csv",
    )


if __name__ == "__main__":
    main()