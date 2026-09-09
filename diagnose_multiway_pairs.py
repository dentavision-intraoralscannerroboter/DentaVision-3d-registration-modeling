from pathlib import Path

import numpy as np
import open3d as o3d

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
from registration3d.config import (
    RegistrationConfig,
)
from registration3d.data_loader import (
    PointCloudLoader,
)
from registration3d.evaluator import (
    RegistrationEvaluator,
)
from registration3d.preprocessing import (
    PointCloudPreprocessor,
)


def calculate_transform_error(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[float, float]:

    error_transform = (
        np.linalg.inv(ground_truth)
        @ estimated
    )

    translation_error = float(
        np.linalg.norm(
            error_transform[:3, 3]
        )
    )

    rotation_error = (
        error_transform[:3, :3]
    )

    cos_angle = (
        np.trace(rotation_error) - 1.0
    ) / 2.0

    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0,
    )

    rotation_error_degrees = float(
        np.degrees(
            np.arccos(cos_angle)
        )
    )

    return (
        translation_error,
        rotation_error_degrees,
    )


def main() -> None:

    project_root = (
        Path(__file__)
        .resolve()
        .parent
    )

    data_directory = (
        project_root
        / "data"
        / "multiway_test"
    )

    # ---------------------------------------------------------
    # Fragmente laden
    # ---------------------------------------------------------

    fragment_paths = sorted(
        data_directory.glob(
            "fragment_*.ply"
        )
    )

    if len(fragment_paths) < 2:
        raise RuntimeError(
            "Keine Multiway-Testfragmente gefunden."
        )

    clouds = [
        PointCloudLoader.load(path)
        for path in fragment_paths
    ]

    # ---------------------------------------------------------
    # Ground Truth laden
    # ---------------------------------------------------------

    acquisition_transforms_path = (
        data_directory
        / "acquisition_transforms.npy"
    )

    if not acquisition_transforms_path.exists():
        raise FileNotFoundError(
            "acquisition_transforms.npy "
            "wurde nicht gefunden."
        )

    acquisition_transforms = np.load(
        acquisition_transforms_path
    )

    # ---------------------------------------------------------
    # Konfiguration
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
    # Preprocessing
    # ---------------------------------------------------------

    preprocessor = (
        PointCloudPreprocessor(
            config
        )
    )

    processed_clouds = [
        preprocessor.preprocess(cloud)
        for cloud in clouds
    ]

    # ---------------------------------------------------------
    # Alle fünf Verfahren
    # ---------------------------------------------------------

    algorithms = [
        PointToPointICP(config),
        PointToPlaneICP(config),
        GeneralizedICP(config),
        FgrFPFHICP(config),
        RansacFPFHICP(config),
    ]

    print()
    print("=" * 120)
    print("PAIRWISE MULTI-ALGORITHM DIAGNOSE")
    print("=" * 120)

    # ---------------------------------------------------------
    # Edges 0->1, 1->2 und 2->3
    # ---------------------------------------------------------

    for source_id in range(
        len(processed_clouds) - 1
    ):

        target_id = (
            source_id + 1
        )

        source = (
            processed_clouds[source_id]
        )

        target = (
            processed_clouds[target_id]
        )

        # -----------------------------------------------------
        # Ground Truth source -> target
        # -----------------------------------------------------

        ground_truth_transform = (
            acquisition_transforms[target_id]
            @ np.linalg.inv(
                acquisition_transforms[
                    source_id
                ]
            )
        )

        print()
        print("=" * 120)
        print(
            f"EDGE {source_id} -> {target_id}"
        )
        print("=" * 120)

        print()
        print(
            f"{'Algorithmus':<26}"
            f"{'Fitness':>12}"
            f"{'RMSE':>12}"
            f"{'Trans. Error':>16}"
            f"{'Rot. Error':>16}"
        )

        print("-" * 120)

        # -----------------------------------------------------
        # Algorithmen ausführen
        # -----------------------------------------------------

        for algorithm_index, algorithm in enumerate(
            algorithms
        ):

            # Reproduzierbarer Seed
            seed = (
                42
                + source_id * 100
                + algorithm_index
            )

            o3d.utility.random.seed(
                seed
            )

            result = algorithm.register(
                source,
                target,
            )

            result = (
                RegistrationEvaluator
                .evaluate_result(
                    result=result,
                    source=source,
                    target=target,
                    config=config,
                )
            )

            (
                translation_error,
                rotation_error,
            ) = calculate_transform_error(
                estimated=(
                    result.transformation
                ),
                ground_truth=(
                    ground_truth_transform
                ),
            )

            print(
                f"{algorithm.name:<26}"
                f"{result.fitness:>12.6f}"
                f"{result.inlier_rmse:>12.6f}"
                f"{translation_error:>16.6f}"
                f"{rotation_error:>15.6f}°"
            )

        print()


if __name__ == "__main__":
    main()