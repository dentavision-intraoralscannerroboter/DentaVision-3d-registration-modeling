from pathlib import Path

import numpy as np
import open3d as o3d

from registration3d.algorithms.fgr_fpfh_icp import (
    FgrFPFHICP,
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


NUMBER_OF_RUNS = 20

SUCCESS_TRANSLATION_THRESHOLD = 1.0
SUCCESS_ROTATION_THRESHOLD = 2.0


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
    # Punktwolken laden
    # ---------------------------------------------------------

    fragment_paths = sorted(
        data_directory.glob(
            "fragment_*.ply"
        )
    )

    clouds = [
        PointCloudLoader.load(path)
        for path in fragment_paths
    ]

    acquisition_transforms = np.load(
        data_directory
        / "acquisition_transforms.npy"
    )

    # ---------------------------------------------------------
    # Registrierungskonfiguration
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

    algorithms = [
        (
            "FGR",
            FgrFPFHICP(config),
        ),
        (
            "RANSAC",
            RansacFPFHICP(config),
        ),
    ]

    print()
    print("=" * 100)
    print("GLOBAL REGISTRATION SEED STABILITY")
    print("=" * 100)

    # ---------------------------------------------------------
    # Alle benachbarten Edges
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

        ground_truth = (
            acquisition_transforms[target_id]
            @ np.linalg.inv(
                acquisition_transforms[
                    source_id
                ]
            )
        )

        print()
        print("=" * 100)
        print(
            f"EDGE {source_id} -> {target_id}"
        )
        print("=" * 100)

        for (
            algorithm_name,
            algorithm,
        ) in algorithms:

            successful_runs = 0

            translation_errors = []
            rotation_errors = []

            print()
            print(
                f"{algorithm_name}:"
            )

            for run_index in range(
                NUMBER_OF_RUNS
            ):

                seed = (
                    42
                    + run_index
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
                    ground_truth=ground_truth,
                )

                translation_errors.append(
                    translation_error
                )

                rotation_errors.append(
                    rotation_error
                )

                success = (
                    translation_error
                    <= SUCCESS_TRANSLATION_THRESHOLD

                    and

                    rotation_error
                    <= SUCCESS_ROTATION_THRESHOLD
                )

                if success:
                    successful_runs += 1

                status = (
                    "OK"
                    if success
                    else "FAIL"
                )

                print(
                    f"  Seed {seed:>3} "
                    f"| {status:<4} "
                    f"| T={translation_error:>8.4f} "
                    f"| R={rotation_error:>8.4f}° "
                    f"| Fitness={result.fitness:.4f} "
                    f"| RMSE={result.inlier_rmse:.4f}"
                )

            success_rate = (
                successful_runs
                / NUMBER_OF_RUNS
                * 100.0
            )

            print()
            print(
                f"  Erfolg: "
                f"{successful_runs}/"
                f"{NUMBER_OF_RUNS} "
                f"({success_rate:.1f} %)"
            )

            print(
                f"  Translation Error Median: "
                f"{np.median(translation_errors):.6f}"
            )

            print(
                f"  Rotation Error Median: "
                f"{np.median(rotation_errors):.6f}°"
            )


if __name__ == "__main__":
    main()