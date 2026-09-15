from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd

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
from registration3d.evaluator import (
    RegistrationEvaluator,
)
from registration3d.preprocessing import (
    PointCloudPreprocessor,
)
from registration3d.realistic_dataset_registry import (
    DENTAL_MODELS,
    JAW_NAMES,
    get_generated_directory,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Same criterion as the previous realistic validation.
SUCCESS_TRANSLATION_THRESHOLD = 1.0
SUCCESS_ROTATION_THRESHOLD_DEGREES = 1.0

BASE_ALGORITHM_SEED = 42

ADJACENT_PAIRS = (
    (0, 1),
    (1, 2),
    (2, 3),
)


# ============================================================
# REGISTRATION CONFIGURATION
# ============================================================

def create_registration_config(
) -> RegistrationConfig:

    return RegistrationConfig(
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


# ============================================================
# ALGORITHMS
# ============================================================

def create_algorithms(
    config: RegistrationConfig,
):

    return [
        PointToPointICP(
            config
        ),
        PointToPlaneICP(
            config
        ),
        GeneralizedICP(
            config
        ),
        FgrFPFHICP(
            config
        ),
        RansacFPFHICP(
            config
        ),
    ]


# ============================================================
# HELPERS
# ============================================================

def load_cloud(
    path: Path,
) -> o3d.geometry.PointCloud:

    if not path.exists():

        raise FileNotFoundError(
            f"Point cloud not found:\n{path}"
        )

    cloud = (
        o3d.io.read_point_cloud(
            str(path)
        )
    )

    if cloud.is_empty():

        raise RuntimeError(
            f"Could not load point cloud:\n{path}"
        )

    return cloud


def calculate_pair_ground_truth(
    acquisition_transforms: np.ndarray,
    source_index: int,
    target_index: int,
) -> np.ndarray:

    """
    Acquisition convention:

        x_observed_i = A_i @ x_world

    We require the transformation from observed source i
    to observed target j:

        x_j = A_j @ inv(A_i) @ x_i

    Therefore:

        T_source_to_target = A_j @ inv(A_i)
    """

    source_acquisition = (
        acquisition_transforms[
            source_index
        ]
    )

    target_acquisition = (
        acquisition_transforms[
            target_index
        ]
    )

    return (
        target_acquisition
        @ np.linalg.inv(
            source_acquisition
        )
    )


def is_success(
    translation_error: float | None,
    rotation_error_degrees: float | None,
) -> bool:

    if (
        translation_error is None
        or rotation_error_degrees is None
    ):
        return False

    return bool(
        translation_error
        <= SUCCESS_TRANSLATION_THRESHOLD
        and
        rotation_error_degrees
        <= SUCCESS_ROTATION_THRESHOLD_DEGREES
    )


def valid_rmse(
    fitness: float,
    rmse: float,
) -> float | None:

    # Open3D can report RMSE = 0 when no
    # valid correspondences exist.
    if fitness <= 0.0:
        return None

    return float(
        rmse
    )


# ============================================================
# SUMMARY TABLES
# ============================================================

def create_overall_summary(
    raw: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        raw.groupby(
            "Algorithm",
            as_index=False,
        )
        .agg(
            Cases=(
                "Success",
                "size",
            ),
            Successes=(
                "Success",
                "sum",
            ),
            Success_Rate=(
                "Success",
                "mean",
            ),
            Fitness_Mean=(
                "Fitness",
                "mean",
            ),
            RMSE_Mean=(
                "Inlier RMSE",
                "mean",
            ),
            Translation_Error_Mean=(
                "Translation Error",
                "mean",
            ),
            Translation_Error_Median=(
                "Translation Error",
                "median",
            ),
            Rotation_Error_Mean=(
                "Rotation Error [deg]",
                "mean",
            ),
            Rotation_Error_Median=(
                "Rotation Error [deg]",
                "median",
            ),
            Runtime_Mean=(
                "Runtime [s]",
                "mean",
            ),
        )
    )

    summary[
        "Success_Rate"
    ] *= 100.0

    return summary


def create_model_summary(
    raw: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        raw.groupby(
            [
                "Model",
                "Algorithm",
            ],
            as_index=False,
        )
        .agg(
            Cases=(
                "Success",
                "size",
            ),
            Successes=(
                "Success",
                "sum",
            ),
            Success_Rate=(
                "Success",
                "mean",
            ),
            Translation_Error_Mean=(
                "Translation Error",
                "mean",
            ),
            Rotation_Error_Mean=(
                "Rotation Error [deg]",
                "mean",
            ),
            Runtime_Mean=(
                "Runtime [s]",
                "mean",
            ),
        )
    )

    summary[
        "Success_Rate"
    ] *= 100.0

    return summary


def create_jaw_summary(
    raw: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        raw.groupby(
            [
                "Jaw",
                "Algorithm",
            ],
            as_index=False,
        )
        .agg(
            Cases=(
                "Success",
                "size",
            ),
            Successes=(
                "Success",
                "sum",
            ),
            Success_Rate=(
                "Success",
                "mean",
            ),
            Translation_Error_Mean=(
                "Translation Error",
                "mean",
            ),
            Rotation_Error_Mean=(
                "Rotation Error [deg]",
                "mean",
            ),
            Runtime_Mean=(
                "Runtime [s]",
                "mean",
            ),
        )
    )

    summary[
        "Success_Rate"
    ] *= 100.0

    return summary


def create_pair_summary(
    raw: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        raw.groupby(
            [
                "Pair",
                "Algorithm",
            ],
            as_index=False,
        )
        .agg(
            Cases=(
                "Success",
                "size",
            ),
            Successes=(
                "Success",
                "sum",
            ),
            Success_Rate=(
                "Success",
                "mean",
            ),
            Translation_Error_Mean=(
                "Translation Error",
                "mean",
            ),
            Rotation_Error_Mean=(
                "Rotation Error [deg]",
                "mean",
            ),
        )
    )

    summary[
        "Success_Rate"
    ] *= 100.0

    return summary


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main() -> None:

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    output_directory = (
        project_root
        / "output"
        / "realistic_model_suite_pairwise"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = (
        create_registration_config()
    )

    preprocessor = (
        PointCloudPreprocessor(
            config
        )
    )

    rows = []

    total_jaws = (
        len(
            DENTAL_MODELS
        )
        * len(
            JAW_NAMES
        )
    )

    total_pairs = (
        total_jaws
        * len(
            ADJACENT_PAIRS
        )
    )

    total_runs = (
        total_pairs
        * 5
    )

    current_run = 0

    print()
    print("=" * 100)
    print(
        "REALISTIC FIVE-MODEL PAIRWISE BENCHMARK"
    )
    print("=" * 100)

    print()
    print(
        f"Models:             "
        f"{len(DENTAL_MODELS)}"
    )

    print(
        f"Jaw meshes:         "
        f"{total_jaws}"
    )

    print(
        f"Adjacent pairs:     "
        f"{total_pairs}"
    )

    print(
        f"Algorithms:         5"
    )

    print(
        f"Total registrations:"
        f" {total_runs}"
    )

    print()
    print(
        "Success criterion:"
    )

    print(
        f"  Translation <= "
        f"{SUCCESS_TRANSLATION_THRESHOLD:.3f} "
        f"model units"
    )

    print(
        f"  Rotation    <= "
        f"{SUCCESS_ROTATION_THRESHOLD_DEGREES:.3f} deg"
    )

    # ========================================================
    # ALL MODELS
    # ========================================================

    for model_index, model in enumerate(
        DENTAL_MODELS
    ):

        for jaw_index, jaw in enumerate(
            JAW_NAMES
        ):

            print()
            print("=" * 100)

            print(
                f"{model.model_id} / "
                f"{jaw.upper()}"
            )

            print("=" * 100)

            data_directory = (
                get_generated_directory(
                    project_root=(
                        project_root
                    ),
                    model_id=(
                        model.model_id
                    ),
                    jaw=jaw,
                )
            )

            transform_path = (
                data_directory
                / "acquisition_transforms.npy"
            )

            if not transform_path.exists():

                raise FileNotFoundError(
                    "Missing acquisition "
                    f"transform file:\n"
                    f"{transform_path}"
                )

            acquisition_transforms = (
                np.load(
                    transform_path
                )
            )

            # ------------------------------------------------
            # Load + preprocess each fragment only once
            # ------------------------------------------------

            processed_clouds = []

            for fragment_index in range(
                4
            ):

                cloud_path = (
                    data_directory
                    / (
                        f"fragment_"
                        f"{fragment_index:02d}.ply"
                    )
                )

                cloud = (
                    load_cloud(
                        cloud_path
                    )
                )

                processed = (
                    preprocessor.preprocess(
                        cloud
                    )
                )

                processed_clouds.append(
                    processed
                )

                print(
                    f"Fragment {fragment_index}: "
                    f"{len(cloud.points):,} raw -> "
                    f"{len(processed.points):,} "
                    f"processed points"
                )

            # =================================================
            # ADJACENT PAIRS
            # =================================================

            for pair_index, (
                source_index,
                target_index,
            ) in enumerate(
                ADJACENT_PAIRS
            ):

                pair_name = (
                    f"{source_index}"
                    f"->{target_index}"
                )

                print()
                print("-" * 100)

                print(
                    f"PAIR {pair_name}"
                )

                print("-" * 100)

                source = (
                    processed_clouds[
                        source_index
                    ]
                )

                target = (
                    processed_clouds[
                        target_index
                    ]
                )

                ground_truth = (
                    calculate_pair_ground_truth(
                        acquisition_transforms=(
                            acquisition_transforms
                        ),
                        source_index=(
                            source_index
                        ),
                        target_index=(
                            target_index
                        ),
                    )
                )

                algorithms = (
                    create_algorithms(
                        config
                    )
                )

                for algorithm_index, algorithm in enumerate(
                    algorithms
                ):

                    current_run += 1

                    # -----------------------------------------
                    # Reproducible algorithm seed
                    # -----------------------------------------

                    algorithm_seed = (
                        BASE_ALGORITHM_SEED
                        + model_index
                        * 10_000
                        + jaw_index
                        * 1_000
                        + pair_index
                        * 100
                        + algorithm_index
                    )

                    o3d.utility.random.seed(
                        algorithm_seed
                    )

                    print()
                    print(
                        f"[{current_run:03d}/"
                        f"{total_runs:03d}] "
                        f"{algorithm.name}"
                    )

                    # -----------------------------------------
                    # Registration
                    # -----------------------------------------

                    result = (
                        algorithm.register(
                            source,
                            target,
                        )
                    )

                    # -----------------------------------------
                    # Ground-truth evaluation
                    # -----------------------------------------

                    result = (
                        RegistrationEvaluator
                        .evaluate_result(
                            result=result,
                            source=source,
                            target=target,
                            config=config,
                            ground_truth=(
                                ground_truth
                            ),
                        )
                    )

                    success = (
                        is_success(
                            result.translation_error,
                            (
                                result
                                .rotation_error_degrees
                            ),
                        )
                    )

                    rmse = (
                        valid_rmse(
                            fitness=result.fitness,
                            rmse=(
                                result.inlier_rmse
                            ),
                        )
                    )

                    print(
                        f"  Fitness: "
                        f"{result.fitness:.4f}"
                    )

                    if rmse is None:

                        print(
                            "  RMSE:    invalid"
                        )

                    else:

                        print(
                            f"  RMSE:    "
                            f"{rmse:.4f}"
                        )

                    print(
                        f"  T error: "
                        f"{result.translation_error:.6f}"
                    )

                    print(
                        f"  R error: "
                        f"{result.rotation_error_degrees:.6f} deg"
                    )

                    print(
                        f"  Runtime: "
                        f"{result.runtime_seconds:.4f} s"
                    )

                    print(
                        f"  Success: "
                        f"{success}"
                    )

                    rows.append(
                        {
                            "Model":
                                model.model_id,

                            "Jaw":
                                jaw,

                            "Pair":
                                pair_name,

                            "Source Fragment":
                                source_index,

                            "Target Fragment":
                                target_index,

                            "Algorithm":
                                result.algorithm,

                            "Algorithm Seed":
                                algorithm_seed,

                            "Fitness":
                                result.fitness,

                            "Inlier RMSE":
                                rmse,

                            "Translation Error":
                                result.translation_error,

                            "Rotation Error [deg]":
                                (
                                    result
                                    .rotation_error_degrees
                                ),

                            "Runtime [s]":
                                result.runtime_seconds,

                            "Success":
                                success,
                        }
                    )

    # ========================================================
    # DATA FRAME
    # ========================================================

    raw = pd.DataFrame(
        rows
    )

    if len(raw) != total_runs:

        raise RuntimeError(
            "Unexpected number of benchmark "
            f"rows: {len(raw)} != {total_runs}"
        )

    # ========================================================
    # SUMMARIES
    # ========================================================

    overall_summary = (
        create_overall_summary(
            raw
        )
    )

    model_summary = (
        create_model_summary(
            raw
        )
    )

    jaw_summary = (
        create_jaw_summary(
            raw
        )
    )

    pair_summary = (
        create_pair_summary(
            raw
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    raw_path = (
        output_directory
        / "pairwise_raw.csv"
    )

    overall_path = (
        output_directory
        / "pairwise_overall_summary.csv"
    )

    model_path = (
        output_directory
        / "pairwise_by_model.csv"
    )

    jaw_path = (
        output_directory
        / "pairwise_by_jaw.csv"
    )

    pair_path = (
        output_directory
        / "pairwise_by_pair.csv"
    )

    raw.to_csv(
        raw_path,
        index=False,
    )

    overall_summary.to_csv(
        overall_path,
        index=False,
    )

    model_summary.to_csv(
        model_path,
        index=False,
    )

    jaw_summary.to_csv(
        jaw_path,
        index=False,
    )

    pair_summary.to_csv(
        pair_path,
        index=False,
    )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print()
    print("=" * 100)
    print(
        "OVERALL RESULTS"
    )
    print("=" * 100)
    print()

    print(
        overall_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("=" * 100)
    print(
        "SUCCESS RATE BY MODEL"
    )
    print("=" * 100)
    print()

    model_success_pivot = (
        model_summary.pivot(
            index="Model",
            columns="Algorithm",
            values="Success_Rate",
        )
    )

    print(
        model_success_pivot.to_string(
            float_format=lambda value: (
                f"{value:.1f}"
            )
        )
    )

    print()
    print("=" * 100)
    print(
        "FILES"
    )
    print("=" * 100)

    for path in (
        raw_path,
        overall_path,
        model_path,
        jaw_path,
        pair_path,
    ):

        print(
            path
        )


if __name__ == "__main__":
    main()