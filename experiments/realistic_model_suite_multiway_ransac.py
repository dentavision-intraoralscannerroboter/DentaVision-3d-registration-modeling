from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd

from registration3d.algorithms.ransac_fpfh_icp import (
    RansacFPFHICP,
)
from registration3d.config import (
    RegistrationConfig,
)
from registration3d.evaluator import (
    RegistrationEvaluator,
)
from registration3d.multiway_evaluator import (
    MultiwayEvaluator,
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
# CONFIG
# ============================================================

BASE_SEED = 42

ADJACENT_PAIRS = (
    (0, 1),
    (1, 2),
    (2, 3),
)

MERGE_VOXEL_SIZE = 0.5


def create_registration_config() -> RegistrationConfig:

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
# HELPERS
# ============================================================

def load_cloud(
    path: Path,
) -> o3d.geometry.PointCloud:

    if not path.exists():
        raise FileNotFoundError(
            f"Missing point cloud:\n{path}"
        )

    cloud = o3d.io.read_point_cloud(
        str(path)
    )

    if cloud.is_empty():
        raise RuntimeError(
            f"Could not load:\n{path}"
        )

    return cloud


def pair_ground_truth(
    acquisition_transforms: np.ndarray,
    source_index: int,
    target_index: int,
) -> np.ndarray:

    a_source = acquisition_transforms[
        source_index
    ]

    a_target = acquisition_transforms[
        target_index
    ]

    return (
        a_target
        @ np.linalg.inv(
            a_source
        )
    )


# ============================================================
# ONE JAW
# ============================================================

def run_one_jaw(
    project_root: Path,
    model_id: str,
    jaw: str,
    model_index: int,
    jaw_index: int,
    config: RegistrationConfig,
) -> tuple[
    list[dict],
    pd.DataFrame,
]:

    data_directory = (
        get_generated_directory(
            project_root=project_root,
            model_id=model_id,
            jaw=jaw,
        )
    )

    output_directory = (
        project_root
        / "output"
        / "realistic_model_suite_multiway_ransac"
        / model_id
        / jaw
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 100)
    print(
        f"{model_id} / {jaw.upper()}"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    raw_clouds = []

    for fragment_index in range(4):

        cloud = load_cloud(
            data_directory
            / f"fragment_{fragment_index:02d}.ply"
        )

        raw_clouds.append(
            cloud
        )

    acquisition_transforms = np.load(
        data_directory
        / "acquisition_transforms.npy"
    )

    ground_truth_poses = np.load(
        data_directory
        / "ground_truth_poses.npy"
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    preprocessor = (
        PointCloudPreprocessor(
            config
        )
    )

    processed_clouds = []

    for index, cloud in enumerate(
        raw_clouds
    ):

        processed = (
            preprocessor.preprocess(
                cloud
            )
        )

        processed_clouds.append(
            processed
        )

        print(
            f"Fragment {index}: "
            f"{len(cloud.points):,} raw -> "
            f"{len(processed.points):,} processed"
        )

    # --------------------------------------------------------
    # Sequential poses
    #
    # P_i maps fragment i into fragment-0/world frame.
    # A_0 = I, therefore this is also identical to the
    # stored ground-truth frame.
    # --------------------------------------------------------

    estimated_poses = [
        np.eye(
            4,
            dtype=float,
        )
    ]

    pair_rows = []

    # --------------------------------------------------------
    # Adjacent registrations
    # --------------------------------------------------------

    for pair_index, (
        source_index,
        target_index,
    ) in enumerate(
        ADJACENT_PAIRS
    ):

        source = processed_clouds[
            source_index
        ]

        target = processed_clouds[
            target_index
        ]

        gt_pair = pair_ground_truth(
            acquisition_transforms=(
                acquisition_transforms
            ),
            source_index=source_index,
            target_index=target_index,
        )

        seed = (
            BASE_SEED
            + model_index * 10_000
            + jaw_index * 1_000
            + pair_index * 100
        )

        o3d.utility.random.seed(
            seed
        )

        algorithm = (
            RansacFPFHICP(
                config
            )
        )

        print()
        print(
            f"Registering "
            f"{source_index}->{target_index} "
            f"(seed={seed}) ..."
        )

        result = (
            algorithm.register(
                source,
                target,
            )
        )

        result = (
            RegistrationEvaluator
            .evaluate_result(
                result=result,
                source=source,
                target=target,
                config=config,
                ground_truth=gt_pair,
            )
        )

        print(
            f"  fitness = "
            f"{result.fitness:.6f}"
        )

        print(
            f"  RMSE    = "
            f"{result.inlier_rmse:.6f}"
        )

        print(
            f"  T error = "
            f"{result.translation_error:.6f}"
        )

        print(
            f"  R error = "
            f"{result.rotation_error_degrees:.6f} deg"
        )

        pair_rows.append(
            {
                "Model":
                    model_id,

                "Jaw":
                    jaw,

                "Pair":
                    (
                        f"{source_index}"
                        f"->{target_index}"
                    ),

                "Seed":
                    seed,

                "Fitness":
                    result.fitness,

                "Inlier RMSE":
                    result.inlier_rmse,

                "Translation Error":
                    result.translation_error,

                "Rotation Error [deg]":
                    result.rotation_error_degrees,

                "Runtime [s]":
                    result.runtime_seconds,
            }
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # result.transformation maps
        #
        #   source -> target.
        #
        # To map the target into the global reference frame:
        #
        #   P_target =
        #       P_source @ inv(T_source_to_target)
        # ----------------------------------------------------

        target_pose = (
            estimated_poses[
                source_index
            ]
            @ np.linalg.inv(
                result.transformation
            )
        )

        estimated_poses.append(
            target_pose
        )

    estimated_poses_array = (
        np.stack(
            estimated_poses,
            axis=0,
        )
    )

    np.save(
        output_directory
        / "estimated_poses.npy",
        estimated_poses_array,
    )

    # --------------------------------------------------------
    # Global pose errors
    # --------------------------------------------------------

    pose_errors = (
        MultiwayEvaluator.evaluate_poses(
            estimated_poses=(
                estimated_poses
            ),
            ground_truth_poses=(
                ground_truth_poses
            ),
        )
    )

    pose_errors.insert(
        0,
        "Jaw",
        jaw,
    )

    pose_errors.insert(
        0,
        "Model",
        model_id,
    )

    pose_errors.to_csv(
        output_directory
        / "pose_errors.csv",
        index=False,
    )

    print()
    print(
        "GLOBAL POSE ERRORS"
    )

    print(
        pose_errors.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    # --------------------------------------------------------
    # Apply global poses to RAW point clouds
    # --------------------------------------------------------

    transformed_clouds = []

    combined_cloud = (
        o3d.geometry.PointCloud()
    )

    for index, (
        cloud,
        pose,
    ) in enumerate(
        zip(
            raw_clouds,
            estimated_poses,
        )
    ):

        transformed = copy.deepcopy(
            cloud
        )

        transformed.transform(
            pose
        )

        transformed_clouds.append(
            transformed
        )

        combined_cloud += (
            transformed
        )

        o3d.io.write_point_cloud(
            str(
                output_directory
                / (
                    f"registered_fragment_"
                    f"{index:02d}.ply"
                )
            ),
            transformed,
        )

    # --------------------------------------------------------
    # Merge/downsample
    # --------------------------------------------------------

    combined_cloud = (
        combined_cloud.voxel_down_sample(
            MERGE_VOXEL_SIZE
        )
    )

    combined_path = (
        output_directory
        / "combined_cloud.ply"
    )

    o3d.io.write_point_cloud(
        str(
            combined_path
        ),
        combined_cloud,
    )

    print()
    print(
        f"Combined cloud: "
        f"{len(combined_cloud.points):,} points"
    )

    # --------------------------------------------------------
    # Save pairwise table
    # --------------------------------------------------------

    pair_df = pd.DataFrame(
        pair_rows
    )

    pair_df.to_csv(
        output_directory
        / "pairwise_edges.csv",
        index=False,
    )

    return (
        pair_rows,
        pose_errors,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    aggregate_directory = (
        project_root
        / "output"
        / "realistic_model_suite_multiway_ransac"
    )

    aggregate_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = (
        create_registration_config()
    )

    all_pair_rows = []

    all_pose_tables = []

    # --------------------------------------------------------
    # All five model sets
    # --------------------------------------------------------

    for model_index, model in enumerate(
        DENTAL_MODELS
    ):

        for jaw_index, jaw in enumerate(
            JAW_NAMES
        ):

            pair_rows, pose_errors = (
                run_one_jaw(
                    project_root=project_root,
                    model_id=model.model_id,
                    jaw=jaw,
                    model_index=model_index,
                    jaw_index=jaw_index,
                    config=config,
                )
            )

            all_pair_rows.extend(
                pair_rows
            )

            all_pose_tables.append(
                pose_errors
            )

    # --------------------------------------------------------
    # Aggregate output
    # --------------------------------------------------------

    pair_raw = pd.DataFrame(
        all_pair_rows
    )

    pose_raw = pd.concat(
        all_pose_tables,
        ignore_index=True,
    )

    pair_raw.to_csv(
        aggregate_directory
        / "all_pairwise_edges.csv",
        index=False,
    )

    pose_raw.to_csv(
        aggregate_directory
        / "all_pose_errors.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Automatically detect MultiwayEvaluator column names
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "POSE ERROR COLUMNS"
    )
    print("=" * 100)

    print(
        list(
            pose_raw.columns
        )
    )

    translation_candidates = [
        "Translation Error",
        "Translation Error [model units]",
        "translation_error",
        "Translation_Error",
    ]

    rotation_candidates = [
        "Rotation Error [deg]",
        "Rotation Error",
        "rotation_error_degrees",
        "Rotation_Error",
    ]

    translation_column = next(
        (
            column
            for column
            in translation_candidates
            if column in pose_raw.columns
        ),
        None,
    )

    rotation_column = next(
        (
            column
            for column
            in rotation_candidates
            if column in pose_raw.columns
        ),
        None,
    )

    if (
        translation_column is None
        or rotation_column is None
    ):

        print()
        print(
            "Could not automatically identify "
            "pose-error columns."
        )

        print(
            "Raw CSVs were still saved correctly."
        )

        return

    # --------------------------------------------------------
    # Exclude fragment 0 for accumulated-error summaries
    # because it is the fixed reference and exactly zero.
    # --------------------------------------------------------

    fragment_candidates = [
        "Fragment",
        "Fragment ID",
        "fragment",
        "fragment_id",
    ]

    fragment_column = next(
        (
            column
            for column
            in fragment_candidates
            if column in pose_raw.columns
        ),
        None,
    )

    if fragment_column is not None:

        evaluated_poses = (
            pose_raw[
                pose_raw[
                    fragment_column
                ]
                != 0
            ]
            .copy()
        )

    else:

        evaluated_poses = (
            pose_raw.copy()
        )

    # --------------------------------------------------------
    # Summary by model
    # --------------------------------------------------------

    by_model = (
        evaluated_poses.groupby(
            "Model",
            as_index=False,
        )
        .agg(
            Translation_Mean=(
                translation_column,
                "mean",
            ),
            Translation_Max=(
                translation_column,
                "max",
            ),
            Rotation_Mean=(
                rotation_column,
                "mean",
            ),
            Rotation_Max=(
                rotation_column,
                "max",
            ),
        )
    )

    # --------------------------------------------------------
    # Summary by jaw
    # --------------------------------------------------------

    by_jaw = (
        evaluated_poses.groupby(
            "Jaw",
            as_index=False,
        )
        .agg(
            Translation_Mean=(
                translation_column,
                "mean",
            ),
            Translation_Max=(
                translation_column,
                "max",
            ),
            Rotation_Mean=(
                rotation_column,
                "mean",
            ),
            Rotation_Max=(
                rotation_column,
                "max",
            ),
        )
    )

    # --------------------------------------------------------
    # Global summary
    # --------------------------------------------------------

    overall = pd.DataFrame(
        [
            {
                "Jaw Meshes":
                    len(DENTAL_MODELS)
                    * len(JAW_NAMES),

                "Evaluated Non-Reference Poses":
                    len(
                        evaluated_poses
                    ),

                "Translation Mean":
                    evaluated_poses[
                        translation_column
                    ].mean(),

                "Translation Median":
                    evaluated_poses[
                        translation_column
                    ].median(),

                "Translation Max":
                    evaluated_poses[
                        translation_column
                    ].max(),

                "Rotation Mean":
                    evaluated_poses[
                        rotation_column
                    ].mean(),

                "Rotation Median":
                    evaluated_poses[
                        rotation_column
                    ].median(),

                "Rotation Max":
                    evaluated_poses[
                        rotation_column
                    ].max(),
            }
        ]
    )

    by_model.to_csv(
        aggregate_directory
        / "pose_summary_by_model.csv",
        index=False,
    )

    by_jaw.to_csv(
        aggregate_directory
        / "pose_summary_by_jaw.csv",
        index=False,
    )

    overall.to_csv(
        aggregate_directory
        / "pose_summary_overall.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print(
        "PAIRWISE RANSAC EDGES"
    )
    print("=" * 100)

    print()

    print(
        pair_raw.groupby(
            "Pair"
        )[
            [
                "Translation Error",
                "Rotation Error [deg]",
            ]
        ]
        .agg(
            [
                "mean",
                "median",
                "max",
            ]
        )
        .to_string(
            float_format=lambda value: (
                f"{value:.6f}"
            )
        )
    )

    print()
    print("=" * 100)
    print(
        "OVERALL MULTIWAY POSE ERROR"
    )
    print("=" * 100)

    print()

    print(
        overall.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("=" * 100)
    print(
        "BY MODEL"
    )
    print("=" * 100)

    print()

    print(
        by_model.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )


if __name__ == "__main__":
    main()