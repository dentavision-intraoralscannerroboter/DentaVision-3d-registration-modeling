from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import open3d as o3d
import pandas as pd


# ---------------------------------------------------------------------
# Konfiguration
# Entspricht der bisherigen RegistrationConfig
# ---------------------------------------------------------------------

VOXEL_SIZE = 0.5

NORMAL_RADIUS_FACTOR = 2.0
NORMAL_MAX_NN = 30

FPFH_RADIUS_FACTOR = 5.0
FPFH_MAX_NN = 100

CORRESPONDENCE_DISTANCE_FACTOR = 2.0

RANSAC_DISTANCE_FACTOR = 1.5
RANSAC_N = 3
RANSAC_MAX_ITERATIONS = 100_000
RANSAC_CONFIDENCE = 0.999

FGR_DISTANCE_FACTOR = 0.5

REFINEMENT_DISTANCE_FACTOR = 0.4

EVALUATION_DISTANCE_FACTOR = 2.0

ICP_MAX_ITERATIONS = 100

SEED = 42


BASE_DIR = Path("data/realistic_jaw")
OUTPUT_DIR = Path("output/realistic_registration")

JAW_DIRS = {
    "upper": BASE_DIR / "upper_test",
    "lower": BASE_DIR / "lower_test",
}

PAIRS = [
    (0, 1),
    (1, 2),
    (2, 3),
]


# ---------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------

def load_cloud(path: Path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))

    if cloud.is_empty():
        raise RuntimeError(f"Punktwolke konnte nicht geladen werden: {path}")

    return cloud


def preprocess(
    cloud: o3d.geometry.PointCloud,
) -> tuple[
    o3d.geometry.PointCloud,
    o3d.pipelines.registration.Feature,
]:

    downsampled = cloud.voxel_down_sample(
        voxel_size=VOXEL_SIZE
    )

    normal_radius = (
        VOXEL_SIZE * NORMAL_RADIUS_FACTOR
    )

    downsampled.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=NORMAL_MAX_NN,
        )
    )

    fpfh_radius = (
        VOXEL_SIZE * FPFH_RADIUS_FACTOR
    )

    fpfh = (
        o3d.pipelines.registration
        .compute_fpfh_feature(
            downsampled,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=fpfh_radius,
                max_nn=FPFH_MAX_NN,
            ),
        )
    )

    return downsampled, fpfh


def translation_error(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
) -> float:

    estimated_translation = estimated[:3, 3]
    ground_truth_translation = ground_truth[:3, 3]

    return float(
        np.linalg.norm(
            estimated_translation
            - ground_truth_translation
        )
    )


def rotation_error_deg(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
) -> float:

    r_est = estimated[:3, :3]
    r_gt = ground_truth[:3, :3]

    delta = r_est @ r_gt.T

    value = (
        np.trace(delta) - 1.0
    ) / 2.0

    value = np.clip(
        value,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(value)
        )
    )


def evaluate(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    ground_truth: np.ndarray,
) -> dict:

    evaluation_distance = (
        VOXEL_SIZE
        * EVALUATION_DISTANCE_FACTOR
    )

    evaluation = (
        o3d.pipelines.registration
        .evaluate_registration(
            source,
            target,
            evaluation_distance,
            transformation,
        )
    )

    return {
        "fitness": float(evaluation.fitness),
        "inlier_rmse": float(
            evaluation.inlier_rmse
        ),
        "translation_error": translation_error(
            transformation,
            ground_truth,
        ),
        "rotation_error_deg": rotation_error_deg(
            transformation,
            ground_truth,
        ),
    }


# ---------------------------------------------------------------------
# Registrierungsverfahren
# ---------------------------------------------------------------------

def point_to_point_icp(
    source,
    target,
):

    distance = (
        VOXEL_SIZE
        * CORRESPONDENCE_DISTANCE_FACTOR
    )

    return (
        o3d.pipelines.registration
        .registration_icp(
            source,
            target,
            distance,
            np.eye(4),
            o3d.pipelines.registration
            .TransformationEstimationPointToPoint(),
            o3d.pipelines.registration
            .ICPConvergenceCriteria(
                max_iteration=ICP_MAX_ITERATIONS
            ),
        )
    )


def point_to_plane_icp(
    source,
    target,
):

    distance = (
        VOXEL_SIZE
        * CORRESPONDENCE_DISTANCE_FACTOR
    )

    return (
        o3d.pipelines.registration
        .registration_icp(
            source,
            target,
            distance,
            np.eye(4),
            o3d.pipelines.registration
            .TransformationEstimationPointToPlane(),
            o3d.pipelines.registration
            .ICPConvergenceCriteria(
                max_iteration=ICP_MAX_ITERATIONS
            ),
        )
    )


def generalized_icp(
    source,
    target,
):

    distance = (
        VOXEL_SIZE
        * CORRESPONDENCE_DISTANCE_FACTOR
    )

    return (
        o3d.pipelines.registration
        .registration_generalized_icp(
            source,
            target,
            distance,
            np.eye(4),
            o3d.pipelines.registration
            .TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration
            .ICPConvergenceCriteria(
                max_iteration=ICP_MAX_ITERATIONS
            ),
        )
    )


def refine_with_point_to_plane(
    source,
    target,
    initial_transform,
):

    distance = (
        VOXEL_SIZE
        * REFINEMENT_DISTANCE_FACTOR
    )

    return (
        o3d.pipelines.registration
        .registration_icp(
            source,
            target,
            distance,
            initial_transform,
            o3d.pipelines.registration
            .TransformationEstimationPointToPlane(),
            o3d.pipelines.registration
            .ICPConvergenceCriteria(
                max_iteration=ICP_MAX_ITERATIONS
            ),
        )
    )


def ransac_fpfh_icp(
    source,
    target,
    source_fpfh,
    target_fpfh,
):

    distance = (
        VOXEL_SIZE
        * RANSAC_DISTANCE_FACTOR
    )

    o3d.utility.random.seed(SEED)

    coarse = (
        o3d.pipelines.registration
        .registration_ransac_based_on_feature_matching(
            source,
            target,
            source_fpfh,
            target_fpfh,
            True,
            distance,
            o3d.pipelines.registration
            .TransformationEstimationPointToPoint(
                False
            ),
            RANSAC_N,
            [
                o3d.pipelines.registration
                .CorrespondenceCheckerBasedOnEdgeLength(
                    0.9
                ),
                o3d.pipelines.registration
                .CorrespondenceCheckerBasedOnDistance(
                    distance
                ),
            ],
            o3d.pipelines.registration
            .RANSACConvergenceCriteria(
                RANSAC_MAX_ITERATIONS,
                RANSAC_CONFIDENCE,
            ),
        )
    )

    return refine_with_point_to_plane(
        source,
        target,
        coarse.transformation,
    )


def fgr_fpfh_icp(
    source,
    target,
    source_fpfh,
    target_fpfh,
):

    distance = (
        VOXEL_SIZE
        * FGR_DISTANCE_FACTOR
    )

    o3d.utility.random.seed(SEED)

    coarse = (
        o3d.pipelines.registration
        .registration_fgr_based_on_feature_matching(
            source,
            target,
            source_fpfh,
            target_fpfh,
            o3d.pipelines.registration
            .FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance
            ),
        )
    )

    return refine_with_point_to_plane(
        source,
        target,
        coarse.transformation,
    )


# ---------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------

def benchmark_jaw(
    jaw_name: str,
    jaw_dir: Path,
) -> list[dict]:

    print()
    print("=" * 80)
    print(f"{jaw_name.upper()} JAW")
    print("=" * 80)

    acquisition_transforms = np.load(
        jaw_dir
        / "acquisition_transforms.npy"
    )

    prepared = {}

    for index in range(4):

        path = (
            jaw_dir
            / f"fragment_{index:02d}.ply"
        )

        print(
            f"Preprocessing Fragment {index} ..."
        )

        cloud = load_cloud(path)

        downsampled, fpfh = preprocess(
            cloud
        )

        prepared[index] = {
            "cloud": downsampled,
            "fpfh": fpfh,
        }

        print(
            f"  {len(cloud.points):,} -> "
            f"{len(downsampled.points):,} Punkte"
        )

    rows = []

    for source_index, target_index in PAIRS:

        print()
        print(
            f"--- {source_index} -> "
            f"{target_index} ---"
        )

        source = prepared[source_index]["cloud"]
        target = prepared[target_index]["cloud"]

        source_fpfh = prepared[source_index]["fpfh"]
        target_fpfh = prepared[target_index]["fpfh"]

        # x_i = A_i * x_world
        #
        # source i -> target j:
        #
        # T_gt = A_j * inv(A_i)

        ground_truth = (
            acquisition_transforms[target_index]
            @ np.linalg.inv(
                acquisition_transforms[source_index]
            )
        )

        algorithms = [
            (
                "Point-to-Point ICP",
                lambda: point_to_point_icp(
                    source,
                    target,
                ),
            ),
            (
                "Point-to-Plane ICP",
                lambda: point_to_plane_icp(
                    source,
                    target,
                ),
            ),
            (
                "Generalized ICP",
                lambda: generalized_icp(
                    source,
                    target,
                ),
            ),
            (
                "FPFH + RANSAC + ICP",
                lambda: ransac_fpfh_icp(
                    source,
                    target,
                    source_fpfh,
                    target_fpfh,
                ),
            ),
            (
                "FPFH + FGR + ICP",
                lambda: fgr_fpfh_icp(
                    source,
                    target,
                    source_fpfh,
                    target_fpfh,
                ),
            ),
        ]

        for algorithm_name, algorithm in algorithms:

            print(
                f"{algorithm_name:<25}",
                end="",
                flush=True,
            )

            start = time.perf_counter()

            result = algorithm()

            runtime = (
                time.perf_counter()
                - start
            )

            metrics = evaluate(
                source=source,
                target=target,
                transformation=result.transformation,
                ground_truth=ground_truth,
            )

            row = {
                "jaw": jaw_name,
                "source": source_index,
                "target": target_index,
                "algorithm": algorithm_name,
                "fitness": metrics["fitness"],
                "inlier_rmse": metrics[
                    "inlier_rmse"
                ],
                "translation_error": metrics[
                    "translation_error"
                ],
                "rotation_error_deg": metrics[
                    "rotation_error_deg"
                ],
                "runtime_s": runtime,
            }

            rows.append(row)

            print(
                f" fitness={row['fitness']:.4f}"
                f" rmse={row['inlier_rmse']:.4f}"
                f" t_err={row['translation_error']:.4f}"
                f" r_err={row['rotation_error_deg']:.4f}°"
                f" time={row['runtime_s']:.3f}s"
            )

    return rows


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for jaw_name, jaw_dir in JAW_DIRS.items():

        rows.extend(
            benchmark_jaw(
                jaw_name,
                jaw_dir,
            )
        )

    dataframe = pd.DataFrame(rows)

    raw_path = (
        OUTPUT_DIR
        / "realistic_registration_results.csv"
    )

    dataframe.to_csv(
        raw_path,
        index=False,
    )

    summary = (
        dataframe
        .groupby("algorithm")
        .agg(
            mean_fitness=(
                "fitness",
                "mean",
            ),
            mean_inlier_rmse=(
                "inlier_rmse",
                "mean",
            ),
            mean_translation_error=(
                "translation_error",
                "mean",
            ),
            median_translation_error=(
                "translation_error",
                "median",
            ),
            mean_rotation_error_deg=(
                "rotation_error_deg",
                "mean",
            ),
            median_rotation_error_deg=(
                "rotation_error_deg",
                "median",
            ),
            mean_runtime_s=(
                "runtime_s",
                "mean",
            ),
        )
        .reset_index()
    )

    summary_path = (
        OUTPUT_DIR
        / "realistic_registration_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("GESAMTERGEBNIS")
    print("=" * 80)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print(
        f"Einzelergebnisse: {raw_path}"
    )

    print(
        f"Zusammenfassung: {summary_path}"
    )


if __name__ == "__main__":
    main()