from __future__ import annotations

import copy
import math
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
# EXPERIMENT CONFIGURATION
# ============================================================

NOISE_LEVELS = (
    0.000,
    0.010,
    0.025,
    0.050,
    0.100,
    0.200,
)

# sigma = 0:
# only one run because no random perturbation is present.
ZERO_NOISE_REPEATS = 1

# sigma > 0:
# independent Gaussian noise realizations.
NONZERO_NOISE_REPEATS = 3

SUCCESS_TRANSLATION_THRESHOLD = 1.0
SUCCESS_ROTATION_THRESHOLD_DEGREES = 1.0

BASE_NOISE_SEED = 10_000
BASE_ALGORITHM_SEED = 100_000

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
# BASIC HELPERS
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

    Therefore the transformation from observed
    source fragment i to observed target fragment j is:

        T_i_to_j = A_j @ inv(A_i)
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


def add_gaussian_noise(
    cloud: o3d.geometry.PointCloud,
    sigma: float,
    rng: np.random.Generator,
) -> o3d.geometry.PointCloud:

    noisy_cloud = copy.deepcopy(
        cloud
    )

    if sigma <= 0.0:
        return noisy_cloud

    points = np.asarray(
        noisy_cloud.points
    ).copy()

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=points.shape,
    )

    points += noise

    noisy_cloud.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )

    # Normals from the original cloud, if present,
    # are no longer geometrically consistent after noise.
    # The normal estimation is performed again during
    # preprocessing.
    if noisy_cloud.has_normals():

        noisy_cloud.normals = (
            o3d.utility.Vector3dVector(
                np.empty(
                    (
                        0,
                        3,
                    )
                )
            )
        )

    return noisy_cloud


def is_success(
    translation_error: float | None,
    rotation_error_degrees: float | None,
) -> bool:

    if (
        translation_error is None
        or rotation_error_degrees is None
    ):
        return False

    if (
        not np.isfinite(
            translation_error
        )
        or
        not np.isfinite(
            rotation_error_degrees
        )
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

    # Open3D can report RMSE = 0 if no valid
    # correspondences exist.
    if (
        fitness <= 0.0
        or
        not np.isfinite(
            rmse
        )
    ):
        return None

    return float(
        rmse
    )


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.96,
) -> tuple[
    float,
    float,
]:

    if trials <= 0:
        return (
            float("nan"),
            float("nan"),
        )

    p = (
        successes
        / trials
    )

    z_squared = (
        z ** 2
    )

    denominator = (
        1.0
        + z_squared
        / trials
    )

    center = (
        p
        + z_squared
        / (
            2.0
            * trials
        )
    ) / denominator

    margin = (
        z
        / denominator
        * math.sqrt(
            (
                p
                * (
                    1.0
                    - p
                )
                / trials
            )
            +
            (
                z_squared
                / (
                    4.0
                    * trials ** 2
                )
            )
        )
    )

    lower = max(
        0.0,
        center - margin,
    )

    upper = min(
        1.0,
        center + margin,
    )

    return (
        lower,
        upper,
    )


def add_wilson_columns(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    summary = summary.copy()

    lowers = []
    uppers = []

    for _, row in (
        summary.iterrows()
    ):

        successes = int(
            row[
                "Successes"
            ]
        )

        trials = int(
            row[
                "Cases"
            ]
        )

        lower, upper = (
            wilson_interval(
                successes=successes,
                trials=trials,
            )
        )

        lowers.append(
            lower
            * 100.0
        )

        uppers.append(
            upper
            * 100.0
        )

    summary[
        "Wilson_95_Lower"
    ] = lowers

    summary[
        "Wilson_95_Upper"
    ] = uppers

    return summary


# ============================================================
# CHECKPOINT / RESUME
# ============================================================

def normalized_key(
    model: str,
    jaw: str,
    noise_std: float,
    repeat: int,
    pair: str,
    algorithm: str,
) -> tuple:

    return (
        model,
        jaw,
        round(
            float(
                noise_std
            ),
            6,
        ),
        int(
            repeat
        ),
        pair,
        algorithm,
    )


def load_existing_results(
    raw_path: Path,
) -> tuple[
    list[
        dict
    ],
    set[
        tuple
    ],
]:

    if not raw_path.exists():

        return (
            [],
            set(),
        )

    existing = pd.read_csv(
        raw_path
    )

    rows = (
        existing
        .to_dict(
            orient="records"
        )
    )

    completed = set()

    for _, row in (
        existing.iterrows()
    ):

        key = normalized_key(
            model=row["Model"],
            jaw=row["Jaw"],
            noise_std=(
                row["Noise Std"]
            ),
            repeat=(
                row["Repeat"]
            ),
            pair=row["Pair"],
            algorithm=(
                row["Algorithm"]
            ),
        )

        completed.add(
            key
        )

    print()
    print(
        f"Resume enabled: "
        f"{len(rows)} existing "
        f"registration results found."
    )

    return (
        rows,
        completed,
    )


def save_checkpoint(
    rows: list[
        dict
    ],
    raw_path: Path,
) -> None:

    dataframe = pd.DataFrame(
        rows
    )

    dataframe.to_csv(
        raw_path,
        index=False,
    )


# ============================================================
# SUMMARY GENERATION
# ============================================================

def aggregate_results(
    raw: pd.DataFrame,
    group_columns: list[
        str
    ],
) -> pd.DataFrame:

    summary = (
        raw.groupby(
            group_columns,
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
            Fitness_Std=(
                "Fitness",
                "std",
            ),
            RMSE_Mean=(
                "Inlier RMSE",
                "mean",
            ),
            RMSE_Std=(
                "Inlier RMSE",
                "std",
            ),
            Translation_Error_Mean=(
                "Translation Error",
                "mean",
            ),
            Translation_Error_Median=(
                "Translation Error",
                "median",
            ),
            Translation_Error_Std=(
                "Translation Error",
                "std",
            ),
            Rotation_Error_Mean=(
                "Rotation Error [deg]",
                "mean",
            ),
            Rotation_Error_Median=(
                "Rotation Error [deg]",
                "median",
            ),
            Rotation_Error_Std=(
                "Rotation Error [deg]",
                "std",
            ),
            Runtime_Mean=(
                "Runtime [s]",
                "mean",
            ),
            Runtime_Std=(
                "Runtime [s]",
                "std",
            ),
        )
    )

    summary[
        "Success_Rate"
    ] *= 100.0

    summary = (
        add_wilson_columns(
            summary
        )
    )

    return summary


# ============================================================
# MAIN
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
        / "realistic_model_suite_noise"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_directory
        / "noise_raw.csv"
    )

    config = (
        create_registration_config()
    )

    preprocessor = (
        PointCloudPreprocessor(
            config
        )
    )

    # --------------------------------------------------------
    # Resume existing run if present
    # --------------------------------------------------------

    rows, completed = (
        load_existing_results(
            raw_path
        )
    )

    # --------------------------------------------------------
    # Calculate expected workload
    # --------------------------------------------------------

    number_of_jaws = (
        len(
            DENTAL_MODELS
        )
        * len(
            JAW_NAMES
        )
    )

    zero_noise_cases = (
        number_of_jaws
        * len(
            ADJACENT_PAIRS
        )
        * ZERO_NOISE_REPEATS
    )

    nonzero_noise_cases = (
        number_of_jaws
        * len(
            ADJACENT_PAIRS
        )
        * NONZERO_NOISE_REPEATS
        * (
            len(
                NOISE_LEVELS
            )
            - 1
        )
    )

    pair_datasets = (
        zero_noise_cases
        + nonzero_noise_cases
    )

    total_algorithms = 5

    total_registrations = (
        pair_datasets
        * total_algorithms
    )

    completed_count = len(
        completed
    )

    print()
    print("=" * 100)
    print(
        "REALISTIC FIVE-MODEL NOISE BENCHMARK"
    )
    print("=" * 100)

    print()
    print(
        f"Models:                  "
        f"{len(DENTAL_MODELS)}"
    )

    print(
        f"Jaw meshes:              "
        f"{number_of_jaws}"
    )

    print(
        f"Adjacent pairs per jaw:  "
        f"{len(ADJACENT_PAIRS)}"
    )

    print(
        f"Noise levels:             "
        f"{len(NOISE_LEVELS)}"
    )

    print(
        f"Repeats at sigma = 0:     "
        f"{ZERO_NOISE_REPEATS}"
    )

    print(
        f"Repeats at sigma > 0:     "
        f"{NONZERO_NOISE_REPEATS}"
    )

    print(
        f"Pair datasets:            "
        f"{pair_datasets}"
    )

    print(
        f"Total registrations:      "
        f"{total_registrations}"
    )

    print(
        f"Already completed:        "
        f"{completed_count}"
    )

    print()
    print(
        "Success criterion:"
    )

    print(
        f"  translation error <= "
        f"{SUCCESS_TRANSLATION_THRESHOLD:.3f} "
        f"model units"
    )

    print(
        f"  rotation error    <= "
        f"{SUCCESS_ROTATION_THRESHOLD_DEGREES:.3f} deg"
    )

    current_new_registration = 0

    # ========================================================
    # ALL MODELS / JAWS
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
            # Load all four ORIGINAL observed fragments
            # ------------------------------------------------

            raw_clouds = []

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

                raw_clouds.append(
                    cloud
                )

            # =================================================
            # NOISE LEVELS
            # =================================================

            for noise_index, sigma in enumerate(
                NOISE_LEVELS
            ):

                if sigma == 0.0:

                    repeat_count = (
                        ZERO_NOISE_REPEATS
                    )

                else:

                    repeat_count = (
                        NONZERO_NOISE_REPEATS
                    )

                for repeat_index in range(
                    repeat_count
                ):

                    repeat_number = (
                        repeat_index
                        + 1
                    )

                    # -----------------------------------------
                    # ONE noise seed for this entire jaw
                    # realization.
                    #
                    # All four fragments are perturbed from
                    # this RNG. All algorithms later see the
                    # exact same resulting clouds.
                    # -----------------------------------------

                    noise_seed = (
                        BASE_NOISE_SEED
                        + model_index
                        * 100_000
                        + jaw_index
                        * 10_000
                        + noise_index
                        * 100
                        + repeat_index
                    )

                    rng = (
                        np.random.default_rng(
                            noise_seed
                        )
                    )

                    print()
                    print("-" * 100)

                    print(
                        f"sigma={sigma:.3f} "
                        f"| repeat "
                        f"{repeat_number}/"
                        f"{repeat_count} "
                        f"| noise seed="
                        f"{noise_seed}"
                    )

                    print("-" * 100)

                    # -----------------------------------------
                    # Add noise ONCE to all four fragments
                    # -----------------------------------------

                    noisy_clouds = []

                    for cloud in (
                        raw_clouds
                    ):

                        noisy = (
                            add_gaussian_noise(
                                cloud=cloud,
                                sigma=sigma,
                                rng=rng,
                            )
                        )

                        noisy_clouds.append(
                            noisy
                        )

                    # -----------------------------------------
                    # Preprocess ONCE.
                    #
                    # Every algorithm receives identical
                    # processed geometry.
                    # -----------------------------------------

                    processed_clouds = []

                    for noisy_cloud in (
                        noisy_clouds
                    ):

                        processed = (
                            preprocessor.preprocess(
                                noisy_cloud
                            )
                        )

                        processed_clouds.append(
                            processed
                        )

                    # =========================================
                    # THREE ADJACENT PAIRS
                    # =========================================

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

                        # =====================================
                        # ALL FIVE ALGORITHMS
                        # =====================================

                        for (
                            algorithm_index,
                            algorithm,
                        ) in enumerate(
                            algorithms
                        ):

                            key = (
                                normalized_key(
                                    model=(
                                        model.model_id
                                    ),
                                    jaw=jaw,
                                    noise_std=sigma,
                                    repeat=(
                                        repeat_number
                                    ),
                                    pair=pair_name,
                                    algorithm=(
                                        algorithm.name
                                    ),
                                )
                            )

                            if key in completed:

                                continue

                            current_new_registration += 1

                            algorithm_seed = (
                                BASE_ALGORITHM_SEED
                                + model_index
                                * 1_000_000
                                + jaw_index
                                * 100_000
                                + noise_index
                                * 10_000
                                + repeat_index
                                * 1_000
                                + pair_index
                                * 100
                                + algorithm_index
                            )

                            o3d.utility.random.seed(
                                algorithm_seed
                            )

                            print(
                                f"  "
                                f"{model.model_id} "
                                f"{jaw} "
                                f"| sigma="
                                f"{sigma:.3f} "
                                f"| rep="
                                f"{repeat_number} "
                                f"| pair="
                                f"{pair_name} "
                                f"| "
                                f"{algorithm.name}"
                            )

                            # ---------------------------------
                            # Default failure values.
                            # ---------------------------------

                            fitness = float(
                                "nan"
                            )

                            rmse = None

                            translation_error = (
                                float(
                                    "nan"
                                )
                            )

                            rotation_error = (
                                float(
                                    "nan"
                                )
                            )

                            runtime = float(
                                "nan"
                            )

                            success = False

                            error_message = ""

                            try:

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
                                        ground_truth=(
                                            ground_truth
                                        ),
                                    )
                                )

                                fitness = float(
                                    result.fitness
                                )

                                rmse = (
                                    valid_rmse(
                                        fitness=(
                                            result.fitness
                                        ),
                                        rmse=(
                                            result.inlier_rmse
                                        ),
                                    )
                                )

                                translation_error = (
                                    float(
                                        result
                                        .translation_error
                                    )
                                )

                                rotation_error = (
                                    float(
                                        result
                                        .rotation_error_degrees
                                    )
                                )

                                runtime = float(
                                    result
                                    .runtime_seconds
                                )

                                success = (
                                    is_success(
                                        translation_error=(
                                            translation_error
                                        ),
                                        rotation_error_degrees=(
                                            rotation_error
                                        ),
                                    )
                                )

                            except Exception as error:

                                error_message = (
                                    f"{type(error).__name__}: "
                                    f"{error}"
                                )

                                print(
                                    "    ERROR: "
                                    f"{error_message}"
                                )

                            print(
                                f"    success="
                                f"{success} "
                                f"| T="
                                f"{translation_error:.6f} "
                                f"| R="
                                f"{rotation_error:.6f}"
                            )

                            rows.append(
                                {
                                    "Model":
                                        model.model_id,

                                    "Jaw":
                                        jaw,

                                    "Noise Std":
                                        float(
                                            sigma
                                        ),

                                    "Repeat":
                                        repeat_number,

                                    "Noise Seed":
                                        noise_seed,

                                    "Pair":
                                        pair_name,

                                    "Source Fragment":
                                        source_index,

                                    "Target Fragment":
                                        target_index,

                                    "Algorithm":
                                        algorithm.name,

                                    "Algorithm Seed":
                                        algorithm_seed,

                                    "Fitness":
                                        fitness,

                                    "Inlier RMSE":
                                        rmse,

                                    "Translation Error":
                                        translation_error,

                                    "Rotation Error [deg]":
                                        rotation_error,

                                    "Runtime [s]":
                                        runtime,

                                    "Success":
                                        success,

                                    "Error":
                                        error_message,
                                }
                            )

                            completed.add(
                                key
                            )

                        # -------------------------------------
                        # Checkpoint after every pair.
                        # -------------------------------------

                        save_checkpoint(
                            rows=rows,
                            raw_path=raw_path,
                        )

    # ========================================================
    # FINAL DATAFRAME
    # ========================================================

    raw = pd.DataFrame(
        rows
    )

    expected_rows = (
        total_registrations
    )

    print()
    print("=" * 100)
    print(
        "RUN COMPLETE"
    )
    print("=" * 100)

    print(
        f"Rows found:    "
        f"{len(raw)}"
    )

    print(
        f"Rows expected: "
        f"{expected_rows}"
    )

    if len(raw) != expected_rows:

        print()
        print(
            "WARNING: result count differs "
            "from expected count."
        )

        print(
            "Check the Error column and "
            "whether the run was interrupted."
        )

    # ========================================================
    # SUMMARIES
    # ========================================================

    overall = (
        aggregate_results(
            raw=raw,
            group_columns=[
                "Noise Std",
                "Algorithm",
            ],
        )
    )

    by_model = (
        aggregate_results(
            raw=raw,
            group_columns=[
                "Model",
                "Noise Std",
                "Algorithm",
            ],
        )
    )

    by_jaw = (
        aggregate_results(
            raw=raw,
            group_columns=[
                "Jaw",
                "Noise Std",
                "Algorithm",
            ],
        )
    )

    by_pair = (
        aggregate_results(
            raw=raw,
            group_columns=[
                "Pair",
                "Noise Std",
                "Algorithm",
            ],
        )
    )

    # ========================================================
    # SAVE FINAL TABLES
    # ========================================================

    overall_path = (
        output_directory
        / "noise_overall_summary.csv"
    )

    model_path = (
        output_directory
        / "noise_by_model.csv"
    )

    jaw_path = (
        output_directory
        / "noise_by_jaw.csv"
    )

    pair_path = (
        output_directory
        / "noise_by_pair.csv"
    )

    raw.to_csv(
        raw_path,
        index=False,
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    by_model.to_csv(
        model_path,
        index=False,
    )

    by_jaw.to_csv(
        jaw_path,
        index=False,
    )

    by_pair.to_csv(
        pair_path,
        index=False,
    )

    # ========================================================
    # CONSOLE RESULT TABLE
    # ========================================================

    print()
    print("=" * 100)
    print(
        "SUCCESS RATE BY NOISE LEVEL"
    )
    print("=" * 100)
    print()

    success_table = (
        overall.pivot(
            index="Noise Std",
            columns="Algorithm",
            values="Success_Rate",
        )
    )

    print(
        success_table.to_string(
            float_format=lambda value: (
                f"{value:.1f}"
            )
        )
    )

    print()
    print("=" * 100)
    print(
        "FPFH + RANSAC + ICP"
    )
    print("=" * 100)
    print()

    ransac = (
        overall[
            overall[
                "Algorithm"
            ]
            == "FPFH + RANSAC + ICP"
        ]
        .copy()
    )

    columns = [
        "Noise Std",
        "Cases",
        "Successes",
        "Success_Rate",
        "Wilson_95_Lower",
        "Wilson_95_Upper",
        "Translation_Error_Mean",
        "Translation_Error_Median",
        "Rotation_Error_Mean",
        "Rotation_Error_Median",
        "Runtime_Mean",
    ]

    print(
        ransac[
            columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("=" * 100)
    print(
        "OUTPUT FILES"
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