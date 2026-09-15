from __future__ import annotations

from pathlib import Path
import copy
import time

import numpy as np
import open3d as o3d
import pandas as pd

from realistic_registration_benchmark import (
    load_cloud,
    preprocess,
    point_to_point_icp,
    point_to_plane_icp,
    generalized_icp,
    ransac_fpfh_icp,
    fgr_fpfh_icp,
    evaluate,
)


BASE_DIR = Path("data/realistic_jaw")
OUTPUT_DIR = Path("output/realistic_noise_benchmark")

JAW_DIRS = {
    "upper": BASE_DIR / "upper_test",
    "lower": BASE_DIR / "lower_test",
}

PAIRS = [
    (0, 1),
    (1, 2),
    (2, 3),
]

NOISE_LEVELS_MM = [
    0.0,
    0.01,
    0.025,
    0.05,
    0.10,
    0.20,
]

REPEATS = 3
BASE_SEED = 42

# Nur für die Erfolgsrate dieses realistischen Experiments.
# Die Rohfehler werden unabhängig davon immer gespeichert.
SUCCESS_TRANSLATION_THRESHOLD_MM = 1.0
SUCCESS_ROTATION_THRESHOLD_DEG = 1.0


def add_gaussian_noise(
    cloud: o3d.geometry.PointCloud,
    std_mm: float,
    seed: int,
) -> o3d.geometry.PointCloud:
    """
    Fügt isotropes gaußsches Positionsrauschen zu einer Punktwolke hinzu.

    std_mm ist die Standardabweichung pro Koordinatenachse.
    """

    noisy = copy.deepcopy(cloud)

    if std_mm <= 0.0:
        return noisy

    rng = np.random.default_rng(seed)

    points = np.asarray(noisy.points).copy()

    noise = rng.normal(
        loc=0.0,
        scale=std_mm,
        size=points.shape,
    )

    noisy.points = o3d.utility.Vector3dVector(
        points + noise
    )

    return noisy


def make_seed(
    jaw_index: int,
    repeat: int,
    noise_index: int,
    fragment_index: int,
) -> int:
    """
    Deterministische Seeds für reproduzierbare Noise-Realisierungen.
    """

    return (
        BASE_SEED
        + jaw_index * 100_000
        + repeat * 10_000
        + noise_index * 100
        + fragment_index
    )


def run_algorithm(
    algorithm_name: str,
    source,
    target,
    source_fpfh,
    target_fpfh,
):
    if algorithm_name == "Point-to-Point ICP":
        return point_to_point_icp(
            source,
            target,
        )

    if algorithm_name == "Point-to-Plane ICP":
        return point_to_plane_icp(
            source,
            target,
        )

    if algorithm_name == "Generalized ICP":
        return generalized_icp(
            source,
            target,
        )

    if algorithm_name == "FPFH + RANSAC + ICP":
        return ransac_fpfh_icp(
            source,
            target,
            source_fpfh,
            target_fpfh,
        )

    if algorithm_name == "FPFH + FGR + ICP":
        return fgr_fpfh_icp(
            source,
            target,
            source_fpfh,
            target_fpfh,
        )

    raise ValueError(
        f"Unbekannter Algorithmus: {algorithm_name}"
    )


ALGORITHMS = [
    "Point-to-Point ICP",
    "Point-to-Plane ICP",
    "Generalized ICP",
    "FPFH + RANSAC + ICP",
    "FPFH + FGR + ICP",
]


def benchmark_jaw(
    jaw_name: str,
    jaw_dir: Path,
    jaw_index: int,
) -> list[dict]:

    print()
    print("=" * 90)
    print(f"{jaw_name.upper()} JAW – NOISE BENCHMARK")
    print("=" * 90)

    acquisition_transforms = np.load(
        jaw_dir / "acquisition_transforms.npy"
    )

    original_clouds = {}

    for index in range(4):
        path = (
            jaw_dir
            / f"fragment_{index:02d}.ply"
        )

        original_clouds[index] = load_cloud(path)

        print(
            f"Fragment {index}: "
            f"{len(original_clouds[index].points):,} Punkte"
        )

    rows = []

    for noise_index, noise_std in enumerate(
        NOISE_LEVELS_MM
    ):

        print()
        print("#" * 90)
        print(
            f"NOISE STD = {noise_std:.3f} mm"
        )
        print("#" * 90)

        for repeat in range(REPEATS):

            print()
            print(
                f"Repeat {repeat + 1}/{REPEATS}"
            )

            prepared = {}

            # Jede Punktwolke wird pro Noise-Stufe und Wiederholung
            # genau einmal verrauscht und vorbereitet.
            #
            # Dadurch erhalten alle Algorithmen dieselben Daten.
            for fragment_index in range(4):

                seed = make_seed(
                    jaw_index=jaw_index,
                    repeat=repeat,
                    noise_index=noise_index,
                    fragment_index=fragment_index,
                )

                noisy_cloud = add_gaussian_noise(
                    cloud=original_clouds[
                        fragment_index
                    ],
                    std_mm=noise_std,
                    seed=seed,
                )

                downsampled, fpfh = preprocess(
                    noisy_cloud
                )

                prepared[fragment_index] = {
                    "cloud": downsampled,
                    "fpfh": fpfh,
                }

            for source_index, target_index in PAIRS:

                source = prepared[source_index][
                    "cloud"
                ]
                target = prepared[target_index][
                    "cloud"
                ]

                source_fpfh = prepared[
                    source_index
                ]["fpfh"]

                target_fpfh = prepared[
                    target_index
                ]["fpfh"]

                # Ground Truth bleibt trotz Noise unverändert:
                #
                # x_i = A_i * x_world
                #
                # source i -> target j:
                #
                # T_gt = A_j @ inv(A_i)

                ground_truth = (
                    acquisition_transforms[
                        target_index
                    ]
                    @ np.linalg.inv(
                        acquisition_transforms[
                            source_index
                        ]
                    )
                )

                print()
                print(
                    f"{source_index} -> "
                    f"{target_index}"
                )

                for algorithm_name in ALGORITHMS:

                    print(
                        f"  {algorithm_name:<25}",
                        end="",
                        flush=True,
                    )

                    start = time.perf_counter()

                    result = run_algorithm(
                        algorithm_name=algorithm_name,
                        source=source,
                        target=target,
                        source_fpfh=source_fpfh,
                        target_fpfh=target_fpfh,
                    )

                    runtime = (
                        time.perf_counter()
                        - start
                    )

                    metrics = evaluate(
                        source=source,
                        target=target,
                        transformation=(
                            result.transformation
                        ),
                        ground_truth=ground_truth,
                    )

                    success = (
                        metrics[
                            "translation_error"
                        ]
                        <= SUCCESS_TRANSLATION_THRESHOLD_MM
                        and
                        metrics[
                            "rotation_error_deg"
                        ]
                        <= SUCCESS_ROTATION_THRESHOLD_DEG
                    )

                    row = {
                        "jaw": jaw_name,
                        "noise_std_mm": noise_std,
                        "repeat": repeat,
                        "source": source_index,
                        "target": target_index,
                        "algorithm": algorithm_name,
                        "fitness": metrics[
                            "fitness"
                        ],
                        "inlier_rmse": metrics[
                            "inlier_rmse"
                        ],
                        "translation_error_mm": (
                            metrics[
                                "translation_error"
                            ]
                        ),
                        "rotation_error_deg": (
                            metrics[
                                "rotation_error_deg"
                            ]
                        ),
                        "runtime_s": runtime,
                        "success": success,
                    }

                    rows.append(row)

                    print(
                        f" fitness={row['fitness']:.4f}"
                        f" rmse={row['inlier_rmse']:.4f}"
                        f" t={row['translation_error_mm']:.4f}"
                        f" r={row['rotation_error_deg']:.4f}°"
                        f" success={success}"
                    )

    return rows


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []

    for jaw_index, (
        jaw_name,
        jaw_dir,
    ) in enumerate(
        JAW_DIRS.items()
    ):
        all_rows.extend(
            benchmark_jaw(
                jaw_name=jaw_name,
                jaw_dir=jaw_dir,
                jaw_index=jaw_index,
            )
        )

    dataframe = pd.DataFrame(all_rows)

    raw_path = (
        OUTPUT_DIR
        / "realistic_noise_results.csv"
    )

    dataframe.to_csv(
        raw_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Zusammenfassung pro Noise-Stufe und Algorithmus
    # ---------------------------------------------------------

    summary = (
        dataframe
        .groupby(
            [
                "noise_std_mm",
                "algorithm",
            ]
        )
        .agg(
            mean_fitness=(
                "fitness",
                "mean",
            ),
            mean_inlier_rmse=(
                "inlier_rmse",
                "mean",
            ),
            mean_translation_error_mm=(
                "translation_error_mm",
                "mean",
            ),
            median_translation_error_mm=(
                "translation_error_mm",
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
            success_rate=(
                "success",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["success_rate_percent"] = (
        summary["success_rate"] * 100.0
    )

    summary_path = (
        OUTPUT_DIR
        / "realistic_noise_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Gesamtauswertung über alle Noise-Stufen
    # ---------------------------------------------------------

    overall = (
        dataframe
        .groupby("algorithm")
        .agg(
            mean_translation_error_mm=(
                "translation_error_mm",
                "mean",
            ),
            median_translation_error_mm=(
                "translation_error_mm",
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
            success_rate=(
                "success",
                "mean",
            ),
        )
        .reset_index()
    )

    overall[
        "success_rate_percent"
    ] = (
        overall["success_rate"]
        * 100.0
    )

    overall_path = (
        OUTPUT_DIR
        / "realistic_noise_overall.csv"
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    print()
    print("=" * 90)
    print("ERGEBNIS PRO NOISE-STUFE")
    print("=" * 90)

    display_columns = [
        "noise_std_mm",
        "algorithm",
        "mean_translation_error_mm",
        "mean_rotation_error_deg",
        "success_rate_percent",
        "mean_runtime_s",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("=" * 90)
    print("GESAMTERGEBNIS ÜBER ALLE NOISE-STUFEN")
    print("=" * 90)

    print(
        overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print(f"Rohdaten:       {raw_path}")
    print(f"Noise Summary:  {summary_path}")
    print(f"Overall:        {overall_path}")


if __name__ == "__main__":
    main()