from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "assets"
    / "marvin"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SYNTHETIC_SUMMARY = (
    PROJECT_ROOT
    / "output"
    / "controlled_experiments"
    / "experiments_summary.csv"
)

SYNTHETIC_RAW = (
    PROJECT_ROOT
    / "output"
    / "controlled_experiments"
    / "experiments_raw.csv"
)

REALISTIC_BPA_RESULTS = (
    PROJECT_ROOT
    / "output"
    / "realistic_bpa_parameter_study"
    / "realistic_bpa_parameter_results.csv"
)

SIMPLIFICATION_RESULTS = (
    PROJECT_ROOT
    / "output"
    / "mesh"
    / "simplification_experiments"
    / "simplification_results.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

ALGORITHM_ORDER = [
    "Point-to-Point ICP",
    "Point-to-Plane ICP",
    "Generalized ICP",
    "FPFH + FGR + ICP",
    "FPFH + RANSAC + ICP",
]


# ============================================================
# HELPERS
# ============================================================

def require_file(
    path: Path,
) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def save_figure(
    filename: str,
) -> None:

    output_path = (
        OUTPUT_DIR
        / filename
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def normalize_algorithm_name(
    name: str,
) -> str:

    mapping = {
        "Point-to-Point ICP":
            "Point-to-Point ICP",

        "Point-to-Plane ICP":
            "Point-to-Plane ICP",

        "Generalized ICP":
            "Generalized ICP",

        "FPFH + FGR + ICP":
            "FPFH + FGR + ICP",

        "FPFH + RANSAC + ICP":
            "FPFH + RANSAC + ICP",
    }

    return mapping.get(
        name,
        name,
    )


# ============================================================
# 1. SYNTHETIC SUCCESS-RATE PLOTS
# ============================================================

def plot_synthetic_success_rates() -> None:

    require_file(
        SYNTHETIC_SUMMARY
    )

    df = pd.read_csv(
        SYNTHETIC_SUMMARY
    )

    print()
    print("=" * 80)
    print(
        "SYNTHETIC SUCCESS-RATE PLOTS"
    )
    print("=" * 80)

    series_configs = {
        "noise": {
            "filename":
                "noise_success_rate.png",

            "xlabel":
                "Gaussian noise standard deviation",

            "title":
                None,
        },

        "overlap": {
            "filename":
                "overlap_success_rate.png",

            "xlabel":
                "Overlap fraction",

            "title":
                None,
        },

        "rotation": {
            "filename":
                "rotation_success_rate.png",

            "xlabel":
                "Initial rotation [deg]",

            "title":
                None,
        },

        "translation": {
            "filename":
                "translation_success_rate.png",

            "xlabel":
                "Initial translation [model units]",

            "title":
                None,
        },
    }

    for series_name, config in (
        series_configs.items()
    ):

        subset = df[
            df["Series"]
            .str.lower()
            == series_name
        ].copy()

        if subset.empty:
            print(
                f"WARNING: no rows for "
                f"series '{series_name}'."
            )
            continue

        subset[
            "Algorithm"
        ] = subset[
            "Algorithm"
        ].map(
            normalize_algorithm_name
        )

        plt.figure(
            figsize=(7.8, 4.8)
        )

        for algorithm in (
            ALGORITHM_ORDER
        ):

            algorithm_data = subset[
                subset[
                    "Algorithm"
                ]
                == algorithm
            ].sort_values(
                "Variable Value"
            )

            if algorithm_data.empty:
                continue

            plt.plot(
                algorithm_data[
                    "Variable Value"
                ],
                algorithm_data[
                    "Success_Rate"
                ],
                marker="o",
                linewidth=1.8,
                label=algorithm,
            )

        plt.xlabel(
            config[
                "xlabel"
            ]
        )

        plt.ylabel(
            "Success rate [%]"
        )

        plt.ylim(
            -3,
            103,
        )

        plt.grid(
            True,
            alpha=0.25,
        )

        plt.legend(
            fontsize=8,
            ncol=2,
        )

        save_figure(
            config[
                "filename"
            ]
        )


# ============================================================
# 2. CHECK SYNTHETIC REPEATS
# ============================================================

def print_synthetic_repeat_check() -> None:

    require_file(
        SYNTHETIC_RAW
    )

    df = pd.read_csv(
        SYNTHETIC_RAW
    )

    grouped = (
        df.groupby(
            [
                "Series",
                "Variable Value",
                "Algorithm",
            ]
        )
        .size()
    )

    print()
    print("=" * 80)
    print(
        "SYNTHETIC REPEAT CHECK"
    )
    print("=" * 80)

    print(
        grouped.value_counts()
    )

    unique_counts = sorted(
        grouped.unique()
    )

    print()
    print(
        "Unique repeat counts:"
    )

    print(
        unique_counts
    )

    if unique_counts == [10]:

        print()
        print(
            "OK: every synthetic condition "
            "contains exactly 10 repetitions."
        )

    else:

        print()
        print(
            "WARNING: repeat count is "
            "not constant."
        )


# ============================================================
# 3. REALISTIC NOISE PLOT
# ============================================================

def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.96,
) -> tuple[
    float,
    float,
]:

    if trials == 0:
        return (
            float("nan"),
            float("nan"),
        )

    p = (
        successes
        / trials
    )

    denominator = (
        1.0
        + z ** 2
        / trials
    )

    center = (
        p
        + z ** 2
        / (
            2.0
            * trials
        )
    ) / denominator

    margin = (
        z
        / denominator
        * math.sqrt(
            p
            * (
                1.0
                - p
            )
            / trials
            + z ** 2
            / (
                4.0
                * trials ** 2
            )
        )
    )

    return (
        center - margin,
        center + margin,
    )


def plot_realistic_noise_success() -> None:

    """
    Values are the validated aggregate results from
    the realistic upper- and lower-jaw noise experiment.

    There are:
        2 jaws
        x 3 adjacent fragment pairs
        x 3 repetitions
        = 18 trials per noise level.

    At sigma = 0 the repeated trials are deterministic
    duplicates, because no positional noise is added.
    """

    noise_levels = np.array(
        [
            0.000,
            0.010,
            0.025,
            0.050,
            0.100,
            0.200,
        ]
    )

    success_percent = {
        "Point-to-Point ICP": [
            33.3333,
            33.3333,
            33.3333,
            33.3333,
            33.3333,
            33.3333,
        ],

        "Point-to-Plane ICP": [
            66.6667,
            66.6667,
            66.6667,
            66.6667,
            66.6667,
            55.5556,
        ],

        "Generalized ICP": [
            66.6667,
            66.6667,
            66.6667,
            66.6667,
            66.6667,
            66.6667,
        ],

        "FPFH + FGR + ICP": [
            66.6667,
            77.7778,
            77.7778,
            66.6667,
            66.6667,
            61.1111,
        ],

        "FPFH + RANSAC + ICP": [
            100.0000,
            100.0000,
            100.0000,
            100.0000,
            100.0000,
            72.2222,
        ],
    }

    trials = 18

    plt.figure(
        figsize=(8.2, 5.0)
    )

    for algorithm in (
        ALGORITHM_ORDER
    ):

        rates = np.asarray(
            success_percent[
                algorithm
            ]
        )

        plt.plot(
            noise_levels,
            rates,
            marker="o",
            linewidth=1.8,
            label=algorithm,
        )

        # Wilson confidence interval only
        # for final selected algorithm.
        if (
            algorithm
            == "FPFH + RANSAC + ICP"
        ):

            lower = []
            upper = []

            for rate in rates:

                successes = int(
                    round(
                        rate
                        / 100.0
                        * trials
                    )
                )

                low, high = (
                    wilson_interval(
                        successes,
                        trials,
                    )
                )

                lower.append(
                    low
                    * 100.0
                )

                upper.append(
                    high
                    * 100.0
                )

            lower = np.asarray(
                lower
            )

            upper = np.asarray(
                upper
            )

            plt.fill_between(
                noise_levels,
                lower,
                upper,
                alpha=0.12,
                label=(
                    "RANSAC + ICP "
                    "95% Wilson interval"
                ),
            )

    plt.xlabel(
        "Gaussian noise standard deviation "
        "[model units]"
    )

    plt.ylabel(
        "Success rate [%]"
    )

    plt.ylim(
        0,
        105,
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend(
        fontsize=8,
        ncol=2,
    )

    save_figure(
        "realistic_noise_success_rate.png"
    )


# ============================================================
# 4. REALISTIC MULTIWAY ERROR PLOTS
# ============================================================

def plot_realistic_multiway_errors() -> None:

    fragments = np.array(
        [
            0,
            1,
            2,
            3,
        ]
    )

    upper_translation = np.array(
        [
            0.000000,
            0.002601,
            0.004223,
            0.005125,
        ]
    )

    lower_translation = np.array(
        [
            0.000000,
            0.001216,
            0.002137,
            0.005928,
        ]
    )

    upper_rotation = np.array(
        [
            0.000000,
            0.007799,
            0.013474,
            0.016531,
        ]
    )

    lower_rotation = np.array(
        [
            0.000000,
            0.008341,
            0.035999,
            0.037936,
        ]
    )

    # --------------------------------------------------------
    # Translation
    # --------------------------------------------------------

    plt.figure(
        figsize=(7.5, 4.5)
    )

    plt.plot(
        fragments,
        upper_translation,
        marker="o",
        linewidth=1.8,
        label="Upper jaw",
    )

    plt.plot(
        fragments,
        lower_translation,
        marker="o",
        linewidth=1.8,
        label="Lower jaw",
    )

    plt.xlabel(
        "Fragment index"
    )

    plt.ylabel(
        "Global translation error "
        "[model units]"
    )

    plt.xticks(
        fragments
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    save_figure(
        "realistic_multiway_translation_error.png"
    )

    # --------------------------------------------------------
    # Rotation
    # --------------------------------------------------------

    plt.figure(
        figsize=(7.5, 4.5)
    )

    plt.plot(
        fragments,
        upper_rotation,
        marker="o",
        linewidth=1.8,
        label="Upper jaw",
    )

    plt.plot(
        fragments,
        lower_rotation,
        marker="o",
        linewidth=1.8,
        label="Lower jaw",
    )

    plt.xlabel(
        "Fragment index"
    )

    plt.ylabel(
        "Global rotation error [deg]"
    )

    plt.xticks(
        fragments
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    save_figure(
        "realistic_multiway_rotation_error.png"
    )


# ============================================================
# 5. REALISTIC BPA PARAMETER STUDY
# ============================================================

def plot_realistic_bpa_study() -> None:

    require_file(
        REALISTIC_BPA_RESULTS
    )

    df = pd.read_csv(
        REALISTIC_BPA_RESULTS
    )

    configuration_order = [
        "A_small",
        "B_baseline",
        "C_medium",
        "D_large",
    ]

    configuration_labels = [
        "A\n1.0 / 1.5 / 2.0",
        "B\n1.5 / 2.0 / 3.0",
        "C\n2.0 / 3.0 / 4.0",
        "D\n2.5 / 4.0 / 6.0",
    ]

    x = np.arange(
        len(
            configuration_order
        )
    )

    # --------------------------------------------------------
    # Boundary edges
    # --------------------------------------------------------

    plt.figure(
        figsize=(7.5, 4.6)
    )

    for jaw in [
        "upper",
        "lower",
    ]:

        subset = (
            df[
                df["jaw"]
                == jaw
            ]
            .set_index(
                "configuration"
            )
            .reindex(
                configuration_order
            )
        )

        plt.plot(
            x,
            subset[
                "boundary_edges"
            ],
            marker="o",
            linewidth=1.8,
            label=(
                "Upper jaw"
                if jaw == "upper"
                else "Lower jaw"
            ),
        )

    plt.xticks(
        x,
        configuration_labels,
    )

    plt.xlabel(
        "Ball Pivoting radius-factor "
        "configuration"
    )

    plt.ylabel(
        "Boundary edges"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    save_figure(
        "realistic_bpa_boundary_edges.png"
    )

    # --------------------------------------------------------
    # Symmetric RMSE
    # --------------------------------------------------------

    plt.figure(
        figsize=(7.5, 4.6)
    )

    for jaw in [
        "upper",
        "lower",
    ]:

        subset = (
            df[
                df["jaw"]
                == jaw
            ]
            .set_index(
                "configuration"
            )
            .reindex(
                configuration_order
            )
        )

        plt.plot(
            x,
            subset[
                "symmetric_rmse"
            ],
            marker="o",
            linewidth=1.8,
            label=(
                "Upper jaw"
                if jaw == "upper"
                else "Lower jaw"
            ),
        )

    plt.xticks(
        x,
        configuration_labels,
    )

    plt.xlabel(
        "Ball Pivoting radius-factor "
        "configuration"
    )

    plt.ylabel(
        "Symmetric RMSE "
        "[model units]"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    save_figure(
        "realistic_bpa_symmetric_rmse.png"
    )


# ============================================================
# 6. SYNTHETIC MESH SIMPLIFICATION
# ============================================================

def plot_mesh_simplification() -> None:

    if not SIMPLIFICATION_RESULTS.exists():

        print()
        print(
            "WARNING: simplification CSV "
            "not found:"
        )

        print(
            SIMPLIFICATION_RESULTS
        )

        print(
            "Skipping simplification plot."
        )

        return

    df = pd.read_csv(
        SIMPLIFICATION_RESULTS
    )

    print()
    print(
        "Simplification columns:"
    )

    print(
        list(
            df.columns
        )
    )

    # Try several possible column names.
    triangle_candidates = [
        "triangles",
        "triangle_count",
        "actual_triangles",
        "Triangles",
    ]

    rmse_candidates = [
        "symmetric_rmse",
        "symmetric_RMSE",
        "Symmetric_RMSE",
        "Symmetric RMSE",
    ]

    triangle_column = next(
        (
            column
            for column
            in triangle_candidates
            if column
            in df.columns
        ),
        None,
    )

    rmse_column = next(
        (
            column
            for column
            in rmse_candidates
            if column
            in df.columns
        ),
        None,
    )

    if (
        triangle_column is None
        or rmse_column is None
    ):

        print()
        print(
            "WARNING: could not identify "
            "triangle/RMSE columns."
        )

        print(
            "Skipping simplification plot."
        )

        return

    df = df.sort_values(
        triangle_column
    )

    plt.figure(
        figsize=(7.4, 4.5)
    )

    plt.plot(
        df[
            triangle_column
        ],
        df[
            rmse_column
        ],
        marker="o",
        linewidth=1.8,
    )

    plt.xlabel(
        "Triangle count"
    )

    plt.ylabel(
        "Symmetric RMSE "
        "[model units]"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    save_figure(
        "mesh_simplification_study.png"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 80)
    print(
        "FINAL REPORT PLOT GENERATOR"
    )
    print("=" * 80)

    print_synthetic_repeat_check()

    plot_synthetic_success_rates()

    plot_realistic_noise_success()

    plot_realistic_multiway_errors()

    plot_realistic_bpa_study()

    plot_mesh_simplification()

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print()
    print(
        "Final report plots saved to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()