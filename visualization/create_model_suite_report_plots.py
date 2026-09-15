from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "assets"
    / "marvin"
    / "model_suite"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


PAIRWISE_DIRECTORY = (
    PROJECT_ROOT
    / "output"
    / "realistic_model_suite_pairwise"
)

NOISE_DIRECTORY = (
    PROJECT_ROOT
    / "output"
    / "realistic_model_suite_noise"
)

MULTIWAY_DIRECTORY = (
    PROJECT_ROOT
    / "output"
    / "realistic_model_suite_multiway_ransac"
)

BPA_DIRECTORY = (
    PROJECT_ROOT
    / "output"
    / "realistic_model_suite_bpa_parameter_study"
)

MESH_DIRECTORY = (
    PROJECT_ROOT
    / "output"
    / "realistic_model_suite_mesh_validation"
)


# ============================================================
# DISPLAY NAMES
# ============================================================

ALGORITHM_LABELS = {
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


BPA_LABELS = {
    "A_small":
        "A: (1.0, 1.5, 2.0)",

    "B_baseline":
        "B: (1.5, 2.0, 3.0)",

    "C_medium":
        "C: (2.0, 3.0, 4.0)",

    "D_large":
        "D: (2.5, 4.0, 6.0)",
}


MODEL_LABELS = {
    "model_01": "Model 1",
    "model_02": "Model 2",
    "model_03": "Model 3",
    "model_04": "Model 4",
    "model_05": "Model 5",
}


# ============================================================
# STYLE HELPERS
# ============================================================

def finish_figure(
    path: Path,
) -> None:

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


def add_bar_labels(
    axis,
    bars,
    decimals: int = 1,
    suffix: str = "",
) -> None:

    for bar in bars:

        value = (
            bar.get_height()
        )

        axis.text(
            bar.get_x()
            + bar.get_width()
            / 2.0,

            value,

            f"{value:.{decimals}f}"
            f"{suffix}",

            ha="center",
            va="bottom",
            fontsize=8,
        )


# ============================================================
# 1. PAIRWISE SUCCESS RATE
# ============================================================

def plot_pairwise_success_rate() -> None:

    path = (
        PAIRWISE_DIRECTORY
        / "pairwise_overall_summary.csv"
    )

    data = pd.read_csv(
        path
    )

    desired_order = [
        "Point-to-Point ICP",
        "Point-to-Plane ICP",
        "Generalized ICP",
        "FPFH + FGR + ICP",
        "FPFH + RANSAC + ICP",
    ]

    data = (
        data
        .set_index(
            "Algorithm"
        )
        .loc[
            desired_order
        ]
        .reset_index()
    )

    labels = [
        ALGORITHM_LABELS[
            algorithm
        ]
        for algorithm
        in data["Algorithm"]
    ]

    values = (
        data[
            "Success_Rate"
        ]
        .to_numpy()
    )

    figure, axis = (
        plt.subplots(
            figsize=(9.2, 5.0)
        )
    )

    bars = axis.bar(
        labels,
        values,
    )

    add_bar_labels(
        axis=axis,
        bars=bars,
        decimals=1,
        suffix="%",
    )

    axis.set_ylabel(
        "Success rate [%]"
    )

    axis.set_xlabel(
        "Registration method"
    )

    axis.set_title(
        "Pairwise Registration Success Across "
        "30 Adjacent Fragment Pairs"
    )

    axis.set_ylim(
        0,
        108,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.tick_params(
        axis="x",
        rotation=18,
    )

    finish_figure(
        OUTPUT_DIRECTORY
        / "model_suite_pairwise_success_rate.png"
    )


# ============================================================
# 2. SUCCESS RATE BY MODEL
# ============================================================

def plot_pairwise_success_by_model() -> None:

    path = (
        PAIRWISE_DIRECTORY
        / "pairwise_by_model.csv"
    )

    data = pd.read_csv(
        path
    )

    pivot = (
        data.pivot(
            index="Model",
            columns="Algorithm",
            values="Success_Rate",
        )
    )

    desired_models = [
        "model_01",
        "model_02",
        "model_03",
        "model_04",
        "model_05",
    ]

    desired_algorithms = [
        "Point-to-Point ICP",
        "Point-to-Plane ICP",
        "Generalized ICP",
        "FPFH + FGR + ICP",
        "FPFH + RANSAC + ICP",
    ]

    pivot = (
        pivot
        .loc[
            desired_models,
            desired_algorithms,
        ]
    )

    figure, axis = (
        plt.subplots(
            figsize=(10.0, 5.5)
        )
    )

    x = np.arange(
        len(
            desired_models
        )
    )

    number_of_algorithms = (
        len(
            desired_algorithms
        )
    )

    total_width = 0.82

    bar_width = (
        total_width
        / number_of_algorithms
    )

    for index, algorithm in enumerate(
        desired_algorithms
    ):

        offset = (
            index
            - (
                number_of_algorithms
                - 1
            )
            / 2.0
        ) * bar_width

        axis.bar(
            x + offset,
            pivot[
                algorithm
            ].to_numpy(),
            width=bar_width,
            label=(
                ALGORITHM_LABELS[
                    algorithm
                ]
            ),
        )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        [
            MODEL_LABELS[
                model
            ]
            for model
            in desired_models
        ]
    )

    axis.set_ylim(
        0,
        108,
    )

    axis.set_ylabel(
        "Success rate [%]"
    )

    axis.set_xlabel(
        "Dental model set"
    )

    axis.set_title(
        "Pairwise Registration Success by Dental Model"
    )

    axis.legend(
        fontsize=8,
        ncol=2,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    finish_figure(
        OUTPUT_DIRECTORY
        / "model_suite_pairwise_success_by_model.png"
    )


# ============================================================
# 3. NOISE ROBUSTNESS
# ============================================================

def plot_noise_robustness() -> None:

    path = (
        NOISE_DIRECTORY
        / "noise_overall_summary.csv"
    )

    data = pd.read_csv(
        path
    )

    desired_algorithms = [
        "Point-to-Point ICP",
        "Point-to-Plane ICP",
        "Generalized ICP",
        "FPFH + FGR + ICP",
        "FPFH + RANSAC + ICP",
    ]

    figure, axis = (
        plt.subplots(
            figsize=(9.2, 5.4)
        )
    )

    for algorithm in (
        desired_algorithms
    ):

        subset = (
            data[
                data[
                    "Algorithm"
                ]
                == algorithm
            ]
            .sort_values(
                "Noise Std"
            )
        )

        axis.plot(
            subset[
                "Noise Std"
            ],
            subset[
                "Success_Rate"
            ],
            marker="o",
            linewidth=1.8,
            label=(
                ALGORITHM_LABELS[
                    algorithm
                ]
            ),
        )

    # --------------------------------------------------------
    # Confidence band for selected final algorithm.
    #
    # sigma = 0 contains 30 distinct deterministic cases.
    # Non-zero levels contain 90 noisy trials.
    # --------------------------------------------------------

    ransac = (
        data[
            data[
                "Algorithm"
            ]
            == "FPFH + RANSAC + ICP"
        ]
        .sort_values(
            "Noise Std"
        )
    )

    nonzero = (
        ransac[
            ransac[
                "Noise Std"
            ]
            > 0.0
        ]
    )

    axis.fill_between(
        nonzero[
            "Noise Std"
        ],
        nonzero[
            "Wilson_95_Lower"
        ],
        nonzero[
            "Wilson_95_Upper"
        ],
        alpha=0.15,
        label=(
            "RANSAC 95% Wilson interval"
        ),
    )

    axis.set_ylim(
        0,
        105,
    )

    axis.set_xlabel(
        "Gaussian noise standard deviation "
        "[model units]"
    )

    axis.set_ylabel(
        "Success rate [%]"
    )

    axis.set_title(
        "Registration Robustness to Gaussian Positional Noise"
    )

    axis.grid(
        alpha=0.25,
    )

    axis.legend(
        fontsize=8,
        ncol=2,
    )

    finish_figure(
        OUTPUT_DIRECTORY
        / "model_suite_noise_success_rate.png"
    )


# ============================================================
# 4. MULTIWAY TRANSLATION ERROR
# ============================================================

def plot_multiway_translation_error() -> None:

    path = (
        MULTIWAY_DIRECTORY
        / "all_pose_errors.csv"
    )

    data = pd.read_csv(
        path
    )

    data = (
        data[
            data[
                "Fragment"
            ]
            > 0
        ]
        .copy()
    )

    grouped = (
        data.groupby(
            "Fragment"
        )[
            "Translation Error"
        ]
        .agg(
            [
                "mean",
                "median",
                "min",
                "max",
            ]
        )
        .reset_index()
    )

    figure, axis = (
        plt.subplots(
            figsize=(7.4, 4.8)
        )
    )

    axis.plot(
        grouped[
            "Fragment"
        ],
        grouped[
            "mean"
        ],
        marker="o",
        linewidth=2.0,
        label="Mean",
    )

    axis.plot(
        grouped[
            "Fragment"
        ],
        grouped[
            "median"
        ],
        marker="s",
        linewidth=1.6,
        label="Median",
    )

    axis.fill_between(
        grouped[
            "Fragment"
        ],
        grouped[
            "min"
        ],
        grouped[
            "max"
        ],
        alpha=0.15,
        label="Range across 10 jaw meshes",
    )

    axis.set_xticks(
        [
            1,
            2,
            3,
        ]
    )

    axis.set_xlabel(
        "Fragment in sequential chain"
    )

    axis.set_ylabel(
        "Translation error [model units]"
    )

    axis.set_title(
        "Accumulated Multiway Translation Error"
    )

    axis.grid(
        alpha=0.25,
    )

    axis.legend()

    finish_figure(
        OUTPUT_DIRECTORY
        / "model_suite_multiway_translation_error.png"
    )


# ============================================================
# 5. MULTIWAY ROTATION ERROR
# ============================================================

def plot_multiway_rotation_error() -> None:

    path = (
        MULTIWAY_DIRECTORY
        / "all_pose_errors.csv"
    )

    data = pd.read_csv(
        path
    )

    data = (
        data[
            data[
                "Fragment"
            ]
            > 0
        ]
        .copy()
    )

    grouped = (
        data.groupby(
            "Fragment"
        )[
            "Rotation Error [deg]"
        ]
        .agg(
            [
                "mean",
                "median",
                "min",
                "max",
            ]
        )
        .reset_index()
    )

    figure, axis = (
        plt.subplots(
            figsize=(7.4, 4.8)
        )
    )

    axis.plot(
        grouped[
            "Fragment"
        ],
        grouped[
            "mean"
        ],
        marker="o",
        linewidth=2.0,
        label="Mean",
    )

    axis.plot(
        grouped[
            "Fragment"
        ],
        grouped[
            "median"
        ],
        marker="s",
        linewidth=1.6,
        label="Median",
    )

    axis.fill_between(
        grouped[
            "Fragment"
        ],
        grouped[
            "min"
        ],
        grouped[
            "max"
        ],
        alpha=0.15,
        label="Range across 10 jaw meshes",
    )

    axis.set_xticks(
        [
            1,
            2,
            3,
        ]
    )

    axis.set_xlabel(
        "Fragment in sequential chain"
    )

    axis.set_ylabel(
        "Rotation error [deg]"
    )

    axis.set_title(
        "Accumulated Multiway Rotation Error"
    )

    axis.grid(
        alpha=0.25,
    )

    axis.legend()

    finish_figure(
        OUTPUT_DIRECTORY
        / "model_suite_multiway_rotation_error.png"
    )


# ============================================================
# 6. BPA PARAMETER TRADE-OFF
# ============================================================

def plot_bpa_tradeoff() -> None:

    path = (
        BPA_DIRECTORY
        / "bpa_parameter_summary.csv"
    )

    data = pd.read_csv(
        path
    )

    desired_order = [
        "A_small",
        "B_baseline",
        "C_medium",
        "D_large",
    ]

    data = (
        data
        .set_index(
            "Configuration"
        )
        .loc[
            desired_order
        ]
        .reset_index()
    )

    labels = [
        BPA_LABELS[
            configuration
        ]
        for configuration
        in data[
            "Configuration"
        ]
    ]

    figure, axis_left = (
        plt.subplots(
            figsize=(9.0, 5.3)
        )
    )

    x = np.arange(
        len(
            labels
        )
    )

    bars = (
        axis_left.bar(
            x,
            data[
                "Boundary_Edges_Mean"
            ],
            width=0.62,
            alpha=0.7,
            label="Mean boundary edges",
        )
    )

    axis_left.set_ylabel(
        "Mean number of boundary edges"
    )

    axis_left.set_xlabel(
        "BPA radius-factor configuration"
    )

    axis_left.set_xticks(
        x
    )

    axis_left.set_xticklabels(
        labels,
        rotation=10,
    )

    axis_left.grid(
        axis="y",
        alpha=0.20,
    )

    axis_right = (
        axis_left.twinx()
    )

    axis_right.plot(
        x,
        data[
            "Symmetric_RMSE_Mean"
        ],
        marker="o",
        linewidth=2.0,
        label="Mean symmetric RMSE",
    )

    axis_right.set_ylabel(
        "Mean symmetric RMSE [model units]"
    )

    axis_left.set_title(
        "BPA Parameter Trade-off Across 10 Jaw Meshes"
    )

    handles_left, labels_left = (
        axis_left
        .get_legend_handles_labels()
    )

    handles_right, labels_right = (
        axis_right
        .get_legend_handles_labels()
    )

    axis_left.legend(
        handles_left
        + handles_right,
        labels_left
        + labels_right,
        loc="upper center",
    )

    # Mark selected configuration B.
    axis_left.text(
        1,
        data.loc[
            data[
                "Configuration"
            ]
            == "B_baseline",
            "Boundary_Edges_Mean",
        ].iloc[0]
        + 160,

        "Selected",

        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

    finish_figure(
        OUTPUT_DIRECTORY
        / "model_suite_bpa_parameter_tradeoff.png"
    )


# ============================================================
# 7. BPA VS POISSON BY MODEL
# ============================================================

def plot_mesh_rmse_by_model() -> None:

    path = (
        MESH_DIRECTORY
        / "mesh_validation_by_model.csv"
    )

    data = pd.read_csv(
        path
    )

    pivot = (
        data.pivot(
            index="Model",
            columns="Method",
            values="Symmetric_RMSE_Mean",
        )
    )

    desired_models = [
        "model_01",
        "model_02",
        "model_03",
        "model_04",
        "model_05",
    ]

    pivot = (
        pivot.loc[
            desired_models
        ]
    )

    x = np.arange(
        len(
            desired_models
        )
    )

    width = 0.36

    figure, axis = (
        plt.subplots(
            figsize=(9.0, 5.2)
        )
    )

    axis.bar(
        x - width / 2.0,
        pivot[
            "BPA"
        ],
        width=width,
        label="Ball Pivoting",
    )

    axis.bar(
        x + width / 2.0,
        pivot[
            "Poisson"
        ],
        width=width,
        label="Poisson",
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        [
            MODEL_LABELS[
                model
            ]
            for model
            in desired_models
        ]
    )

    axis.set_xlabel(
        "Dental model set"
    )

    axis.set_ylabel(
        "Mean symmetric RMSE [model units]"
    )

    axis.set_title(
        "Surface Reconstruction Error Across Five Dental Models"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    finish_figure(
        OUTPUT_DIRECTORY
        / "model_suite_mesh_rmse_by_model.png"
    )


# ============================================================
# 8. DIRECTIONAL MESH ERROR
# ============================================================

def plot_mesh_directional_error() -> None:

    path = (
        MESH_DIRECTORY
        / "mesh_validation_summary.csv"
    )

    data = pd.read_csv(
        path
    )

    methods = [
        "BPA",
        "Poisson",
    ]

    data = (
        data
        .set_index(
            "Method"
        )
        .loc[
            methods
        ]
        .reset_index()
    )

    x = np.arange(
        len(
            methods
        )
    )

    width = 0.34

    figure, axis = (
        plt.subplots(
            figsize=(7.4, 4.9)
        )
    )

    axis.bar(
        x - width / 2.0,
        data[
            "Ref_to_Rec_RMSE_Mean"
        ],
        width=width,
        label="Reference → Reconstruction",
    )

    axis.bar(
        x + width / 2.0,
        data[
            "Rec_to_Ref_RMSE_Mean"
        ],
        width=width,
        label="Reconstruction → Reference",
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        [
            "Ball Pivoting",
            "Poisson",
        ]
    )

    axis.set_ylabel(
        "Mean directional RMSE [model units]"
    )

    axis.set_title(
        "Directional Surface Reconstruction Error"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    finish_figure(
        OUTPUT_DIRECTORY
        / "model_suite_mesh_directional_rmse.png"
    )


# ============================================================
# 9. AREA RATIO + SELF INTERSECTIONS
# ============================================================

def plot_mesh_area_ratio() -> None:

    path = (
        MESH_DIRECTORY
        / "mesh_validation_summary.csv"
    )

    data = pd.read_csv(
        path
    )

    data = (
        data
        .set_index(
            "Method"
        )
        .loc[
            [
                "BPA",
                "Poisson",
            ]
        ]
        .reset_index()
    )

    figure, axis = (
        plt.subplots(
            figsize=(6.7, 4.8)
        )
    )

    labels = [
        "Ball Pivoting",
        "Poisson",
    ]

    bars = axis.bar(
        labels,
        data[
            "Area_Ratio_Mean"
        ],
        width=0.55,
    )

    axis.axhline(
        1.0,
        linestyle="--",
        linewidth=1.2,
        label="Reference surface area",
    )

    add_bar_labels(
        axis=axis,
        bars=bars,
        decimals=3,
    )

    axis.set_ylabel(
        "Mean reconstructed/reference area ratio"
    )

    axis.set_title(
        "Reconstructed Surface Area Relative to Reference"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    finish_figure(
        OUTPUT_DIRECTORY
        / "model_suite_mesh_area_ratio.png"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 80)
    print(
        "CREATING FIVE-MODEL REPORT FIGURES"
    )
    print("=" * 80)
    print()

    plot_pairwise_success_rate()

    plot_pairwise_success_by_model()

    plot_noise_robustness()

    plot_multiway_translation_error()

    plot_multiway_rotation_error()

    plot_bpa_tradeoff()

    plot_mesh_rmse_by_model()

    plot_mesh_directional_error()

    plot_mesh_area_ratio()

    print()
    print("=" * 80)
    print(
        "DONE"
    )
    print("=" * 80)

    print()
    print(
        f"Figures written to:\n"
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()