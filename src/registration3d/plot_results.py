from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_success_rate(
    data: pd.DataFrame,
    series_name: str,
    output_directory: Path,
) -> None:

    subset = data[
        data["Series"] == series_name
    ]

    if subset.empty:
        return

    plt.figure(figsize=(10, 6))

    for algorithm in subset["Algorithm"].unique():

        algorithm_data = subset[
            subset["Algorithm"] == algorithm
        ].sort_values("Variable Value")

        plt.plot(
            algorithm_data["Variable Value"],
            algorithm_data["Success_Rate"],
            marker="o",
            label=algorithm,
        )

    variable_name = subset["Variable"].iloc[0]

    plt.xlabel(variable_name)
    plt.ylabel("Success Rate [%]")

    plt.title(
        f"Registration Success vs. {variable_name}"
    )

    plt.ylim(-5, 105)

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    output_path = (
        output_directory
        / f"success_rate_{series_name}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"Diagramm gespeichert: {output_path}"
    )


def plot_mean_and_median(
    data: pd.DataFrame,
    series_name: str,
    mean_metric: str,
    median_metric: str,
    ylabel: str,
    filename_prefix: str,
    output_directory: Path,
) -> None:

    subset = data[
        data["Series"] == series_name
    ]

    if subset.empty:
        return

    variable_name = subset["Variable"].iloc[0]

    # ---------------------------------------------------------
    # Mittelwert
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for algorithm in subset["Algorithm"].unique():

        algorithm_data = subset[
            subset["Algorithm"] == algorithm
        ].sort_values("Variable Value")

        plt.plot(
            algorithm_data["Variable Value"],
            algorithm_data[mean_metric],
            marker="o",
            label=algorithm,
        )

    plt.xlabel(variable_name)
    plt.ylabel(f"Mean {ylabel}")

    plt.title(
        f"Mean {ylabel} vs. {variable_name}"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    mean_output_path = (
        output_directory
        / f"{filename_prefix}_mean_{series_name}.png"
    )

    plt.savefig(
        mean_output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"Diagramm gespeichert: {mean_output_path}"
    )

    # ---------------------------------------------------------
    # Median
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for algorithm in subset["Algorithm"].unique():

        algorithm_data = subset[
            subset["Algorithm"] == algorithm
        ].sort_values("Variable Value")

        plt.plot(
            algorithm_data["Variable Value"],
            algorithm_data[median_metric],
            marker="o",
            label=algorithm,
        )

    plt.xlabel(variable_name)
    plt.ylabel(f"Median {ylabel}")

    plt.title(
        f"Median {ylabel} vs. {variable_name}"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    median_output_path = (
        output_directory
        / f"{filename_prefix}_median_{series_name}.png"
    )

    plt.savefig(
        median_output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"Diagramm gespeichert: {median_output_path}"
    )


def main() -> None:

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    experiment_directory = (
        project_root
        / "output"
        / "controlled_experiments"
    )

    input_path = (
        experiment_directory
        / "experiments_summary.csv"
    )

    plot_directory = (
        experiment_directory
        / "plots"
    )

    plot_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Ergebnisdatei nicht gefunden: {input_path}"
        )

    data = pd.read_csv(
        input_path
    )

    required_columns = [
        "Series",
        "Variable",
        "Variable Value",
        "Algorithm",
        "Success_Rate",
        "Translation_Error_Mean",
        "Translation_Error_Median",
        "Rotation_Error_Mean",
        "Rotation_Error_Median",
        "Runtime_Mean",
        "Runtime_Median",
        "RMSE_Mean",
        "RMSE_Median",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Folgende benötigte Spalten fehlen in "
            "experiments_summary.csv: "
            + ", ".join(missing_columns)
        )

    series_names = [
        "overlap",
        "rotation",
        "translation",
        "noise",
    ]

    for series_name in series_names:

        # -----------------------------------------------------
        # Erfolgsrate
        # -----------------------------------------------------

        plot_success_rate(
            data=data,
            series_name=series_name,
            output_directory=plot_directory,
        )

        # -----------------------------------------------------
        # Translationsfehler
        # -----------------------------------------------------

        plot_mean_and_median(
            data=data,
            series_name=series_name,
            mean_metric="Translation_Error_Mean",
            median_metric="Translation_Error_Median",
            ylabel="Translation Error",
            filename_prefix="translation_error",
            output_directory=plot_directory,
        )

        # -----------------------------------------------------
        # Rotationsfehler
        # -----------------------------------------------------

        plot_mean_and_median(
            data=data,
            series_name=series_name,
            mean_metric="Rotation_Error_Mean",
            median_metric="Rotation_Error_Median",
            ylabel="Rotation Error [deg]",
            filename_prefix="rotation_error",
            output_directory=plot_directory,
        )

        # -----------------------------------------------------
        # RMSE
        # -----------------------------------------------------

        plot_mean_and_median(
            data=data,
            series_name=series_name,
            mean_metric="RMSE_Mean",
            median_metric="RMSE_Median",
            ylabel="Inlier RMSE",
            filename_prefix="rmse",
            output_directory=plot_directory,
        )

        # -----------------------------------------------------
        # Laufzeit
        # -----------------------------------------------------

        plot_mean_and_median(
            data=data,
            series_name=series_name,
            mean_metric="Runtime_Mean",
            median_metric="Runtime_Median",
            ylabel="Runtime [s]",
            filename_prefix="runtime",
            output_directory=plot_directory,
        )


if __name__ == "__main__":
    main()