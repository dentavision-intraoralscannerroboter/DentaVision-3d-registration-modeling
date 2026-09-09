from pathlib import Path

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
from registration3d.config import RegistrationConfig
from registration3d.evaluator import (
    RegistrationEvaluator,
)
from registration3d.experiment_config import (
    ExperimentConfig,
)
from registration3d.preprocessing import (
    PointCloudPreprocessor,
)
from registration3d.synthetic_data import (
    SyntheticDentalDataGenerator,
)


class ExperimentRunner:

    def __init__(
        self,
        registration_config: RegistrationConfig,
        experiment_config: ExperimentConfig,
        output_directory: str | Path,
    ):

        self.registration_config = registration_config
        self.experiment_config = experiment_config

        self.output_directory = Path(
            output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.preprocessor = (
            PointCloudPreprocessor(
                registration_config
            )
        )

        self.generator = (
            SyntheticDentalDataGenerator(
                number_of_points=(
                    experiment_config.number_of_points
                )
            )
        )

    # ---------------------------------------------------------
    # Algorithmen
    # ---------------------------------------------------------

    def _create_algorithms(self):

        config = self.registration_config

        return [
            PointToPointICP(config),
            PointToPlaneICP(config),
            GeneralizedICP(config),
            RansacFPFHICP(config),
            FgrFPFHICP(config),
        ]

    # ---------------------------------------------------------
    # Experiment starten
    # ---------------------------------------------------------

    def run(self) -> None:

        rows = []

        # Jede Versuchsreihe bekommt einen eigenen
        # Seed-Bereich.
        #
        # Innerhalb derselben Reihe verwendet dieselbe
        # Wiederholung denselben Datensatz-Seed.
        series_seed_offsets = {
            "overlap": 0,
            "rotation": 10_000,
            "translation": 20_000,
            "noise": 30_000,
        }

        scenarios = (
            self.experiment_config.scenarios
        )

        repeats = (
            self.experiment_config.repeats
        )

        total_datasets = (
            len(scenarios)
            * repeats
        )

        current_dataset = 0

        # -----------------------------------------------------
        # Alle Szenarien
        # -----------------------------------------------------

        for scenario in scenarios:

            if scenario.series not in series_seed_offsets:
                raise ValueError(
                    f"Unbekannte Versuchsreihe: "
                    f"{scenario.series}"
                )

            series_offset = (
                series_seed_offsets[
                    scenario.series
                ]
            )

            print()
            print("=" * 80)

            print(
                f"VERSUCHSREIHE: "
                f"{scenario.series.upper()}"
            )

            print(
                f"SZENARIO: {scenario.name}"
            )

            print("=" * 80)

            print(
                f"Overlap:      "
                f"{scenario.overlap_fraction:.0%}"
            )

            print(
                f"Rotation:     "
                f"{scenario.rotation_degrees:.1f}°"
            )

            print(
                f"Translation:  "
                f"{scenario.translation_magnitude:.3f}"
            )

            print(
                f"Noise:        "
                f"{scenario.noise_std:.3f}"
            )

            # -------------------------------------------------
            # Wiederholungen
            # -------------------------------------------------

            for repeat in range(repeats):

                current_dataset += 1

                # ---------------------------------------------
                # Datensatz-Seed
                # ---------------------------------------------
                #
                # Innerhalb einer Versuchsreihe bekommt
                # dieselbe Wiederholung bei jedem Variablenwert
                # denselben Seed.
                #
                # Beispiel:
                #
                # rotation_0, Repeat 1  -> gleicher Seed
                # rotation_5, Repeat 1  -> gleicher Seed
                # rotation_10, Repeat 1 -> gleicher Seed
                #
                # Damit ändert sich möglichst nur die
                # untersuchte Variable.

                data_seed = (
                    self.experiment_config.base_seed
                    + series_offset
                    + repeat
                )

                print()
                print(
                    f"Datensatz "
                    f"{current_dataset}/"
                    f"{total_datasets}"
                    f" - Wiederholung "
                    f"{repeat + 1}/{repeats}"
                )

                print(
                    f"  Data Seed: {data_seed}"
                )

                # ---------------------------------------------
                # Synthetische Daten erzeugen
                # ---------------------------------------------

                dataset = self.generator.generate(
                    scenario=scenario,
                    seed=data_seed,
                )

                # ---------------------------------------------
                # Preprocessing
                # ---------------------------------------------

                source = (
                    self.preprocessor.preprocess(
                        dataset.source
                    )
                )

                target = (
                    self.preprocessor.preprocess(
                        dataset.target
                    )
                )

                # ---------------------------------------------
                # Beispielwolken speichern
                # ---------------------------------------------

                if repeat == 0:

                    self._save_example_clouds(
                        scenario.name,
                        dataset.source,
                        dataset.target,
                    )

                # ---------------------------------------------
                # Algorithmen erzeugen
                # ---------------------------------------------

                algorithms = (
                    self._create_algorithms()
                )

                # ---------------------------------------------
                # Alle Algorithmen ausführen
                # ---------------------------------------------

                for algorithm_index, algorithm in enumerate(
                    algorithms
                ):

                    print(
                        f"  -> {algorithm.name}"
                    )

                    # -----------------------------------------
                    # Algorithmus-Seed
                    # -----------------------------------------
                    #
                    # Insbesondere RANSAC arbeitet zufällig.
                    #
                    # Derselbe Algorithmus bekommt innerhalb
                    # einer Versuchsreihe bei derselben
                    # Wiederholung denselben Seed.

                    algorithm_seed = (
                        self.experiment_config.base_seed
                        + series_offset
                        + repeat * 100
                        + algorithm_index
                    )

                    o3d.utility.random.seed(
                        algorithm_seed
                    )

                    # -----------------------------------------
                    # Registrierung
                    # -----------------------------------------

                    result = algorithm.register(
                        source,
                        target,
                    )

                    # -----------------------------------------
                    # Einheitliche Evaluation
                    # -----------------------------------------

                    result = (
                        RegistrationEvaluator
                        .evaluate_result(
                            result=result,
                            source=source,
                            target=target,
                            config=(
                                self.registration_config
                            ),
                            ground_truth=(
                                dataset.ground_truth
                            ),
                        )
                    )

                    # -----------------------------------------
                    # Erfolg bestimmen
                    # -----------------------------------------

                    success = self._is_success(
                        result.translation_error,
                        result.rotation_error_degrees,
                    )

                    # -----------------------------------------
                    # RMSE absichern
                    # -----------------------------------------
                    #
                    # Bei Fitness = 0 wurden keine gültigen
                    # Korrespondenzen gefunden.
                    #
                    # Open3D kann dann RMSE = 0 liefern.
                    # Das darf nicht als perfektes Ergebnis
                    # interpretiert werden.

                    if result.fitness > 0.0:
                        valid_rmse = (
                            result.inlier_rmse
                        )
                    else:
                        valid_rmse = None

                    # -----------------------------------------
                    # Ergebnis speichern
                    # -----------------------------------------

                    rows.append(
                        {
                            "Series":
                                scenario.series,

                            "Scenario":
                                scenario.name,

                            "Variable":
                                scenario.variable_name,

                            "Variable Value":
                                scenario.variable_value,

                            "Repeat":
                                repeat + 1,

                            "Data Seed":
                                data_seed,

                            "Algorithm Seed":
                                algorithm_seed,

                            "Algorithm":
                                result.algorithm,

                            "Overlap":
                                scenario.overlap_fraction,

                            "Rotation Input [deg]":
                                scenario.rotation_degrees,

                            "Translation Input":
                                scenario.translation_magnitude,

                            "Noise Std":
                                scenario.noise_std,

                            "Fitness":
                                result.fitness,

                            "Inlier RMSE":
                                valid_rmse,

                            "Translation Error":
                                result.translation_error,

                            "Rotation Error [deg]":
                                result.rotation_error_degrees,

                            "Runtime [s]":
                                result.runtime_seconds,

                            "Success":
                                success,
                        }
                    )

        # -----------------------------------------------------
        # DataFrames erzeugen
        # -----------------------------------------------------

        raw_results = pd.DataFrame(
            rows
        )

        summary = self._create_summary(
            raw_results
        )

        # -----------------------------------------------------
        # Speichern
        # -----------------------------------------------------

        self._save_results(
            raw_results,
            summary,
        )

        print()
        print("=" * 80)
        print("EXPERIMENT ABGESCHLOSSEN")
        print("=" * 80)

        print()

        print(
            summary.to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Erfolgskriterium
    # ---------------------------------------------------------

    def _is_success(
        self,
        translation_error: float | None,
        rotation_error: float | None,
    ) -> bool:

        if (
            translation_error is None
            or rotation_error is None
        ):
            return False

        translation_ok = (
            translation_error
            <= (
                self.experiment_config
                .success_translation_threshold
            )
        )

        rotation_ok = (
            rotation_error
            <= (
                self.experiment_config
                .success_rotation_threshold_degrees
            )
        )

        return (
            translation_ok
            and rotation_ok
        )

    # ---------------------------------------------------------
    # Zusammenfassung
    # ---------------------------------------------------------

    @staticmethod
    def _create_summary(
        raw_results: pd.DataFrame,
    ) -> pd.DataFrame:

        summary = (
            raw_results
            .groupby(
                [
                    "Series",
                    "Variable",
                    "Variable Value",
                    "Algorithm",
                ],
                as_index=False,
            )
            .agg(
                # ---------------------------------------------
                # Fitness
                # ---------------------------------------------

                Fitness_Mean=(
                    "Fitness",
                    "mean",
                ),

                Fitness_Std=(
                    "Fitness",
                    "std",
                ),

                Fitness_Median=(
                    "Fitness",
                    "median",
                ),

                # ---------------------------------------------
                # RMSE
                # ---------------------------------------------

                RMSE_Mean=(
                    "Inlier RMSE",
                    "mean",
                ),

                RMSE_Median=(
                    "Inlier RMSE",
                    "median",
                ),

                # ---------------------------------------------
                # Translationsfehler
                # ---------------------------------------------

                Translation_Error_Mean=(
                    "Translation Error",
                    "mean",
                ),

                Translation_Error_Std=(
                    "Translation Error",
                    "std",
                ),

                Translation_Error_Median=(
                    "Translation Error",
                    "median",
                ),

                # ---------------------------------------------
                # Rotationsfehler
                # ---------------------------------------------

                Rotation_Error_Mean=(
                    "Rotation Error [deg]",
                    "mean",
                ),

                Rotation_Error_Std=(
                    "Rotation Error [deg]",
                    "std",
                ),

                Rotation_Error_Median=(
                    "Rotation Error [deg]",
                    "median",
                ),

                # ---------------------------------------------
                # Laufzeit
                # ---------------------------------------------

                Runtime_Mean=(
                    "Runtime [s]",
                    "mean",
                ),

                Runtime_Std=(
                    "Runtime [s]",
                    "std",
                ),

                Runtime_Median=(
                    "Runtime [s]",
                    "median",
                ),

                # ---------------------------------------------
                # Erfolgsrate
                # ---------------------------------------------

                Success_Rate=(
                    "Success",
                    "mean",
                ),
            )
        )

        # Boolean-Mittelwert 0..1 -> Prozent
        summary["Success_Rate"] *= 100.0

        return summary

    # ---------------------------------------------------------
    # CSV-Dateien
    # ---------------------------------------------------------

    def _save_results(
        self,
        raw_results: pd.DataFrame,
        summary: pd.DataFrame,
    ) -> None:

        raw_path = (
            self.output_directory
            / "experiments_raw.csv"
        )

        summary_path = (
            self.output_directory
            / "experiments_summary.csv"
        )

        raw_results.to_csv(
            raw_path,
            index=False,
        )

        summary.to_csv(
            summary_path,
            index=False,
        )

        print()
        print(
            f"Rohdaten gespeichert: "
            f"{raw_path}"
        )

        print(
            f"Zusammenfassung gespeichert: "
            f"{summary_path}"
        )

    # ---------------------------------------------------------
    # Beispielpunktwolken
    # ---------------------------------------------------------

    def _save_example_clouds(
        self,
        scenario_name: str,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
    ) -> None:

        example_directory = (
            self.output_directory
            / "examples"
        )

        example_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        o3d.io.write_point_cloud(
            str(
                example_directory
                / f"{scenario_name}_source.ply"
            ),
            source,
        )

        o3d.io.write_point_cloud(
            str(
                example_directory
                / f"{scenario_name}_target.ply"
            ),
            target,
        )