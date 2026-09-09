from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd

from registration3d.config import RegistrationConfig
from registration3d.result import RegistrationResult


class RegistrationEvaluator:

    @staticmethod
    def evaluate_result(
        result: RegistrationResult,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        config: RegistrationConfig,
        ground_truth: np.ndarray | None = None,
    ) -> RegistrationResult:
        """
        Bewertet eine berechnete Registrierung.

        Fitness und RMSE werden für alle Algorithmen mit derselben
        Korrespondenzdistanz berechnet.

        Falls eine Ground-Truth-Transformation vorhanden ist, werden
        zusätzlich Translations- und Rotationsfehler bestimmt.
        """

        # -----------------------------------------------------
        # 1. Einheitliche Open3D-Evaluation
        # -----------------------------------------------------

        evaluation = (
            o3d.pipelines.registration.evaluate_registration(
                source,
                target,
                config.evaluation_distance,
                result.transformation,
            )
        )

        result.fitness = evaluation.fitness
        result.inlier_rmse = evaluation.inlier_rmse

        # -----------------------------------------------------
        # 2. Ground-Truth-Evaluation
        # -----------------------------------------------------

        if ground_truth is not None:

            if ground_truth.shape != (4, 4):
                raise ValueError(
                    "Ground Truth muss eine 4x4-Transformationsmatrix sein."
                )

            # Relative Abweichung:
            #
            # T_error = T_ground_truth^-1 * T_estimated
            #
            # Bei perfekter Registrierung ergibt sich die Einheitsmatrix.
            error_transform = (
                np.linalg.inv(ground_truth)
                @ result.transformation
            )

            # -------------------------------------------------
            # 2a. Translationsfehler
            # -------------------------------------------------

            translation_error = np.linalg.norm(
                error_transform[:3, 3]
            )

            result.translation_error = float(
                translation_error
            )

            # -------------------------------------------------
            # 2b. Rotationsfehler
            # -------------------------------------------------

            rotation_error = error_transform[:3, :3]

            cos_angle = (
                np.trace(rotation_error) - 1.0
            ) / 2.0

            # Numerische Rundungsfehler können beispielsweise
            # 1.0000000001 erzeugen. arccos wäre dann undefiniert.
            cos_angle = np.clip(
                cos_angle,
                -1.0,
                1.0,
            )

            angle_radians = np.arccos(
                cos_angle
            )

            result.rotation_error_degrees = float(
                np.degrees(angle_radians)
            )

        return result

    @staticmethod
    def print_result(
        result: RegistrationResult,
    ) -> None:

        print()
        print("=" * 60)
        print(result.algorithm)
        print("=" * 60)

        print(
            f"Fitness:             "
            f"{result.fitness:.6f}"
        )

        print(
            f"Inlier RMSE:         "
            f"{result.inlier_rmse:.6f}"
        )

        print(
            f"Runtime:             "
            f"{result.runtime_seconds:.4f} s"
        )

        if result.translation_error is not None:
            print(
                f"Translation Error:   "
                f"{result.translation_error:.6f}"
            )

        if result.rotation_error_degrees is not None:
            print(
                f"Rotation Error:      "
                f"{result.rotation_error_degrees:.6f}°"
            )

        print()
        print("Transformation:")
        print(result.transformation)

    @staticmethod
    def create_comparison_table(
        results: list[RegistrationResult],
    ) -> pd.DataFrame:

        data = []

        for result in results:
            data.append(
                {
                    "Algorithm": result.algorithm,
                    "Fitness": result.fitness,
                    "Inlier RMSE": result.inlier_rmse,
                    "Translation Error":
                        result.translation_error,
                    "Rotation Error [deg]":
                        result.rotation_error_degrees,
                    "Runtime [s]":
                        result.runtime_seconds,
                }
            )

        return pd.DataFrame(data)

    @staticmethod
    def print_comparison_table(
        results: list[RegistrationResult],
    ) -> None:

        table = (
            RegistrationEvaluator
            .create_comparison_table(results)
        )

        print()
        print("=" * 120)
        print("ALGORITHMUSVERGLEICH")
        print("=" * 120)

        print(
            table.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    @staticmethod
    def save_results_csv(
        results: list[RegistrationResult],
        output_path: str | Path,
    ) -> None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        table = (
            RegistrationEvaluator
            .create_comparison_table(results)
        )

        table.to_csv(
            output_path,
            index=False,
        )

        print()
        print(
            f"Ergebnisse gespeichert: "
            f"{output_path}"
        )