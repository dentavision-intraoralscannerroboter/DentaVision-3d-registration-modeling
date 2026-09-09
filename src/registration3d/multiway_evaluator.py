from pathlib import Path

import numpy as np
import pandas as pd


class MultiwayEvaluator:

    @staticmethod
    def evaluate_poses(
        estimated_poses: list[np.ndarray],
        ground_truth_poses: np.ndarray,
    ) -> pd.DataFrame:

        if ground_truth_poses.ndim != 3:
            raise ValueError(
                "Ground Truth muss die Form "
                "(N, 4, 4) haben."
            )

        if (
            ground_truth_poses.shape[1:]
            != (4, 4)
        ):
            raise ValueError(
                "Jede Ground-Truth-Pose "
                "muss eine 4x4-Matrix sein."
            )

        if (
            len(estimated_poses)
            != len(ground_truth_poses)
        ):
            raise ValueError(
                "Anzahl der geschätzten und "
                "Ground-Truth-Posen stimmt "
                "nicht überein."
            )

        rows = []

        for index, (
            estimated,
            ground_truth,
        ) in enumerate(
            zip(
                estimated_poses,
                ground_truth_poses,
            )
        ):

            error_transform = (
                np.linalg.inv(
                    ground_truth
                )
                @ estimated
            )

            # ---------------------------------------------
            # Translation Error
            # ---------------------------------------------

            translation_error = float(
                np.linalg.norm(
                    error_transform[
                        :3,
                        3,
                    ]
                )
            )

            # ---------------------------------------------
            # Rotation Error
            # ---------------------------------------------

            rotation_error = (
                error_transform[
                    :3,
                    :3,
                ]
            )

            cos_angle = (
                np.trace(
                    rotation_error
                )
                - 1.0
            ) / 2.0

            cos_angle = np.clip(
                cos_angle,
                -1.0,
                1.0,
            )

            rotation_error_degrees = float(
                np.degrees(
                    np.arccos(
                        cos_angle
                    )
                )
            )

            rows.append(
                {
                    "Fragment":
                        index,

                    "Translation Error":
                        translation_error,

                    "Rotation Error [deg]":
                        rotation_error_degrees,
                }
            )

        return pd.DataFrame(
            rows
        )

    @staticmethod
    def save_csv(
        table: pd.DataFrame,
        output_path: str | Path,
    ) -> None:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        table.to_csv(
            output_path,
            index=False,
        )