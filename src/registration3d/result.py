from dataclasses import dataclass

import numpy as np


@dataclass
class RegistrationResult:
    algorithm: str
    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    runtime_seconds: float

    # Fehler gegenüber der bekannten Ground Truth.
    # Bei realen Daten ohne Ground Truth bleiben diese Werte None.
    translation_error: float | None = None
    rotation_error_degrees: float | None = None