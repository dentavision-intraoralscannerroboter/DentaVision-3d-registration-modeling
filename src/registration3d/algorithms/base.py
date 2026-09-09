from abc import ABC, abstractmethod

import numpy as np
import open3d as o3d

from registration3d.result import RegistrationResult


class RegistrationAlgorithm(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Name des Registrierungsalgorithmus."""
        pass

    @abstractmethod
    def register(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        initial_transform: np.ndarray | None = None,
    ) -> RegistrationResult:
        """Registriert source auf target."""
        pass