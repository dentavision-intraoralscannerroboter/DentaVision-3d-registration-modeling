import time

import numpy as np
import open3d as o3d

from registration3d.algorithms.base import RegistrationAlgorithm
from registration3d.config import RegistrationConfig
from registration3d.result import RegistrationResult


class PointToPointICP(RegistrationAlgorithm):

    def __init__(self, config: RegistrationConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "Point-to-Point ICP"

    def register(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        initial_transform: np.ndarray | None = None,
    ) -> RegistrationResult:

        if initial_transform is None:
            initial_transform = np.eye(4)

        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=self.config.max_iterations
        )

        start_time = time.perf_counter()

        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            self.config.max_correspondence_distance,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria,
        )

        runtime = time.perf_counter() - start_time

        return RegistrationResult(
            algorithm=self.name,
            transformation=result.transformation,
            fitness=result.fitness,
            inlier_rmse=result.inlier_rmse,
            runtime_seconds=runtime,
        )