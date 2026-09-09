import time

import numpy as np
import open3d as o3d

from registration3d.algorithms.base import (
    RegistrationAlgorithm,
)
from registration3d.config import RegistrationConfig
from registration3d.features import FPFHFeatureExtractor
from registration3d.result import RegistrationResult


class FgrFPFHICP(RegistrationAlgorithm):

    def __init__(
        self,
        config: RegistrationConfig,
    ):
        self.config = config

        self.feature_extractor = (
            FPFHFeatureExtractor(config)
        )

    @property
    def name(self) -> str:
        return "FPFH + FGR + ICP"

    def register(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        initial_transform: np.ndarray | None = None,
    ) -> RegistrationResult:

        start_time = time.perf_counter()

        # --------------------------------------------------
        # 1. FPFH Features
        # --------------------------------------------------

        source_fpfh = self.feature_extractor.compute(
            source
        )

        target_fpfh = self.feature_extractor.compute(
            target
        )

        # --------------------------------------------------
        # 2. Fast Global Registration
        # --------------------------------------------------

        fgr_result = (
            o3d.pipelines.registration
            .registration_fgr_based_on_feature_matching(
                source,
                target,
                source_fpfh,
                target_fpfh,
                o3d.pipelines.registration
                .FastGlobalRegistrationOption(
                    maximum_correspondence_distance=(
                        self.config.fgr_distance
                    )
                ),
            )
        )

        # --------------------------------------------------
        # 3. Point-to-Plane ICP
        # --------------------------------------------------

        icp_result = (
            o3d.pipelines.registration
            .registration_icp(
                source,
                target,
                self.config.refinement_distance,
                fgr_result.transformation,
                o3d.pipelines.registration
                .TransformationEstimationPointToPlane(),
                o3d.pipelines.registration
                .ICPConvergenceCriteria(
                    max_iteration=(
                        self.config.max_iterations
                    )
                ),
            )
        )

        runtime = (
            time.perf_counter()
            - start_time
        )

        return RegistrationResult(
            algorithm=self.name,
            transformation=icp_result.transformation,
            fitness=icp_result.fitness,
            inlier_rmse=icp_result.inlier_rmse,
            runtime_seconds=runtime,
        )