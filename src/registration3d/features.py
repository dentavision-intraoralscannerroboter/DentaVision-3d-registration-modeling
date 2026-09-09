import open3d as o3d

from registration3d.config import RegistrationConfig


class FPFHFeatureExtractor:

    def __init__(
        self,
        config: RegistrationConfig,
    ):
        self.config = config

    def compute(
        self,
        cloud: o3d.geometry.PointCloud,
    ) -> o3d.pipelines.registration.Feature:

        if not cloud.has_normals():
            raise ValueError(
                "Für FPFH müssen zuerst Normalen "
                "berechnet werden."
            )

        features = (
            o3d.pipelines.registration
            .compute_fpfh_feature(
                cloud,
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.config.fpfh_radius,
                    max_nn=self.config.fpfh_max_nn,
                ),
            )
        )

        return features