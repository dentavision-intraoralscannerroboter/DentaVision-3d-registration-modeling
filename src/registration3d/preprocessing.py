import open3d as o3d

from registration3d.config import RegistrationConfig


class PointCloudPreprocessor:
    def __init__(self, config: RegistrationConfig):
        self.config = config

    def preprocess(
        self,
        cloud: o3d.geometry.PointCloud,
    ) -> o3d.geometry.PointCloud:

        # 1. Punktwolke verkleinern
        downsampled = cloud.voxel_down_sample(
            voxel_size=self.config.voxel_size
        )

        if downsampled.is_empty():
            raise ValueError(
                "Punktwolke ist nach dem Downsampling leer."
            )

        # 2. Oberflächennormalen berechnen
        downsampled.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.config.normal_radius,
                max_nn=self.config.max_nn,
            )
        )

        return downsampled