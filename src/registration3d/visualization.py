import copy

import open3d as o3d


class RegistrationVisualizer:

    @staticmethod
    def show_result(
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        result,
    ) -> None:

        source_copy = copy.deepcopy(source)
        target_copy = copy.deepcopy(target)

        source_copy.transform(
            result.transformation
        )

        source_copy.paint_uniform_color(
            [1.0, 0.2, 0.2]
        )

        target_copy.paint_uniform_color(
            [0.2, 0.8, 0.2]
        )

        o3d.visualization.draw_geometries(
            [
                source_copy,
                target_copy,
            ],
            window_name=result.algorithm,
        )


    @staticmethod
    def show_cloud(
        cloud: o3d.geometry.PointCloud,
        window_name: str = "Point Cloud",
    ) -> None:

        cloud_copy = copy.deepcopy(cloud)

        o3d.visualization.draw_geometries(
            [cloud_copy],
            window_name=window_name,
        )

    @staticmethod
    def show_pair(
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        window_name: str = "Point Clouds",
    ) -> None:

        source_copy = copy.deepcopy(source)
        target_copy = copy.deepcopy(target)

        source_copy.paint_uniform_color([1.0, 0.2, 0.2])
        target_copy.paint_uniform_color([0.2, 0.8, 0.2])

        o3d.visualization.draw_geometries(
            [source_copy, target_copy],
            window_name=window_name,
        )