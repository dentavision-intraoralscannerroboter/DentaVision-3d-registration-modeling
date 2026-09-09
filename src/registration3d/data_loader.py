from pathlib import Path

import open3d as o3d


class PointCloudLoader:

    @staticmethod
    def load(path: str | Path) -> o3d.geometry.PointCloud:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Punktwolke nicht gefunden: {path}"
            )

        cloud = o3d.io.read_point_cloud(
            str(path),
            remove_nan_points=True,
            remove_infinite_points=True,
        )

        if cloud.is_empty():
            raise ValueError(
                f"Punktwolke ist leer oder konnte nicht gelesen werden: {path}"
            )

        print(
            f"Geladen: {path.name} "
            f"({len(cloud.points):,} Punkte)"
        )

        return cloud