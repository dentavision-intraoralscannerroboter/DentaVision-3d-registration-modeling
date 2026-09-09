import copy

import numpy as np
import open3d as o3d

from registration3d.reconstruction.mesh_config import (
    MeshConfig,
)


class MeshReconstructor:

    def __init__(
        self,
        config: MeshConfig,
    ) -> None:

        self.config = config

    # =========================================================
    # Punktabstand bestimmen
    # =========================================================

    @staticmethod
    def estimate_mean_point_distance(
        cloud: o3d.geometry.PointCloud,
    ) -> float:

        if cloud.is_empty():
            raise ValueError(
                "Die Punktwolke ist leer."
            )

        distances = (
            cloud.compute_nearest_neighbor_distance()
        )

        if len(distances) == 0:
            raise ValueError(
                "Punktabstände konnten nicht bestimmt werden."
            )

        mean_distance = float(
            np.mean(
                np.asarray(
                    distances
                )
            )
        )

        if (
            not np.isfinite(
                mean_distance
            )
            or mean_distance <= 0.0
        ):
            raise ValueError(
                "Ungültiger mittlerer Punktabstand."
            )

        return mean_distance

    # =========================================================
    # Normalen vorbereiten
    # =========================================================

    def prepare_point_cloud(
        self,
        cloud: o3d.geometry.PointCloud,
    ) -> o3d.geometry.PointCloud:

        if cloud.is_empty():
            raise ValueError(
                "Die Punktwolke ist leer."
            )

        prepared_cloud = (
            copy.deepcopy(
                cloud
            )
        )

        mean_distance = (
            self.estimate_mean_point_distance(
                prepared_cloud
            )
        )

        normal_radius = (
            mean_distance
            * self.config.normal_radius_factor
        )

        print()
        print(
            "Bereite Punktwolke "
            "für Mesh-Rekonstruktion vor ..."
        )

        print(
            f"  Punkte: "
            f"{len(prepared_cloud.points):,}"
        )

        print(
            f"  Mittlerer Punktabstand: "
            f"{mean_distance:.6f}"
        )

        print(
            f"  Normalenradius: "
            f"{normal_radius:.6f}"
        )

        # -----------------------------------------------------
        # Normalen schätzen
        # -----------------------------------------------------

        prepared_cloud.estimate_normals(
            search_param=(
                o3d.geometry
                .KDTreeSearchParamHybrid(
                    radius=(
                        normal_radius
                    ),
                    max_nn=(
                        self.config
                        .normal_max_nn
                    ),
                )
            )
        )

        # -----------------------------------------------------
        # Normalen möglichst konsistent ausrichten
        #
        # Poisson benötigt sinnvoll orientierte Normalen.
        # -----------------------------------------------------

        number_of_points = (
            len(
                prepared_cloud.points
            )
        )

        if number_of_points >= 10:

            orientation_neighbors = min(
                30,
                number_of_points - 1,
            )

            prepared_cloud\
                .orient_normals_consistent_tangent_plane(
                    orientation_neighbors
                )

        prepared_cloud.normalize_normals()

        print(
            "  Normalen geschätzt "
            "und ausgerichtet."
        )

        return prepared_cloud

    # =========================================================
    # Poisson Surface Reconstruction
    # =========================================================

    def reconstruct_poisson(
        self,
        cloud: o3d.geometry.PointCloud,
    ) -> tuple[
        o3d.geometry.TriangleMesh,
        np.ndarray,
    ]:

        if cloud.is_empty():
            raise ValueError(
                "Die Punktwolke ist leer."
            )

        if not cloud.has_normals():
            raise ValueError(
                "Für Poisson Reconstruction "
                "werden Normalen benötigt."
            )

        print()
        print("=" * 80)
        print(
            "POISSON SURFACE RECONSTRUCTION"
        )
        print("=" * 80)

        print(
            f"Poisson Depth: "
            f"{self.config.poisson_depth}"
        )

        mesh, densities = (
            o3d.geometry.TriangleMesh
            .create_from_point_cloud_poisson(
                cloud,
                depth=(
                    self.config
                    .poisson_depth
                ),
            )
        )

        densities = (
            np.asarray(
                densities
            )
        )

        mesh.compute_vertex_normals()

        print()
        print(
            "Poisson-Rekonstruktion abgeschlossen."
        )

        print(
            f"  Vertices: "
            f"{len(mesh.vertices):,}"
        )

        print(
            f"  Dreiecke: "
            f"{len(mesh.triangles):,}"
        )

        return (
            mesh,
            densities,
        )

    # =========================================================
    # Niedrige Poisson-Dichten entfernen
    # =========================================================

    def remove_low_density_vertices(
        self,
        mesh: o3d.geometry.TriangleMesh,
        densities: np.ndarray,
    ) -> o3d.geometry.TriangleMesh:

        if len(mesh.vertices) == 0:
            raise ValueError(
                "Das Mesh ist leer."
            )

        if (
            len(densities)
            != len(mesh.vertices)
        ):
            raise ValueError(
                "Die Anzahl der Dichtewerte stimmt "
                "nicht mit der Anzahl der Mesh-Vertices überein."
            )

        cleaned_mesh = (
            copy.deepcopy(
                mesh
            )
        )

        quantile = (
            self.config
            .poisson_density_quantile
        )

        density_threshold = float(
            np.quantile(
                densities,
                quantile,
            )
        )

        vertices_to_remove = (
            densities
            < density_threshold
        )

        number_to_remove = int(
            np.count_nonzero(
                vertices_to_remove
            )
        )

        print()
        print(
            "Entferne Bereiche "
            "mit niedriger Poisson-Dichte ..."
        )

        print(
            f"  Quantil: "
            f"{quantile:.3f}"
        )

        print(
            f"  Density Threshold: "
            f"{density_threshold:.6f}"
        )

        print(
            f"  Zu entfernende Vertices: "
            f"{number_to_remove:,}"
        )

        cleaned_mesh.remove_vertices_by_mask(
            vertices_to_remove
        )

        cleaned_mesh.compute_vertex_normals()

        print(
            f"  Verbleibende Vertices: "
            f"{len(cleaned_mesh.vertices):,}"
        )

        print(
            f"  Verbleibende Dreiecke: "
            f"{len(cleaned_mesh.triangles):,}"
        )

        return cleaned_mesh


        # =========================================================
    # Ball Pivoting Reconstruction
    # =========================================================

    def reconstruct_ball_pivoting(
        self,
        cloud: o3d.geometry.PointCloud,
    ) -> o3d.geometry.TriangleMesh:

        if cloud.is_empty():
            raise ValueError(
                "Die Punktwolke ist leer."
            )

        if not cloud.has_normals():
            raise ValueError(
                "Für Ball Pivoting werden "
                "Punktnormalen benötigt."
            )

        print()
        print("=" * 80)
        print(
            "BALL PIVOTING RECONSTRUCTION"
        )
        print("=" * 80)

        # -----------------------------------------------------
        # Mittleren Punktabstand bestimmen
        # -----------------------------------------------------

        mean_distance = (
            self.estimate_mean_point_distance(
                cloud
            )
        )

        # -----------------------------------------------------
        # BPA-Radien aus den konfigurierten Faktoren erzeugen
        #
        # Beispiel:
        #
        # mean_distance = 0.16
        #
        # Faktoren:
        # 1.5, 2.0, 3.0
        #
        # Radien:
        # 0.24, 0.32, 0.48
        # -----------------------------------------------------

        radii = [
            mean_distance * factor
            for factor
            in self.config
            .ball_pivoting_radius_factors
        ]

        print()
        print(
            f"Mittlerer Punktabstand: "
            f"{mean_distance:.6f}"
        )

        print(
            "Ball-Pivoting-Radien:"
        )

        for index, radius in enumerate(
            radii,
            start=1,
        ):

            print(
                f"  Radius {index}: "
                f"{radius:.6f}"
            )

        # -----------------------------------------------------
        # Open3D benötigt DoubleVector
        # -----------------------------------------------------

        o3d_radii = (
            o3d.utility.DoubleVector(
                radii
            )
        )

        # -----------------------------------------------------
        # Ball Pivoting
        # -----------------------------------------------------

        mesh = (
            o3d.geometry.TriangleMesh
            .create_from_point_cloud_ball_pivoting(
                cloud,
                o3d_radii,
            )
        )

        if len(mesh.vertices) == 0:

            raise RuntimeError(
                "Ball Pivoting hat ein "
                "leeres Mesh erzeugt."
            )

        if len(mesh.triangles) == 0:

            raise RuntimeError(
                "Ball Pivoting hat keine "
                "Dreiecke erzeugt."
            )

        mesh.compute_vertex_normals()

        print()
        print(
            "Ball-Pivoting-Rekonstruktion "
            "abgeschlossen."
        )

        print(
            f"  Vertices: "
            f"{len(mesh.vertices):,}"
        )

        print(
            f"  Dreiecke: "
            f"{len(mesh.triangles):,}"
        )

        return mesh