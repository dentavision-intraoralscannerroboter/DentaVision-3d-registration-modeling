from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from registration3d.reconstruction.mesh_config import (
    MeshConfig,
)


@dataclass
class MeshEvaluationResult:

    # =========================================================
    # Größe
    # =========================================================

    vertices: int
    triangles: int

    # =========================================================
    # Topologie
    # =========================================================

    connected_components: int
    largest_component_triangles: int
    largest_component_ratio: float

    edge_manifold_with_boundary: bool
    edge_manifold_without_boundary: bool
    vertex_manifold: bool
    self_intersecting: bool
    watertight: bool
    orientable: bool

    # =========================================================
    # Geometrischer Abstand
    #
    # Approximation über gesampelte Mesh-Punkte.
    # =========================================================

    cloud_to_mesh_mean: float
    cloud_to_mesh_median: float
    cloud_to_mesh_rmse: float
    cloud_to_mesh_p95: float
    cloud_to_mesh_max: float

    mesh_to_cloud_mean: float
    mesh_to_cloud_median: float
    mesh_to_cloud_rmse: float
    mesh_to_cloud_p95: float
    mesh_to_cloud_max: float

    symmetric_mean_distance: float
    symmetric_rmse: float


class MeshEvaluator:

    def __init__(
        self,
        config: MeshConfig,
    ) -> None:

        self.config = config

    # =========================================================
    # Hauptauswertung
    # =========================================================

    def evaluate(
        self,
        mesh: o3d.geometry.TriangleMesh,
        reference_cloud: o3d.geometry.PointCloud,
    ) -> MeshEvaluationResult:

        if len(mesh.vertices) == 0:
            raise ValueError(
                "Das zu evaluierende Mesh enthält "
                "keine Vertices."
            )

        if len(mesh.triangles) == 0:
            raise ValueError(
                "Das zu evaluierende Mesh enthält "
                "keine Dreiecke."
            )

        if reference_cloud.is_empty():
            raise ValueError(
                "Die Referenzpunktwolke ist leer."
            )

        print()
        print("=" * 80)
        print("MESH EVALUATION")
        print("=" * 80)

        # =====================================================
        # Grundlegende Größe
        # =====================================================

        number_of_vertices = (
            len(mesh.vertices)
        )

        number_of_triangles = (
            len(mesh.triangles)
        )

        # =====================================================
        # Zusammenhängende Komponenten
        # =====================================================

        (
            triangle_clusters,
            cluster_n_triangles,
            cluster_area,
        ) = (
            mesh.cluster_connected_triangles()
        )

        cluster_n_triangles_array = (
            np.asarray(
                cluster_n_triangles,
                dtype=np.int64,
            )
        )

        connected_components = int(
            len(
                cluster_n_triangles_array
            )
        )

        if connected_components > 0:

            largest_component_triangles = int(
                np.max(
                    cluster_n_triangles_array
                )
            )

        else:

            largest_component_triangles = 0

        if number_of_triangles > 0:

            largest_component_ratio = float(
                largest_component_triangles
                / number_of_triangles
            )

        else:

            largest_component_ratio = 0.0

        # =====================================================
        # Topologische Eigenschaften
        # =====================================================

        edge_manifold_with_boundary = bool(
            mesh.is_edge_manifold(
                allow_boundary_edges=True
            )
        )

        edge_manifold_without_boundary = bool(
            mesh.is_edge_manifold(
                allow_boundary_edges=False
            )
        )

        vertex_manifold = bool(
            mesh.is_vertex_manifold()
        )

        self_intersecting = bool(
            mesh.is_self_intersecting()
        )

        watertight = bool(
            mesh.is_watertight()
        )

        orientable = bool(
            mesh.is_orientable()
        )

        # =====================================================
        # Mesh-Punkte für Distanzmessung samplen
        # =====================================================

        sample_count = (
            self.config
            .evaluation_sample_points
        )

        print()
        print(
            "Sample Mesh für geometrische Evaluation ..."
        )

        print(
            f"  Sample-Punkte: "
            f"{sample_count:,}"
        )

        sampled_mesh_cloud = (
            mesh.sample_points_uniformly(
                number_of_points=(
                    sample_count
                )
            )
        )

        # =====================================================
        # Punktwolke -> Mesh
        #
        # Open3D Legacy bietet hier keinen direkten
        # Punkt-zu-Dreiecks-Abstand.
        #
        # Deshalb approximieren wir das Mesh durch viele
        # gleichmäßig gesampelte Oberflächenpunkte.
        # =====================================================

        cloud_to_mesh_distances = (
            reference_cloud
            .compute_point_cloud_distance(
                sampled_mesh_cloud
            )
        )

        cloud_to_mesh_array = (
            np.asarray(
                cloud_to_mesh_distances,
                dtype=float,
            )
        )

        # =====================================================
        # Mesh -> Punktwolke
        # =====================================================

        mesh_to_cloud_distances = (
            sampled_mesh_cloud
            .compute_point_cloud_distance(
                reference_cloud
            )
        )

        mesh_to_cloud_array = (
            np.asarray(
                mesh_to_cloud_distances,
                dtype=float,
            )
        )

        # =====================================================
        # Distanzstatistiken
        # =====================================================

        (
            cloud_to_mesh_mean,
            cloud_to_mesh_median,
            cloud_to_mesh_rmse,
            cloud_to_mesh_p95,
            cloud_to_mesh_max,
        ) = (
            self._distance_statistics(
                cloud_to_mesh_array
            )
        )

        (
            mesh_to_cloud_mean,
            mesh_to_cloud_median,
            mesh_to_cloud_rmse,
            mesh_to_cloud_p95,
            mesh_to_cloud_max,
        ) = (
            self._distance_statistics(
                mesh_to_cloud_array
            )
        )

        symmetric_mean_distance = float(
            (
                cloud_to_mesh_mean
                + mesh_to_cloud_mean
            )
            / 2.0
        )

        symmetric_rmse = float(
            np.sqrt(
                (
                    cloud_to_mesh_rmse ** 2
                    + mesh_to_cloud_rmse ** 2
                )
                / 2.0
            )
        )

        # =====================================================
        # Ergebnis
        # =====================================================

        result = (
            MeshEvaluationResult(
                vertices=(
                    number_of_vertices
                ),

                triangles=(
                    number_of_triangles
                ),

                connected_components=(
                    connected_components
                ),

                largest_component_triangles=(
                    largest_component_triangles
                ),

                largest_component_ratio=(
                    largest_component_ratio
                ),

                edge_manifold_with_boundary=(
                    edge_manifold_with_boundary
                ),

                edge_manifold_without_boundary=(
                    edge_manifold_without_boundary
                ),

                vertex_manifold=(
                    vertex_manifold
                ),

                self_intersecting=(
                    self_intersecting
                ),

                watertight=(
                    watertight
                ),

                orientable=(
                    orientable
                ),

                cloud_to_mesh_mean=(
                    cloud_to_mesh_mean
                ),

                cloud_to_mesh_median=(
                    cloud_to_mesh_median
                ),

                cloud_to_mesh_rmse=(
                    cloud_to_mesh_rmse
                ),

                cloud_to_mesh_p95=(
                    cloud_to_mesh_p95
                ),

                cloud_to_mesh_max=(
                    cloud_to_mesh_max
                ),

                mesh_to_cloud_mean=(
                    mesh_to_cloud_mean
                ),

                mesh_to_cloud_median=(
                    mesh_to_cloud_median
                ),

                mesh_to_cloud_rmse=(
                    mesh_to_cloud_rmse
                ),

                mesh_to_cloud_p95=(
                    mesh_to_cloud_p95
                ),

                mesh_to_cloud_max=(
                    mesh_to_cloud_max
                ),

                symmetric_mean_distance=(
                    symmetric_mean_distance
                ),

                symmetric_rmse=(
                    symmetric_rmse
                ),
            )
        )

        self.print_result(
            result
        )

        return result

    # =========================================================
    # Distanzstatistik
    # =========================================================

    @staticmethod
    def _distance_statistics(
        distances: np.ndarray,
    ) -> tuple[
        float,
        float,
        float,
        float,
        float,
    ]:

        if distances.size == 0:
            raise ValueError(
                "Keine Distanzwerte vorhanden."
            )

        mean = float(
            np.mean(
                distances
            )
        )

        median = float(
            np.median(
                distances
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    distances ** 2
                )
            )
        )

        p95 = float(
            np.percentile(
                distances,
                95
            )
        )

        maximum = float(
            np.max(
                distances
            )
        )

        return (
            mean,
            median,
            rmse,
            p95,
            maximum,
        )

    # =========================================================
    # Ergebnis ausgeben
    # =========================================================

    @staticmethod
    def print_result(
        result: MeshEvaluationResult,
    ) -> None:

        print()
        print("-" * 80)
        print("MESH STATISTIK")
        print("-" * 80)

        print(
            f"Vertices: "
            f"{result.vertices:,}"
        )

        print(
            f"Dreiecke: "
            f"{result.triangles:,}"
        )

        print()
        print("Topologie:")

        print(
            f"  Connected Components: "
            f"{result.connected_components}"
        )

        print(
            f"  Größte Komponente: "
            f"{result.largest_component_triangles:,} "
            f"Dreiecke"
        )

        print(
            f"  Anteil größte Komponente: "
            f"{result.largest_component_ratio * 100:.2f} %"
        )

        print(
            f"  Edge Manifold "
            f"(Boundary erlaubt): "
            f"{result.edge_manifold_with_boundary}"
        )

        print(
            f"  Edge Manifold "
            f"(geschlossen): "
            f"{result.edge_manifold_without_boundary}"
        )

        print(
            f"  Vertex Manifold: "
            f"{result.vertex_manifold}"
        )

        print(
            f"  Self Intersecting: "
            f"{result.self_intersecting}"
        )

        print(
            f"  Watertight: "
            f"{result.watertight}"
        )

        print(
            f"  Orientable: "
            f"{result.orientable}"
        )

        print()
        print(
            "Punktwolke -> Mesh:"
        )

        print(
            f"  Mean:   "
            f"{result.cloud_to_mesh_mean:.6f}"
        )

        print(
            f"  Median: "
            f"{result.cloud_to_mesh_median:.6f}"
        )

        print(
            f"  RMSE:   "
            f"{result.cloud_to_mesh_rmse:.6f}"
        )

        print(
            f"  P95:    "
            f"{result.cloud_to_mesh_p95:.6f}"
        )

        print(
            f"  Max:    "
            f"{result.cloud_to_mesh_max:.6f}"
        )

        print()
        print(
            "Mesh -> Punktwolke:"
        )

        print(
            f"  Mean:   "
            f"{result.mesh_to_cloud_mean:.6f}"
        )

        print(
            f"  Median: "
            f"{result.mesh_to_cloud_median:.6f}"
        )

        print(
            f"  RMSE:   "
            f"{result.mesh_to_cloud_rmse:.6f}"
        )

        print(
            f"  P95:    "
            f"{result.mesh_to_cloud_p95:.6f}"
        )

        print(
            f"  Max:    "
            f"{result.mesh_to_cloud_max:.6f}"
        )

        print()
        print(
            "Symmetrische Distanz:"
        )

        print(
            f"  Mean: "
            f"{result.symmetric_mean_distance:.6f}"
        )

        print(
            f"  RMSE: "
            f"{result.symmetric_rmse:.6f}"
        )

    # =========================================================
    # Ergebnis als JSON speichern
    # =========================================================

    @staticmethod
    def save_json(
        result: MeshEvaluationResult,
        path: str | Path,
    ) -> None:

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                asdict(
                    result
                ),
                file,
                indent=4,
                ensure_ascii=False,
            )