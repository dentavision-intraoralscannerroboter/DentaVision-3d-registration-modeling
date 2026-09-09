from dataclasses import dataclass
import copy

import numpy as np
import open3d as o3d

from registration3d.reconstruction.mesh_config import (
    MeshConfig,
)


@dataclass
class MeshCleanupResult:

    mesh: o3d.geometry.TriangleMesh

    vertices_before: int
    vertices_after: int

    triangles_before: int
    triangles_after: int

    components_before: int
    components_after: int


class MeshCleaner:

    def __init__(
        self,
        config: MeshConfig,
    ) -> None:

        self.config = (
            config
        )

    # =========================================================
    # Basis-Cleanup
    # =========================================================

    def cleanup_basic(
        self,
        mesh: o3d.geometry.TriangleMesh,
    ) -> MeshCleanupResult:

        if len(mesh.vertices) == 0:

            raise ValueError(
                "Das Mesh enthält keine Vertices."
            )

        if len(mesh.triangles) == 0:

            raise ValueError(
                "Das Mesh enthält keine Dreiecke."
            )

        cleaned_mesh = (
            copy.deepcopy(
                mesh
            )
        )

        vertices_before = (
            len(
                cleaned_mesh.vertices
            )
        )

        triangles_before = (
            len(
                cleaned_mesh.triangles
            )
        )

        components_before = (
            self._count_components(
                cleaned_mesh
            )
        )

        print()
        print("=" * 80)
        print("MESH CLEANUP")
        print("=" * 80)

        print()
        print(
            "Vor Cleanup:"
        )

        print(
            f"  Vertices: "
            f"{vertices_before:,}"
        )

        print(
            f"  Dreiecke: "
            f"{triangles_before:,}"
        )

        print(
            f"  Komponenten: "
            f"{components_before}"
        )

        # -----------------------------------------------------
        # Doppelte Vertices
        # -----------------------------------------------------

        before = (
            len(
                cleaned_mesh.vertices
            )
        )

        cleaned_mesh.remove_duplicated_vertices()

        after = (
            len(
                cleaned_mesh.vertices
            )
        )

        print()
        print(
            "Doppelte Vertices entfernt:"
        )

        print(
            f"  {before - after:,}"
        )

        # -----------------------------------------------------
        # Doppelte Dreiecke
        # -----------------------------------------------------

        before = (
            len(
                cleaned_mesh.triangles
            )
        )

        cleaned_mesh.remove_duplicated_triangles()

        after = (
            len(
                cleaned_mesh.triangles
            )
        )

        print(
            "Doppelte Dreiecke entfernt:"
        )

        print(
            f"  {before - after:,}"
        )

        # -----------------------------------------------------
        # Degenerierte Dreiecke
        # -----------------------------------------------------

        before = (
            len(
                cleaned_mesh.triangles
            )
        )

        cleaned_mesh.remove_degenerate_triangles()

        after = (
            len(
                cleaned_mesh.triangles
            )
        )

        print(
            "Degenerierte Dreiecke entfernt:"
        )

        print(
            f"  {before - after:,}"
        )

        # -----------------------------------------------------
        # Unreferenzierte Vertices
        # -----------------------------------------------------

        before = (
            len(
                cleaned_mesh.vertices
            )
        )

        cleaned_mesh.remove_unreferenced_vertices()

        after = (
            len(
                cleaned_mesh.vertices
            )
        )

        print(
            "Unreferenzierte Vertices entfernt:"
        )

        print(
            f"  {before - after:,}"
        )

        # -----------------------------------------------------
        # Normalen neu berechnen
        # -----------------------------------------------------

        cleaned_mesh.compute_triangle_normals()

        cleaned_mesh.compute_vertex_normals()

        # -----------------------------------------------------
        # Ergebnis
        # -----------------------------------------------------

        vertices_after = (
            len(
                cleaned_mesh.vertices
            )
        )

        triangles_after = (
            len(
                cleaned_mesh.triangles
            )
        )

        components_after = (
            self._count_components(
                cleaned_mesh
            )
        )

        print()
        print(
            "Nach Cleanup:"
        )

        print(
            f"  Vertices: "
            f"{vertices_after:,}"
        )

        print(
            f"  Dreiecke: "
            f"{triangles_after:,}"
        )

        print(
            f"  Komponenten: "
            f"{components_after}"
        )

        return (
            MeshCleanupResult(
                mesh=(
                    cleaned_mesh
                ),

                vertices_before=(
                    vertices_before
                ),

                vertices_after=(
                    vertices_after
                ),

                triangles_before=(
                    triangles_before
                ),

                triangles_after=(
                    triangles_after
                ),

                components_before=(
                    components_before
                ),

                components_after=(
                    components_after
                ),
            )
        )

    # =========================================================
    # Kleine Komponenten entfernen
    # =========================================================

    def remove_small_components(
        self,
        mesh: o3d.geometry.TriangleMesh,
        minimum_triangles: int | None = None,
    ) -> o3d.geometry.TriangleMesh:

        if len(mesh.vertices) == 0:

            raise ValueError(
                "Das Mesh enthält keine Vertices."
            )

        if len(mesh.triangles) == 0:

            raise ValueError(
                "Das Mesh enthält keine Dreiecke."
            )

        if minimum_triangles is None:

            minimum_triangles = (
                self.config
                .minimum_component_triangles
            )

        if minimum_triangles <= 0:

            raise ValueError(
                "minimum_triangles muss "
                "größer als 0 sein."
            )

        cleaned_mesh = (
            copy.deepcopy(
                mesh
            )
        )

        (
            triangle_clusters,
            cluster_n_triangles,
            _,
        ) = (
            cleaned_mesh
            .cluster_connected_triangles()
        )

        triangle_clusters = (
            np.asarray(
                triangle_clusters,
                dtype=np.int64,
            )
        )

        cluster_n_triangles = (
            np.asarray(
                cluster_n_triangles,
                dtype=np.int64,
            )
        )

        if (
            triangle_clusters.size == 0
            or cluster_n_triangles.size == 0
        ):

            return (
                cleaned_mesh
            )

        components_before = (
            len(
                cluster_n_triangles
            )
        )

        # -----------------------------------------------------
        # Kleine Komponenten identifizieren
        # -----------------------------------------------------

        small_component_ids = (
            np.where(
                cluster_n_triangles
                < minimum_triangles
            )[0]
        )

        triangle_mask = (
            np.isin(
                triangle_clusters,
                small_component_ids,
            )
        )

        triangles_to_remove = int(
            np.count_nonzero(
                triangle_mask
            )
        )

        removed_components = (
            len(
                small_component_ids
            )
        )

        print()
        print("=" * 80)
        print(
            "SMALL COMPONENT CLEANUP"
        )
        print("=" * 80)

        print()
        print(
            f"Minimum Dreiecke pro Komponente: "
            f"{minimum_triangles}"
        )

        print(
            f"Komponenten vorher: "
            f"{components_before}"
        )

        print(
            f"Zu entfernende Komponenten: "
            f"{removed_components}"
        )

        print(
            f"Zu entfernende Dreiecke: "
            f"{triangles_to_remove:,}"
        )

        # -----------------------------------------------------
        # Entfernen
        # -----------------------------------------------------

        cleaned_mesh.remove_triangles_by_mask(
            triangle_mask
        )

        cleaned_mesh.remove_unreferenced_vertices()

        # -----------------------------------------------------
        # Normalen neu berechnen
        # -----------------------------------------------------

        cleaned_mesh.compute_triangle_normals()

        cleaned_mesh.compute_vertex_normals()

        components_after = (
            self._count_components(
                cleaned_mesh
            )
        )

        print()
        print(
            "Nach Component Cleanup:"
        )

        print(
            f"  Vertices: "
            f"{len(cleaned_mesh.vertices):,}"
        )

        print(
            f"  Dreiecke: "
            f"{len(cleaned_mesh.triangles):,}"
        )

        print(
            f"  Komponenten: "
            f"{components_after}"
        )

        return (
            cleaned_mesh
        )

    # =========================================================
    # Anzahl Connected Components
    # =========================================================

    @staticmethod
    def _count_components(
        mesh: o3d.geometry.TriangleMesh,
    ) -> int:

        if len(mesh.triangles) == 0:

            return 0

        (
            _,
            cluster_n_triangles,
            _,
        ) = (
            mesh.cluster_connected_triangles()
        )

        return (
            len(
                cluster_n_triangles
            )
        )