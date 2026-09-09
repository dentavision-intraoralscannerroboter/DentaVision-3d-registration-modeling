from pathlib import Path
import copy

import numpy as np
import open3d as o3d

from registration3d.data_loader import (
    PointCloudLoader,
)
from registration3d.reconstruction.mesh_config import (
    MeshConfig,
)
from registration3d.reconstruction.mesh_evaluator import (
    MeshEvaluator,
)


class DisjointSet:

    def __init__(
        self,
        elements: list[int],
    ) -> None:

        self.parent = {
            element: element
            for element in elements
        }

    def find(
        self,
        element: int,
    ) -> int:

        parent = self.parent[element]

        if parent != element:

            self.parent[element] = (
                self.find(
                    parent
                )
            )

        return self.parent[element]

    def union(
        self,
        a: int,
        b: int,
    ) -> None:

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:

            self.parent[root_b] = root_a


class MeshTopologyRepairer:

    # =========================================================
    # Non-Manifold Vertices durch Vertex Splitting reparieren
    # =========================================================

    def split_non_manifold_vertices(
        self,
        mesh: o3d.geometry.TriangleMesh,
    ) -> o3d.geometry.TriangleMesh:

        if len(mesh.vertices) == 0:

            raise ValueError(
                "Mesh enthält keine Vertices."
            )

        if len(mesh.triangles) == 0:

            raise ValueError(
                "Mesh enthält keine Dreiecke."
            )

        print()
        print("=" * 80)
        print("NON-MANIFOLD VERTEX REPAIR")
        print("=" * 80)

        original_vertices = np.asarray(
            mesh.vertices,
            dtype=np.float64,
        )

        triangles = np.asarray(
            mesh.triangles,
            dtype=np.int64,
        ).copy()

        original_vertex_count = (
            len(original_vertices)
        )

        # -----------------------------------------------------
        # Aktuelle Vertexliste
        # -----------------------------------------------------

        new_vertices = [
            vertex.copy()
            for vertex
            in original_vertices
        ]

        # -----------------------------------------------------
        # Incident Triangles für jeden ursprünglichen Vertex
        # -----------------------------------------------------

        incident_triangles: list[list[int]] = [
            []
            for _
            in range(
                original_vertex_count
            )
        ]

        for triangle_id, triangle in enumerate(
            triangles
        ):

            for vertex_id in triangle:

                vertex_id = int(
                    vertex_id
                )

                if (
                    vertex_id
                    < original_vertex_count
                ):

                    incident_triangles[
                        vertex_id
                    ].append(
                        triangle_id
                    )

        # -----------------------------------------------------
        # Non-Manifold Vertices bestimmen
        # -----------------------------------------------------

        non_manifold_vertices = np.asarray(
            mesh.get_non_manifold_vertices(),
            dtype=np.int64,
        )

        print()
        print(
            "Non-Manifold Vertices vorher: "
            f"{len(non_manifold_vertices):,}"
        )

        split_vertex_count = 0
        duplicated_vertex_count = 0

        # =====================================================
        # Jeden problematischen Vertex analysieren
        # =====================================================

        for vertex_id in non_manifold_vertices:

            vertex_id = int(
                vertex_id
            )

            incident = (
                incident_triangles[
                    vertex_id
                ]
            )

            if len(incident) < 2:
                continue

            # -------------------------------------------------
            # Für jedes Dreieck bestimmen wir die beiden
            # Nachbarvertices neben vertex_id.
            #
            # Zwei Dreiecke gehören zum selben Triangle-Fan,
            # wenn sie eine Kante teilen, die vertex_id enthält.
            # -------------------------------------------------

            edge_to_triangles: dict[
                int,
                list[int],
            ] = {}

            for triangle_id in incident:

                triangle = (
                    triangles[
                        triangle_id
                    ]
                )

                for other_vertex_id in triangle:

                    other_vertex_id = int(
                        other_vertex_id
                    )

                    if (
                        other_vertex_id
                        == vertex_id
                    ):
                        continue

                    edge_to_triangles.setdefault(
                        other_vertex_id,
                        [],
                    ).append(
                        triangle_id
                    )

            # -------------------------------------------------
            # Zusammenhängende Triangle-Fans bestimmen
            # -------------------------------------------------

            disjoint_set = (
                DisjointSet(
                    incident
                )
            )

            for triangle_ids in (
                edge_to_triangles.values()
            ):

                if len(triangle_ids) < 2:
                    continue

                first = (
                    triangle_ids[0]
                )

                for other in (
                    triangle_ids[1:]
                ):

                    disjoint_set.union(
                        first,
                        other,
                    )

            components: dict[
                int,
                list[int],
            ] = {}

            for triangle_id in incident:

                root = (
                    disjoint_set.find(
                        triangle_id
                    )
                )

                components.setdefault(
                    root,
                    [],
                ).append(
                    triangle_id
                )

            triangle_fans = list(
                components.values()
            )

            # -------------------------------------------------
            # Ein Triangle-Fan = manifold an diesem Vertex.
            # -------------------------------------------------

            if len(triangle_fans) <= 1:
                continue

            # -------------------------------------------------
            # Größten Fan am ursprünglichen Vertex lassen.
            # Kleine/weitere Fans bekommen je ein Duplikat.
            # -------------------------------------------------

            triangle_fans.sort(
                key=len,
                reverse=True,
            )

            split_vertex_count += 1

            for fan in triangle_fans[1:]:

                new_vertex_id = (
                    len(
                        new_vertices
                    )
                )

                new_vertices.append(
                    original_vertices[
                        vertex_id
                    ].copy()
                )

                duplicated_vertex_count += 1

                # ---------------------------------------------
                # In diesem Fan den ursprünglichen Vertex
                # durch das geometrisch identische Duplikat
                # ersetzen.
                # ---------------------------------------------

                for triangle_id in fan:

                    triangle = (
                        triangles[
                            triangle_id
                        ]
                    )

                    triangle[
                        triangle
                        == vertex_id
                    ] = (
                        new_vertex_id
                    )

        # =====================================================
        # Neues Mesh erzeugen
        # =====================================================

        repaired_mesh = (
            o3d.geometry.TriangleMesh()
        )

        repaired_mesh.vertices = (
            o3d.utility.Vector3dVector(
                np.asarray(
                    new_vertices,
                    dtype=np.float64,
                )
            )
        )

        repaired_mesh.triangles = (
            o3d.utility.Vector3iVector(
                triangles.astype(
                    np.int32
                )
            )
        )

        repaired_mesh.compute_triangle_normals()
        repaired_mesh.compute_vertex_normals()

        # =====================================================
        # Ergebnis
        # =====================================================

        remaining_non_manifold_vertices = (
            np.asarray(
                repaired_mesh
                .get_non_manifold_vertices(),
                dtype=np.int64,
            )
        )

        print()
        print(
            "Ursprüngliche Vertices: "
            f"{original_vertex_count:,}"
        )

        print(
            "Aufgespaltene Problemvertices: "
            f"{split_vertex_count:,}"
        )

        print(
            "Neu erzeugte Vertex-Duplikate: "
            f"{duplicated_vertex_count:,}"
        )

        print(
            "Vertices nach Reparatur: "
            f"{len(repaired_mesh.vertices):,}"
        )

        print()
        print(
            "Non-Manifold Vertices nach Reparatur: "
            f"{len(remaining_non_manifold_vertices):,}"
        )

        return repaired_mesh


def print_topology(
    name: str,
    mesh: o3d.geometry.TriangleMesh,
) -> None:

    print()
    print("-" * 80)
    print(name)
    print("-" * 80)

    non_manifold_edges = np.asarray(
        mesh.get_non_manifold_edges(
            allow_boundary_edges=True
        )
    )

    non_closed_edges = np.asarray(
        mesh.get_non_manifold_edges(
            allow_boundary_edges=False
        )
    )

    non_manifold_vertices = np.asarray(
        mesh.get_non_manifold_vertices()
    )

    print()
    print(
        f"Vertices: "
        f"{len(mesh.vertices):,}"
    )

    print(
        f"Dreiecke: "
        f"{len(mesh.triangles):,}"
    )

    print()
    print(
        "Edge Manifold "
        "(Boundary erlaubt): "
        f"{mesh.is_edge_manifold(True)}"
    )

    print(
        "Edge Manifold "
        "(geschlossen): "
        f"{mesh.is_edge_manifold(False)}"
    )

    print(
        f"Vertex Manifold: "
        f"{mesh.is_vertex_manifold()}"
    )

    print(
        f"Orientable: "
        f"{mesh.is_orientable()}"
    )

    print(
        f"Watertight: "
        f"{mesh.is_watertight()}"
    )

    print(
        f"Self Intersecting: "
        f"{mesh.is_self_intersecting()}"
    )

    print()
    print(
        "Echte Non-Manifold Edges: "
        f"{len(non_manifold_edges):,}"
    )

    print(
        "Boundary / Non-Closed Edges: "
        f"{len(non_closed_edges):,}"
    )

    print(
        "Non-Manifold Vertices: "
        f"{len(non_manifold_vertices):,}"
    )


def main() -> None:

    # =========================================================
    # Projektpfade
    # =========================================================

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    mesh_path = (
        project_root
        / "output"
        / "mesh"
        / "ball_pivoting"
        / "final_candidate_mesh.ply"
    )

    reference_cloud_path = (
        project_root
        / "output"
        / "multiway"
        / "fgr_baseline"
        / "combined_cloud.ply"
    )

    output_directory = (
        project_root
        / "output"
        / "mesh"
        / "ball_pivoting"
        / "topology_repair"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # Dateien prüfen
    # =========================================================

    if not mesh_path.exists():

        raise FileNotFoundError(
            "Finales BPA-Mesh fehlt:\n"
            f"{mesh_path}"
        )

    if not reference_cloud_path.exists():

        raise FileNotFoundError(
            "Referenzpunktwolke fehlt:\n"
            f"{reference_cloud_path}"
        )

    # =========================================================
    # Mesh laden
    # =========================================================

    print()
    print("=" * 80)
    print("MESH TOPOLOGY REPAIR")
    print("=" * 80)

    print()
    print(
        f"Mesh: {mesh_path}"
    )

    mesh = (
        o3d.io.read_triangle_mesh(
            str(mesh_path)
        )
    )

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    # =========================================================
    # Topologie vorher
    # =========================================================

    print_topology(
        "TOPOLOGIE VOR REPARATUR",
        mesh,
    )

    # =========================================================
    # Vertex Splitting
    # =========================================================

    repairer = (
        MeshTopologyRepairer()
    )

    repaired_mesh = (
        repairer
        .split_non_manifold_vertices(
            mesh
        )
    )

    # =========================================================
    # Topologie nach Vertex Splitting
    # =========================================================

    print_topology(
        "TOPOLOGIE NACH VERTEX SPLITTING",
        repaired_mesh,
    )

    # =========================================================
    # Repariertes Mesh speichern
    # =========================================================

    repaired_path = (
        output_directory
        / "vertex_split_mesh.ply"
    )

    success = (
        o3d.io.write_triangle_mesh(
            str(repaired_path),
            repaired_mesh,
        )
    )

    if not success:

        raise RuntimeError(
            "Repariertes Mesh konnte "
            "nicht gespeichert werden."
        )

    print()
    print(
        "Repariertes Mesh gespeichert:"
    )

    print(
        f"  {repaired_path}"
    )

    # =========================================================
    # Orientierung versuchen
    # =========================================================

    final_mesh = (
        copy.deepcopy(
            repaired_mesh
        )
    )

    print()
    print("=" * 80)
    print("ORIENTATION TEST")
    print("=" * 80)

    if final_mesh.is_orientable():

        print()
        print(
            "Mesh ist nach Vertex Splitting orientierbar."
        )

        orientation_success = bool(
            final_mesh.orient_triangles()
        )

        print(
            "orient_triangles() erfolgreich: "
            f"{orientation_success}"
        )

    else:

        print()
        print(
            "Mesh ist auch nach Vertex Splitting "
            "noch nicht orientierbar."
        )

        orientation_success = False

    final_mesh.compute_triangle_normals()
    final_mesh.compute_vertex_normals()

    print_topology(
        "FINALE TOPOLOGIE",
        final_mesh,
    )

    # =========================================================
    # Finale Reparaturversion speichern
    # =========================================================

    final_path = (
        output_directory
        / "topology_repaired_mesh.ply"
    )

    success = (
        o3d.io.write_triangle_mesh(
            str(final_path),
            final_mesh,
        )
    )

    if not success:

        raise RuntimeError(
            "Finale Reparaturversion konnte "
            "nicht gespeichert werden."
        )

    print()
    print(
        "Finale Reparaturversion gespeichert:"
    )

    print(
        f"  {final_path}"
    )

    # =========================================================
    # Geometrische Evaluation
    #
    # Da Vertex Splitting keine Koordinaten verschiebt,
    # erwarten wir praktisch unveränderte Distanzen.
    # =========================================================

    reference_cloud = (
        PointCloudLoader.load(
            reference_cloud_path
        )
    )

    config = (
        MeshConfig(
            normal_radius_factor=3.0,
            normal_max_nn=50,

            poisson_depth=9,
            poisson_density_quantile=0.02,

            ball_pivoting_radius_factors=(
                1.0,
                1.5,
                2.0,
            ),

            minimum_component_triangles=250,

            target_triangle_count=100_000,

            evaluation_sample_points=100_000,
        )
    )

    evaluator = (
        MeshEvaluator(
            config
        )
    )

    # Für reproduzierbarere Mesh-Samples
    o3d.utility.random.seed(
        42
    )

    evaluation = (
        evaluator.evaluate(
            mesh=final_mesh,
            reference_cloud=reference_cloud,
        )
    )

    evaluation_path = (
        output_directory
        / "topology_repair_evaluation.json"
    )

    evaluator.save_json(
        result=evaluation,
        path=evaluation_path,
    )

    # =========================================================
    # Abschluss
    # =========================================================

    print()
    print("=" * 80)
    print("TOPOLOGY REPAIR ABGESCHLOSSEN")
    print("=" * 80)

    print()
    print(
        f"Vertex Manifold: "
        f"{final_mesh.is_vertex_manifold()}"
    )

    print(
        f"Orientable: "
        f"{final_mesh.is_orientable()}"
    )

    print(
        f"Self Intersecting: "
        f"{final_mesh.is_self_intersecting()}"
    )

    print(
        f"Symmetric RMSE: "
        f"{evaluation.symmetric_rmse:.6f}"
    )

    print()
    print(
        "Öffne repariertes Mesh ..."
    )

    o3d.visualization.draw_geometries(
        [
            final_mesh
        ],
        window_name=(
            "Topology Repaired BPA Mesh"
        ),
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()