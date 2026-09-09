from pathlib import Path
import copy
import json

import numpy as np
import open3d as o3d


def main() -> None:

    # =========================================================
    # Projektpfade
    # =========================================================

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    input_path = (
        project_root
        / "output"
        / "mesh"
        / "ball_pivoting"
        / "final_candidate_mesh.ply"
    )

    output_directory = (
        project_root
        / "output"
        / "mesh"
        / "ball_pivoting"
        / "topology_analysis"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # Input prüfen
    # =========================================================

    if not input_path.exists():

        raise FileNotFoundError(
            "Finales BPA-Kandidaten-Mesh "
            "wurde nicht gefunden:\n"
            f"{input_path}"
        )

    # =========================================================
    # Mesh laden
    # =========================================================

    print()
    print("=" * 80)
    print("MESH TOPOLOGY ANALYSIS")
    print("=" * 80)

    print()
    print(
        f"Eingabe: "
        f"{input_path}"
    )

    mesh = (
        o3d.io.read_triangle_mesh(
            str(input_path)
        )
    )

    if len(mesh.vertices) == 0:
        raise ValueError(
            "Das Mesh enthält keine Vertices."
        )

    if len(mesh.triangles) == 0:
        raise ValueError(
            "Das Mesh enthält keine Dreiecke."
        )

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    print()
    print(
        f"Vertices: "
        f"{len(mesh.vertices):,}"
    )

    print(
        f"Dreiecke: "
        f"{len(mesh.triangles):,}"
    )

    # =========================================================
    # Globale Topologie
    # =========================================================

    edge_manifold_boundary_allowed = bool(
        mesh.is_edge_manifold(
            allow_boundary_edges=True
        )
    )

    edge_manifold_closed = bool(
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

    # =========================================================
    # Problematische Kanten
    #
    # allow_boundary_edges=True:
    # nur echte Non-Manifold-Kanten
    #
    # allow_boundary_edges=False:
    # auch offene Randkanten
    # =========================================================

    non_manifold_edges = np.asarray(
        mesh.get_non_manifold_edges(
            allow_boundary_edges=True
        ),
        dtype=np.int64,
    )

    non_closed_edges = np.asarray(
        mesh.get_non_manifold_edges(
            allow_boundary_edges=False
        ),
        dtype=np.int64,
    )

    # =========================================================
    # Boundary Edges bestimmen
    #
    # non_closed_edges enthält:
    # - Boundary Edges
    # - echte Non-Manifold Edges
    #
    # Da unser BPA-Mesh bereits Edge-Manifold mit erlaubten
    # Rändern ist, erwarten wir hier hauptsächlich Boundary.
    # =========================================================

    non_manifold_edge_set = {
        tuple(
            sorted(
                (
                    int(edge[0]),
                    int(edge[1]),
                )
            )
        )
        for edge in non_manifold_edges
    }

    boundary_edges = []

    for edge in non_closed_edges:

        edge_tuple = tuple(
            sorted(
                (
                    int(edge[0]),
                    int(edge[1]),
                )
            )
        )

        if (
            edge_tuple
            not in non_manifold_edge_set
        ):
            boundary_edges.append(
                edge_tuple
            )

    # =========================================================
    # Non-Manifold Vertices
    # =========================================================

    non_manifold_vertices = np.asarray(
        mesh.get_non_manifold_vertices(),
        dtype=np.int64,
    )

    # =========================================================
    # Self-Intersection-Paare
    # =========================================================

    self_intersection_pairs = np.asarray(
        mesh.get_self_intersecting_triangles(),
        dtype=np.int64,
    )

    # =========================================================
    # Versuch, Dreiecksorientierung zu korrigieren
    # =========================================================

    orientation_test_mesh = (
        copy.deepcopy(
            mesh
        )
    )

    orient_triangles_success = bool(
        orientation_test_mesh
        .orient_triangles()
    )

    orientable_after_attempt = bool(
        orientation_test_mesh
        .is_orientable()
    )

    # =========================================================
    # Ausgabe
    # =========================================================

    print()
    print("-" * 80)
    print("TOPOLOGIE")
    print("-" * 80)

    print()
    print(
        f"Edge Manifold "
        f"(Boundary erlaubt): "
        f"{edge_manifold_boundary_allowed}"
    )

    print(
        f"Edge Manifold "
        f"(geschlossen): "
        f"{edge_manifold_closed}"
    )

    print(
        f"Vertex Manifold: "
        f"{vertex_manifold}"
    )

    print(
        f"Self Intersecting: "
        f"{self_intersecting}"
    )

    print(
        f"Watertight: "
        f"{watertight}"
    )

    print(
        f"Orientable: "
        f"{orientable}"
    )

    print()
    print("-" * 80)
    print("PROBLEMSTELLEN")
    print("-" * 80)

    print()
    print(
        f"Echte Non-Manifold Edges: "
        f"{len(non_manifold_edges):,}"
    )

    print(
        f"Boundary Edges: "
        f"{len(boundary_edges):,}"
    )

    print(
        f"Non-Manifold Vertices: "
        f"{len(non_manifold_vertices):,}"
    )

    print(
        f"Self-Intersection-Paare: "
        f"{len(self_intersection_pairs):,}"
    )

    print()
    print(
        f"orient_triangles() erfolgreich: "
        f"{orient_triangles_success}"
    )

    print(
        f"Danach orientierbar: "
        f"{orientable_after_attempt}"
    )

    # =========================================================
    # Prozentualer Anteil problematischer Vertices
    # =========================================================

    non_manifold_vertex_ratio = (
        len(non_manifold_vertices)
        / len(mesh.vertices)
    )

    print()
    print(
        f"Anteil Non-Manifold Vertices: "
        f"{non_manifold_vertex_ratio * 100:.4f} %"
    )

    # =========================================================
    # JSON speichern
    # =========================================================

    result = {
        "vertices": (
            len(mesh.vertices)
        ),

        "triangles": (
            len(mesh.triangles)
        ),

        "edge_manifold_boundary_allowed": (
            edge_manifold_boundary_allowed
        ),

        "edge_manifold_closed": (
            edge_manifold_closed
        ),

        "vertex_manifold": (
            vertex_manifold
        ),

        "self_intersecting": (
            self_intersecting
        ),

        "watertight": (
            watertight
        ),

        "orientable": (
            orientable
        ),

        "non_manifold_edges": (
            len(non_manifold_edges)
        ),

        "boundary_edges": (
            len(boundary_edges)
        ),

        "non_manifold_vertices": (
            len(non_manifold_vertices)
        ),

        "non_manifold_vertex_ratio": (
            non_manifold_vertex_ratio
        ),

        "self_intersection_pairs": (
            len(self_intersection_pairs)
        ),

        "orient_triangles_success": (
            orient_triangles_success
        ),

        "orientable_after_attempt": (
            orientable_after_attempt
        ),
    }

    json_path = (
        output_directory
        / "topology_analysis.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print(
        "Topologie-Auswertung gespeichert:"
    )

    print(
        f"  {json_path}"
    )

    # =========================================================
    # Problematische Vertices separat speichern
    # =========================================================

    if len(non_manifold_vertices) > 0:

        vertices = np.asarray(
            mesh.vertices
        )

        problem_cloud = (
            o3d.geometry.PointCloud()
        )

        problem_cloud.points = (
            o3d.utility.Vector3dVector(
                vertices[
                    non_manifold_vertices
                ]
            )
        )

        problem_cloud.paint_uniform_color(
            [
                1.0,
                0.0,
                0.0,
            ]
        )

        problem_vertices_path = (
            output_directory
            / "non_manifold_vertices.ply"
        )

        o3d.io.write_point_cloud(
            str(
                problem_vertices_path
            ),
            problem_cloud,
        )

        print()
        print(
            "Problematische Vertices gespeichert:"
        )

        print(
            f"  {problem_vertices_path}"
        )

    # =========================================================
    # Orientierte Testversion speichern,
    # falls Open3D die Orientierung korrigieren konnte.
    # =========================================================

    if orient_triangles_success:

        orientation_test_mesh.compute_triangle_normals()
        orientation_test_mesh.compute_vertex_normals()

        oriented_mesh_path = (
            output_directory
            / "oriented_test_mesh.ply"
        )

        o3d.io.write_triangle_mesh(
            str(
                oriented_mesh_path
            ),
            orientation_test_mesh,
        )

        print()
        print(
            "Orientierte Testversion gespeichert:"
        )

        print(
            f"  {oriented_mesh_path}"
        )

    # =========================================================
    # Abschluss
    # =========================================================

    print()
    print("=" * 80)
    print("TOPOLOGIE-ANALYSE ABGESCHLOSSEN")
    print("=" * 80)


if __name__ == "__main__":
    main()