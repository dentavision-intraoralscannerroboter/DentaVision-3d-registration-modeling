from pathlib import Path
import copy

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

    output_path = (
        output_directory
        / "topology_colored_mesh.ply"
    )

    # =========================================================
    # Mesh laden
    # =========================================================

    if not input_path.exists():

        raise FileNotFoundError(
            "Finaler BPA-Kandidat nicht gefunden:\n"
            f"{input_path}"
        )

    print()
    print("=" * 80)
    print("TOPOLOGY VISUALIZATION")
    print("=" * 80)

    print()
    print(
        f"Eingabe: {input_path}"
    )

    mesh = (
        o3d.io.read_triangle_mesh(
            str(input_path)
        )
    )

    if len(mesh.vertices) == 0:

        raise ValueError(
            "Mesh enthält keine Vertices."
        )

    if len(mesh.triangles) == 0:

        raise ValueError(
            "Mesh enthält keine Dreiecke."
        )

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    print()
    print(
        f"Vertices: {len(mesh.vertices):,}"
    )

    print(
        f"Dreiecke: {len(mesh.triangles):,}"
    )

    # =========================================================
    # Non-Manifold Vertices bestimmen
    # =========================================================

    non_manifold_vertices = np.asarray(
        mesh.get_non_manifold_vertices(),
        dtype=np.int64,
    )

    print()
    print(
        "Non-Manifold Vertices: "
        f"{len(non_manifold_vertices):,}"
    )

    if len(non_manifold_vertices) == 0:

        print()
        print(
            "Keine Non-Manifold Vertices gefunden."
        )

        return

    ratio = (
        len(non_manifold_vertices)
        / len(mesh.vertices)
    )

    print(
        "Anteil: "
        f"{ratio * 100:.4f} %"
    )

    # =========================================================
    # Farbkopie erzeugen
    #
    # Normale Vertices:
    #   hellgrau
    #
    # Non-Manifold Vertices:
    #   rot
    # =========================================================

    colored_mesh = (
        copy.deepcopy(
            mesh
        )
    )

    number_of_vertices = (
        len(colored_mesh.vertices)
    )

    colors = np.full(
        (
            number_of_vertices,
            3,
        ),
        [
            0.72,
            0.72,
            0.72,
        ],
        dtype=np.float64,
    )

    colors[
        non_manifold_vertices
    ] = [
        1.0,
        0.0,
        0.0,
    ]

    colored_mesh.vertex_colors = (
        o3d.utility.Vector3dVector(
            colors
        )
    )

    # =========================================================
    # Problemvertices zusätzlich als rote Punktwolke
    #
    # Dadurch sind sie auch dann deutlich sichtbar,
    # wenn Vertexfarben im Renderer schlecht erkennbar sind.
    # =========================================================

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

    # =========================================================
    # Boundary Edges bestimmen
    # =========================================================

    boundary_edges = np.asarray(
        mesh.get_non_manifold_edges(
            allow_boundary_edges=False
        ),
        dtype=np.int64,
    )

    true_non_manifold_edges = np.asarray(
        mesh.get_non_manifold_edges(
            allow_boundary_edges=True
        ),
        dtype=np.int64,
    )

    true_non_manifold_edge_set = {
        tuple(
            sorted(
                (
                    int(edge[0]),
                    int(edge[1]),
                )
            )
        )
        for edge
        in true_non_manifold_edges
    }

    pure_boundary_edges = []

    for edge in boundary_edges:

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
            not in true_non_manifold_edge_set
        ):

            pure_boundary_edges.append(
                edge_tuple
            )

    print(
        "Boundary Edges: "
        f"{len(pure_boundary_edges):,}"
    )

    print(
        "Echte Non-Manifold Edges: "
        f"{len(true_non_manifold_edges):,}"
    )

    # =========================================================
    # Boundary Edges als blaue Linien darstellen
    # =========================================================

    boundary_line_set = (
        o3d.geometry.LineSet()
    )

    boundary_line_set.points = (
        o3d.utility.Vector3dVector(
            vertices
        )
    )

    if len(pure_boundary_edges) > 0:

        boundary_line_set.lines = (
            o3d.utility.Vector2iVector(
                np.asarray(
                    pure_boundary_edges,
                    dtype=np.int32,
                )
            )
        )

        boundary_colors = np.tile(
            [
                0.0,
                0.3,
                1.0,
            ],
            (
                len(pure_boundary_edges),
                1,
            ),
        )

        boundary_line_set.colors = (
            o3d.utility.Vector3dVector(
                boundary_colors
            )
        )

    # =========================================================
    # Eingefärbtes Mesh speichern
    # =========================================================

    success = (
        o3d.io.write_triangle_mesh(
            str(output_path),
            colored_mesh,
        )
    )

    if not success:

        raise RuntimeError(
            "Eingefärbtes Mesh konnte "
            "nicht gespeichert werden."
        )

    print()
    print(
        "Eingefärbtes Mesh gespeichert:"
    )

    print(
        f"  {output_path}"
    )

    # =========================================================
    # Legende
    # =========================================================

    print()
    print("-" * 80)
    print("LEGENDE")
    print("-" * 80)

    print()
    print(
        "Grau  = reguläre Mesh-Oberfläche"
    )

    print(
        "Rot   = Non-Manifold Vertices"
    )

    print(
        "Blau  = offene Boundary Edges"
    )

    # =========================================================
    # Visualisierung
    # =========================================================

    print()
    print(
        "Öffne Topologie-Visualisierung ..."
    )

    visualizer = (
        o3d.visualization.Visualizer()
    )

    visualizer.create_window(
        window_name=(
            "BPA Mesh - Topology Problems"
        ),
        width=1400,
        height=900,
    )

    visualizer.add_geometry(
        colored_mesh
    )

    visualizer.add_geometry(
        problem_cloud
    )

    if len(pure_boundary_edges) > 0:

        visualizer.add_geometry(
            boundary_line_set
        )

    render_option = (
        visualizer.get_render_option()
    )

    if render_option is not None:

        render_option.point_size = 8.0

        render_option.line_width = 2.0

        render_option.mesh_show_back_face = True

        render_option.background_color = (
            np.asarray(
                [
                    1.0,
                    1.0,
                    1.0,
                ]
            )
        )

    visualizer.run()

    visualizer.destroy_window()


if __name__ == "__main__":
    main()