from collections import defaultdict, deque
from pathlib import Path
import json

import numpy as np
import open3d as o3d


def build_edge_map(
    triangles: np.ndarray,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """
    Erzeugt für jede ungerichtete Mesh-Kante eine Liste der
    angrenzenden Dreiecke.

    Gespeichert wird zusätzlich die Orientierung der Kante
    innerhalb des jeweiligen Dreiecks:

        +1: kleinere Vertex-ID -> größere Vertex-ID
        -1: größere Vertex-ID -> kleinere Vertex-ID
    """

    edge_map = defaultdict(list)

    for triangle_id, triangle in enumerate(triangles):

        a = int(triangle[0])
        b = int(triangle[1])
        c = int(triangle[2])

        directed_edges = (
            (a, b),
            (b, c),
            (c, a),
        )

        for start, end in directed_edges:

            edge = (
                min(start, end),
                max(start, end),
            )

            direction = (
                1
                if start < end
                else -1
            )

            edge_map[edge].append(
                (
                    triangle_id,
                    direction,
                )
            )

    return dict(edge_map)


def analyze_orientability(
    mesh: o3d.geometry.TriangleMesh,
) -> dict:

    triangles = np.asarray(
        mesh.triangles,
        dtype=np.int64,
    )

    number_of_triangles = len(
        triangles
    )

    edge_map = build_edge_map(
        triangles
    )

    # =========================================================
    # Dualen Triangle-Graph aufbauen
    #
    # Zwei Dreiecke, die eine Kante teilen, müssen diese Kante
    # nach konsistenter Orientierung entgegengesetzt durchlaufen.
    #
    # flip:
    #   0 = Dreieck bleibt
    #   1 = Dreieck wird umgedreht
    # =========================================================

    adjacency: list[
        list[tuple[int, int, tuple[int, int]]]
    ] = [
        []
        for _ in range(
            number_of_triangles
        )
    ]

    manifold_shared_edges = 0
    boundary_edges = 0
    non_manifold_edges = 0

    for edge, incident in edge_map.items():

        if len(incident) == 1:

            boundary_edges += 1
            continue

        if len(incident) != 2:

            non_manifold_edges += 1
            continue

        manifold_shared_edges += 1

        (
            triangle_a,
            direction_a,
        ) = incident[0]

        (
            triangle_b,
            direction_b,
        ) = incident[1]

        # Wenn beide Dreiecke die gemeinsame Kante aktuell
        # gleichgerichtet durchlaufen, muss eines geflippt werden.
        required_difference = (
            1
            if direction_a == direction_b
            else 0
        )

        adjacency[
            triangle_a
        ].append(
            (
                triangle_b,
                required_difference,
                edge,
            )
        )

        adjacency[
            triangle_b
        ].append(
            (
                triangle_a,
                required_difference,
                edge,
            )
        )

    # =========================================================
    # Paritätszuweisung per BFS
    # =========================================================

    orientation_state = np.full(
        number_of_triangles,
        -1,
        dtype=np.int8,
    )

    conflict_edges: set[
        tuple[int, int]
    ] = set()

    conflict_triangles: set[int] = set()

    triangle_components = 0

    for start_triangle in range(
        number_of_triangles
    ):

        if orientation_state[
            start_triangle
        ] != -1:
            continue

        triangle_components += 1

        orientation_state[
            start_triangle
        ] = 0

        queue = deque(
            [
                start_triangle
            ]
        )

        while queue:

            current = (
                queue.popleft()
            )

            current_state = int(
                orientation_state[
                    current
                ]
            )

            for (
                neighbor,
                required_difference,
                edge,
            ) in adjacency[
                current
            ]:

                expected_state = (
                    current_state
                    ^ required_difference
                )

                if orientation_state[
                    neighbor
                ] == -1:

                    orientation_state[
                        neighbor
                    ] = (
                        expected_state
                    )

                    queue.append(
                        neighbor
                    )

                elif (
                    orientation_state[
                        neighbor
                    ]
                    != expected_state
                ):

                    conflict_edges.add(
                        edge
                    )

                    conflict_triangles.add(
                        current
                    )

                    conflict_triangles.add(
                        neighbor
                    )

    # =========================================================
    # Ergebnis
    # =========================================================

    conflict_triangle_ratio = (
        len(conflict_triangles)
        / number_of_triangles
        if number_of_triangles > 0
        else 0.0
    )

    return {
        "vertices": (
            len(mesh.vertices)
        ),

        "triangles": (
            number_of_triangles
        ),

        "triangle_graph_components": (
            triangle_components
        ),

        "boundary_edges": (
            boundary_edges
        ),

        "manifold_shared_edges": (
            manifold_shared_edges
        ),

        "non_manifold_edges": (
            non_manifold_edges
        ),

        "orientation_conflict_edges": (
            len(conflict_edges)
        ),

        "orientation_conflict_triangles": (
            len(conflict_triangles)
        ),

        "orientation_conflict_triangle_ratio": (
            conflict_triangle_ratio
        ),

        "open3d_orientable": bool(
            mesh.is_orientable()
        ),

        "conflict_edges": [
            [
                int(edge[0]),
                int(edge[1]),
            ]
            for edge in sorted(
                conflict_edges
            )
        ],

        "conflict_triangles": sorted(
            int(value)
            for value in conflict_triangles
        ),
    }


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
        / "orientability_analysis"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # Laden
    # =========================================================

    if not input_path.exists():

        raise FileNotFoundError(
            "Finaler BPA-Kandidat fehlt:\n"
            f"{input_path}"
        )

    print()
    print("=" * 80)
    print("MESH ORIENTABILITY ANALYSIS")
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

    # =========================================================
    # Analyse
    # =========================================================

    result = (
        analyze_orientability(
            mesh
        )
    )

    print()
    print("-" * 80)
    print("ERGEBNIS")
    print("-" * 80)

    print()
    print(
        f"Vertices: "
        f"{result['vertices']:,}"
    )

    print(
        f"Dreiecke: "
        f"{result['triangles']:,}"
    )

    print()
    print(
        "Boundary Edges: "
        f"{result['boundary_edges']:,}"
    )

    print(
        "Echte Non-Manifold Edges: "
        f"{result['non_manifold_edges']:,}"
    )

    print(
        "Triangle-Graph-Komponenten: "
        f"{result['triangle_graph_components']:,}"
    )

    print()
    print(
        "Orientierungs-Konfliktkanten: "
        f"{result['orientation_conflict_edges']:,}"
    )

    print(
        "Von Konflikten betroffene Dreiecke: "
        f"{result['orientation_conflict_triangles']:,}"
    )

    print(
        "Anteil betroffener Dreiecke: "
        f"{result['orientation_conflict_triangle_ratio'] * 100:.4f} %"
    )

    print()
    print(
        "Open3D is_orientable(): "
        f"{result['open3d_orientable']}"
    )

    # =========================================================
    # Konfliktdreiecke visualisieren
    # =========================================================

    conflict_triangles = set(
        result[
            "conflict_triangles"
        ]
    )

    visualization_mesh = (
        o3d.geometry.TriangleMesh(
            mesh
        )
    )

    number_of_vertices = (
        len(
            visualization_mesh.vertices
        )
    )

    colors = np.full(
        (
            number_of_vertices,
            3,
        ),
        [
            0.75,
            0.75,
            0.75,
        ],
        dtype=np.float64,
    )

    triangles = np.asarray(
        mesh.triangles,
        dtype=np.int64,
    )

    conflict_vertex_ids = set()

    for triangle_id in (
        conflict_triangles
    ):

        for vertex_id in (
            triangles[
                triangle_id
            ]
        ):

            conflict_vertex_ids.add(
                int(vertex_id)
            )

    if conflict_vertex_ids:

        conflict_vertex_array = (
            np.asarray(
                sorted(
                    conflict_vertex_ids
                ),
                dtype=np.int64,
            )
        )

        colors[
            conflict_vertex_array
        ] = [
            1.0,
            0.0,
            0.0,
        ]

    visualization_mesh.vertex_colors = (
        o3d.utility.Vector3dVector(
            colors
        )
    )

    visualization_path = (
        output_directory
        / "orientation_conflicts.ply"
    )

    o3d.io.write_triangle_mesh(
        str(
            visualization_path
        ),
        visualization_mesh,
        write_vertex_colors=True,
    )

    # =========================================================
    # JSON
    # =========================================================

    json_path = (
        output_directory
        / "orientability_analysis.json"
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
        "Analyse gespeichert:"
    )

    print(
        f"  {json_path}"
    )

    print()
    print(
        "Visualisierung gespeichert:"
    )

    print(
        f"  {visualization_path}"
    )

    print()
    print("=" * 80)
    print("ANALYSE ABGESCHLOSSEN")
    print("=" * 80)

    print()
    print(
        "Öffne Orientierungs-Konflikte ..."
    )

    o3d.visualization.draw_geometries(
        [
            visualization_mesh
        ],
        window_name=(
            "BPA Orientation Conflicts"
        ),
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()