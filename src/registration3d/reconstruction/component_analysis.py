from pathlib import Path
import copy
import csv

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
        / "ball_pivoting_experiments"
        / "A_small"
        / "cleaned_mesh.ply"
    )

    output_directory = (
        project_root
        / "output"
        / "mesh"
        / "ball_pivoting_experiments"
        / "A_small"
        / "component_analysis"
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
            "A_small Cleaned Mesh wurde "
            "nicht gefunden:\n"
            f"{input_path}\n\n"
            "Bitte zuerst das "
            "Ball-Pivoting-Experiment ausführen."
        )

    # =========================================================
    # Mesh laden
    # =========================================================

    print()
    print("=" * 80)
    print("BALL PIVOTING COMPONENT ANALYSIS")
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

    print(
        f"Vertices: "
        f"{len(mesh.vertices):,}"
    )

    print(
        f"Dreiecke: "
        f"{len(mesh.triangles):,}"
    )

    # =========================================================
    # Komponenten bestimmen
    # =========================================================

    (
        triangle_clusters,
        cluster_n_triangles,
        cluster_area,
    ) = (
        mesh.cluster_connected_triangles()
    )

    triangle_clusters = np.asarray(
        triangle_clusters,
        dtype=np.int64,
    )

    cluster_n_triangles = np.asarray(
        cluster_n_triangles,
        dtype=np.int64,
    )

    cluster_area = np.asarray(
        cluster_area,
        dtype=float,
    )

    number_of_components = (
        len(cluster_n_triangles)
    )

    total_triangles = (
        len(mesh.triangles)
    )

    total_area = float(
        np.sum(cluster_area)
    )

    # =========================================================
    # Nach Dreiecksanzahl sortieren
    # =========================================================

    sorted_indices = np.argsort(
        cluster_n_triangles
    )[::-1]

    print()
    print(
        f"Connected Components: "
        f"{number_of_components}"
    )

    # =========================================================
    # Größte Komponenten anzeigen
    # =========================================================

    print()
    print("-" * 80)
    print("GRÖSSTE KOMPONENTEN")
    print("-" * 80)

    print()
    print(
        f"{'Rang':>5}"
        f"{'ID':>7}"
        f"{'Triangles':>14}"
        f"{'Triangle %':>14}"
        f"{'Area':>14}"
        f"{'Area %':>12}"
    )

    print(
        "-" * 66
    )

    number_to_print = min(
        20,
        number_of_components,
    )

    for rank, component_id in enumerate(
        sorted_indices[:number_to_print],
        start=1,
    ):

        triangles = int(
            cluster_n_triangles[
                component_id
            ]
        )

        area = float(
            cluster_area[
                component_id
            ]
        )

        triangle_ratio = (
            triangles
            / total_triangles
            * 100.0
        )

        area_ratio = (
            area
            / total_area
            * 100.0
            if total_area > 0.0
            else 0.0
        )

        print(
            f"{rank:>5}"
            f"{component_id:>7}"
            f"{triangles:>14,}"
            f"{triangle_ratio:>13.3f}%"
            f"{area:>14.6f}"
            f"{area_ratio:>11.3f}%"
        )

    # =========================================================
    # Größenverteilung
    # =========================================================

    thresholds = (
        5,
        10,
        25,
        50,
        100,
        250,
        500,
    )

    print()
    print("-" * 80)
    print("KOMPONENTEN-GRÖSSENVERTEILUNG")
    print("-" * 80)

    print()

    for threshold in thresholds:

        mask = (
            cluster_n_triangles
            < threshold
        )

        component_count = int(
            np.count_nonzero(
                mask
            )
        )

        affected_triangles = int(
            np.sum(
                cluster_n_triangles[
                    mask
                ]
            )
        )

        triangle_ratio = (
            affected_triangles
            / total_triangles
            * 100.0
        )

        print(
            f"< {threshold:>3} Dreiecke: "
            f"{component_count:>3} Komponenten | "
            f"{affected_triangles:>5,} Dreiecke | "
            f"{triangle_ratio:>7.3f} % des Meshes"
        )

    # =========================================================
    # Größte Komponente
    # =========================================================

    largest_component_id = int(
        sorted_indices[0]
    )

    largest_triangle_count = int(
        cluster_n_triangles[
            largest_component_id
        ]
    )

    largest_ratio = (
        largest_triangle_count
        / total_triangles
    )

    print()
    print("-" * 80)
    print("HAUPTKOMPONENTE")
    print("-" * 80)

    print()
    print(
        f"Component ID: "
        f"{largest_component_id}"
    )

    print(
        f"Dreiecke: "
        f"{largest_triangle_count:,}"
    )

    print(
        f"Anteil am gesamten Mesh: "
        f"{largest_ratio * 100:.3f} %"
    )

    # =========================================================
    # Hauptkomponente separat erzeugen
    # =========================================================

    main_component_mesh = (
        copy.deepcopy(
            mesh
        )
    )

    triangles_to_remove = (
        triangle_clusters
        != largest_component_id
    )

    main_component_mesh.remove_triangles_by_mask(
        triangles_to_remove
    )

    main_component_mesh.remove_unreferenced_vertices()

    main_component_mesh.compute_triangle_normals()
    main_component_mesh.compute_vertex_normals()

    main_component_path = (
        output_directory
        / "largest_component.ply"
    )

    success = (
        o3d.io.write_triangle_mesh(
            str(main_component_path),
            main_component_mesh,
        )
    )

    if not success:
        raise RuntimeError(
            "Hauptkomponente konnte "
            "nicht gespeichert werden."
        )

    print()
    print(
        "Hauptkomponente gespeichert:"
    )

    print(
        f"  {main_component_path}"
    )

    print(
        f"  Vertices: "
        f"{len(main_component_mesh.vertices):,}"
    )

    print(
        f"  Dreiecke: "
        f"{len(main_component_mesh.triangles):,}"
    )

    # =========================================================
    # Jede Komponente für Visualisierung einfärben
    # =========================================================

    colored_mesh = (
        copy.deepcopy(
            mesh
        )
    )

    vertex_count = (
        len(colored_mesh.vertices)
    )

    vertex_component_votes = [
        []
        for _ in range(
            vertex_count
        )
    ]

    triangles = np.asarray(
        colored_mesh.triangles,
        dtype=np.int64,
    )

    for triangle_index, triangle in enumerate(
        triangles
    ):

        component_id = int(
            triangle_clusters[
                triangle_index
            ]
        )

        for vertex_id in triangle:
            vertex_component_votes[
                int(vertex_id)
            ].append(
                component_id
            )

    rng = np.random.default_rng(
        42
    )

    component_colors = rng.random(
        (
            number_of_components,
            3,
        )
    )

    vertex_colors = np.zeros(
        (
            vertex_count,
            3,
        ),
        dtype=float,
    )

    for vertex_id, votes in enumerate(
        vertex_component_votes
    ):

        if not votes:
            continue

        component_id = int(
            np.bincount(
                votes
            ).argmax()
        )

        vertex_colors[
            vertex_id
        ] = (
            component_colors[
                component_id
            ]
        )

    colored_mesh.vertex_colors = (
        o3d.utility.Vector3dVector(
            vertex_colors
        )
    )

    colored_mesh_path = (
        output_directory
        / "colored_components.ply"
    )

    o3d.io.write_triangle_mesh(
        str(
            colored_mesh_path
        ),
        colored_mesh,
        write_vertex_colors=True,
    )

    print()
    print(
        "Eingefärbtes Komponenten-Mesh gespeichert:"
    )

    print(
        f"  {colored_mesh_path}"
    )

    # =========================================================
    # CSV mit allen Komponenten
    # =========================================================

    csv_path = (
        output_directory
        / "components.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "Component ID",
                "Triangles",
                "Triangle Ratio",
                "Area",
                "Area Ratio",
            ]
        )

        for component_id in sorted_indices:

            triangles_count = int(
                cluster_n_triangles[
                    component_id
                ]
            )

            area = float(
                cluster_area[
                    component_id
                ]
            )

            writer.writerow(
                [
                    int(
                        component_id
                    ),
                    triangles_count,
                    (
                        triangles_count
                        / total_triangles
                    ),
                    area,
                    (
                        area
                        / total_area
                        if total_area > 0.0
                        else 0.0
                    ),
                ]
            )

    print()
    print(
        "Komponenten-CSV gespeichert:"
    )

    print(
        f"  {csv_path}"
    )

    # =========================================================
    # Visualisierung
    # =========================================================

    print()
    print("=" * 80)
    print("ANALYSE ABGESCHLOSSEN")
    print("=" * 80)

    print()
    print(
        "Öffne eingefärbte Mesh-Komponenten ..."
    )

    o3d.visualization.draw_geometries(
        [
            colored_mesh
        ],
        window_name=(
            "A_small - Connected Components"
        ),
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()