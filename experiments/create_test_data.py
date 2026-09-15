from pathlib import Path
import copy

import numpy as np
import open3d as o3d


def create_ground_truth_transform() -> np.ndarray:
    """
    Erstellt die bekannte Transformation, mit der
    die zweite Punktwolke verändert wird.
    """

    rotation = o3d.geometry.get_rotation_matrix_from_xyz(
        np.radians([
            18.0,
            -10.0,
            25.0,
        ])
    )

    translation = np.array([
        8.0,
        -5.0,
        4.0,
    ])

    transformation = np.eye(4)

    transformation[:3, :3] = rotation
    transformation[:3, 3] = translation

    return transformation


def create_asymmetric_mesh() -> o3d.geometry.TriangleMesh:
    """
    Erzeugt ein bewusst asymmetrisches künstliches Objekt.
    """

    # Grundkörper
    box = o3d.geometry.TriangleMesh.create_box(
        width=30.0,
        height=14.0,
        depth=8.0,
    )

    # Größere Kugel auf einer Seite
    sphere_1 = o3d.geometry.TriangleMesh.create_sphere(
        radius=5.0
    )

    sphere_1.translate([
        7.0,
        7.0,
        8.0,
    ])

    # Kleinere Kugel an anderer Position
    sphere_2 = o3d.geometry.TriangleMesh.create_sphere(
        radius=3.0
    )

    sphere_2.translate([
        23.0,
        4.0,
        8.0,
    ])

    # Zylinder als weiteres eindeutiges Merkmal
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=2.0,
        height=9.0,
    )

    cylinder.translate([
        17.0,
        10.0,
        4.0,
    ])

    mesh = (
        box
        + sphere_1
        + sphere_2
        + cylinder
    )

    mesh.compute_vertex_normals()

    return mesh


def crop_cloud(
    cloud: o3d.geometry.PointCloud,
    min_x: float | None = None,
    max_x: float | None = None,
) -> o3d.geometry.PointCloud:

    points = np.asarray(cloud.points)

    mask = np.ones(
        len(points),
        dtype=bool,
    )

    if min_x is not None:
        mask &= points[:, 0] >= min_x

    if max_x is not None:
        mask &= points[:, 0] <= max_x

    indices = np.where(mask)[0]

    return cloud.select_by_index(
        indices.tolist()
    )


def main() -> None:

    data_directory = Path("data")

    data_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 1. Vollständiges künstliches Modell
    # ---------------------------------------------------------

    mesh = create_asymmetric_mesh()

    # Wir erzeugen zwei unabhängige Scans desselben Objekts.
    source_full = mesh.sample_points_poisson_disk(
        number_of_points=50_000
    )

    target_full = mesh.sample_points_poisson_disk(
        number_of_points=50_000
    )

    reference = mesh.sample_points_poisson_disk(
        number_of_points=60_000
    )

    # ---------------------------------------------------------
    # 2. Zwei teilweise überlappende Fragmente
    # ---------------------------------------------------------

    # Linker / mittlerer Bereich
    source = crop_cloud(
        source_full,
        max_x=21.0,
    )

    # Mittlerer / rechter Bereich
    target = crop_cloud(
        target_full,
        min_x=9.0,
    )

    # Damit liegt ungefähr im Bereich x=9..21
    # eine Überlappung vor.

    # ---------------------------------------------------------
    # 3. Ground-Truth-Transformation
    # ---------------------------------------------------------

    ground_truth = (
        create_ground_truth_transform()
    )

    # Target bewusst aus der richtigen Lage bewegen.
    target_transformed = copy.deepcopy(
        target
    )

    target_transformed.transform(
        ground_truth
    )

    # ---------------------------------------------------------
    # 4. Daten speichern
    # ---------------------------------------------------------

    success_source = (
        o3d.io.write_point_cloud(
            str(
                data_directory
                / "cloud_01.ply"
            ),
            source,
        )
    )

    success_target = (
        o3d.io.write_point_cloud(
            str(
                data_directory
                / "cloud_02.ply"
            ),
            target_transformed,
        )
    )

    success_reference = (
        o3d.io.write_point_cloud(
            str(
                data_directory
                / "reference_cloud.ply"
            ),
            reference,
        )
    )

    if not (
        success_source
        and success_target
        and success_reference
    ):
        raise RuntimeError(
            "Testdaten konnten nicht gespeichert werden."
        )

    np.save(
        data_directory
        / "ground_truth.npy",
        ground_truth,
    )

    print()
    print("Testdaten erstellt.")
    print()
    print(
        f"Source: "
        f"{len(source.points):,} Punkte"
    )
    print(
        f"Target: "
        f"{len(target_transformed.points):,} Punkte"
    )

    print()
    print("Ground-Truth-Transformation:")
    print(ground_truth)


if __name__ == "__main__":
    main()