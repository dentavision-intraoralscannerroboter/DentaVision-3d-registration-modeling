from pathlib import Path

import numpy as np
import open3d as o3d


BASE_DIR = Path("data/realistic_jaw")

JAW_FILES = {
    "upper": BASE_DIR / "upper_jaw.stl",
    "lower": BASE_DIR / "lower_jaw.stl",
}

NUMBER_OF_SAMPLE_POINTS = 120_000
NUMBER_OF_FRAGMENTS = 4
OVERLAP = 0.60
SEED = 42


def rotation_matrix_xyz(
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
) -> np.ndarray:
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    rx_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rx), -np.sin(rx)],
            [0.0, np.sin(rx), np.cos(rx)],
        ]
    )

    ry_matrix = np.array(
        [
            [np.cos(ry), 0.0, np.sin(ry)],
            [0.0, 1.0, 0.0],
            [-np.sin(ry), 0.0, np.cos(ry)],
        ]
    )

    rz_matrix = np.array(
        [
            [np.cos(rz), -np.sin(rz), 0.0],
            [np.sin(rz), np.cos(rz), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return rz_matrix @ ry_matrix @ rx_matrix


def make_transform(
    rx: float,
    ry: float,
    rz: float,
    tx: float,
    ty: float,
    tz: float,
) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix_xyz(rx, ry, rz)
    transform[:3, 3] = np.array([tx, ty, tz])

    return transform


def create_acquisition_transforms() -> list[np.ndarray]:
    """
    Kontrollierte Transformationen der einzelnen Fragmente.

    x_i = A_i @ x_world
    """

    return [
        np.eye(4),
        make_transform(
            rx=2.0,
            ry=-3.0,
            rz=12.0,
            tx=3.0,
            ty=-1.5,
            tz=1.0,
        ),
        make_transform(
            rx=-2.0,
            ry=4.0,
            rz=-10.0,
            tx=-3.5,
            ty=2.0,
            tz=1.5,
        ),
        make_transform(
            rx=3.0,
            ry=-2.0,
            rz=15.0,
            tx=4.5,
            ty=1.0,
            tz=-1.0,
        ),
    ]


def create_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    return cloud


def get_principal_axis(points: np.ndarray) -> np.ndarray:
    """
    Bestimmt über PCA die größte räumliche Ausdehnungsrichtung.
    Entlang dieser Richtung wird der Kiefer in überlappende
    Teilpunktwolken zerlegt.
    """

    centered = points - points.mean(axis=0)

    covariance = np.cov(centered.T)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

    return principal_axis


def fragment_point_cloud(
    points: np.ndarray,
    number_of_fragments: int,
    overlap: float,
) -> list[np.ndarray]:

    centered = points - points.mean(axis=0)

    principal_axis = get_principal_axis(points)

    projection = centered @ principal_axis

    minimum = float(projection.min())
    maximum = float(projection.max())

    total_span = maximum - minimum

    # Bei benachbarten Fenstern gilt:
    #
    # step = width * (1 - overlap)
    #
    # Gesamtspanne:
    # width + (n - 1) * step
    #
    window_width = total_span / (
        1.0
        + (number_of_fragments - 1)
        * (1.0 - overlap)
    )

    step = window_width * (1.0 - overlap)

    fragments = []

    print()
    print("Fragmentierung")
    print("-----------------------------")
    print(f"PCA-Spanne:       {total_span:.3f}")
    print(f"Fensterbreite:    {window_width:.3f}")
    print(f"Schrittweite:     {step:.3f}")
    print(f"Ziel-Überlappung: {overlap * 100:.1f} %")
    print()

    for index in range(number_of_fragments):

        lower = minimum + index * step
        upper = lower + window_width

        if index == number_of_fragments - 1:
            upper = maximum + 1e-9

        mask = (
            (projection >= lower)
            & (projection <= upper)
        )

        fragment = points[mask]

        fragments.append(fragment)

        print(
            f"Fragment {index}: "
            f"{len(fragment):,} Punkte"
        )

    return fragments


def process_jaw(
    jaw_name: str,
    input_path: Path,
) -> None:

    print()
    print("=" * 70)
    print(f"{jaw_name.upper()} JAW")
    print("=" * 70)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden: {input_path}"
        )

    output_dir = BASE_DIR / f"{jaw_name}_test"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Lade Mesh: {input_path}")

    mesh = o3d.io.read_triangle_mesh(
        str(input_path)
    )

    if mesh.is_empty():
        raise RuntimeError(
            f"Mesh konnte nicht geladen werden: {input_path}"
        )

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()

    print()
    print("Originalmesh")
    print("-----------------------------")
    print(
        f"Vertices:  "
        f"{len(mesh.vertices):,}"
    )
    print(
        f"Triangles: "
        f"{len(mesh.triangles):,}"
    )

    extent = np.asarray(
        mesh.get_axis_aligned_bounding_box().get_extent()
    )

    print(
        "Bounding Box: "
        f"{extent[0]:.3f} x "
        f"{extent[1]:.3f} x "
        f"{extent[2]:.3f}"
    )

    try:
        area = mesh.get_surface_area()
        print(
            f"Oberfläche: "
            f"{area:.3f}"
        )
    except RuntimeError:
        pass

    print()
    print(
        f"Sample {NUMBER_OF_SAMPLE_POINTS:,} "
        "Punkte von der STL-Oberfläche ..."
    )

    o3d.utility.random.seed(SEED)

    reference_cloud = mesh.sample_points_uniformly(
        number_of_points=NUMBER_OF_SAMPLE_POINTS
    )

    points = np.asarray(
        reference_cloud.points
    )

    reference_path = (
        output_dir / "reference_cloud.ply"
    )

    o3d.io.write_point_cloud(
        str(reference_path),
        reference_cloud,
    )

    print(
        f"Referenzpunktwolke gespeichert: "
        f"{reference_path}"
    )

    fragments_world = fragment_point_cloud(
        points=points,
        number_of_fragments=NUMBER_OF_FRAGMENTS,
        overlap=OVERLAP,
    )

    acquisition_transforms = (
        create_acquisition_transforms()
    )

    ground_truth_poses = []

    for index, (
        fragment_world,
        acquisition_transform,
    ) in enumerate(
        zip(
            fragments_world,
            acquisition_transforms,
        )
    ):

        world_cloud = create_point_cloud(
            fragment_world
        )

        observed_cloud = create_point_cloud(
            fragment_world.copy()
        )

        observed_cloud.transform(
            acquisition_transform
        )

        fragment_path = (
            output_dir
            / f"fragment_{index:02d}.ply"
        )

        world_path = (
            output_dir
            / f"fragment_{index:02d}_world.ply"
        )

        o3d.io.write_point_cloud(
            str(fragment_path),
            observed_cloud,
        )

        o3d.io.write_point_cloud(
            str(world_path),
            world_cloud,
        )

        # Pose, die das beobachtete Fragment wieder
        # in das Referenzkoordinatensystem überführt.
        ground_truth_pose = np.linalg.inv(
            acquisition_transform
        )

        ground_truth_poses.append(
            ground_truth_pose
        )

        print(
            f"Gespeichert: {fragment_path.name}"
        )

    acquisition_array = np.stack(
        acquisition_transforms
    )

    ground_truth_array = np.stack(
        ground_truth_poses
    )

    np.save(
        output_dir
        / "acquisition_transforms.npy",
        acquisition_array,
    )

    np.save(
        output_dir
        / "ground_truth_poses.npy",
        ground_truth_array,
    )

    print()
    print(
        "Ground Truth gespeichert:"
    )
    print(
        output_dir
        / "acquisition_transforms.npy"
    )
    print(
        output_dir
        / "ground_truth_poses.npy"
    )

    print()
    print(
        f"{jaw_name.upper()} abgeschlossen."
    )


def main() -> None:

    print()
    print(
        "REALISTIC DENTAL TEST DATA GENERATOR"
    )

    for jaw_name, input_path in JAW_FILES.items():
        process_jaw(
            jaw_name=jaw_name,
            input_path=input_path,
        )

    print()
    print("=" * 70)
    print("FERTIG")
    print("=" * 70)


if __name__ == "__main__":
    main()