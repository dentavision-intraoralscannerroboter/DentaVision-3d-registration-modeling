from pathlib import Path
import copy

import numpy as np
import open3d as o3d

from registration3d.synthetic_data import (
    SyntheticDentalDataGenerator,
)


def create_transform(
    rotation_degrees: tuple[
        float,
        float,
        float,
    ],
    translation: tuple[
        float,
        float,
        float,
    ],
) -> np.ndarray:

    rotation = (
        o3d.geometry
        .get_rotation_matrix_from_xyz(
            np.radians(
                rotation_degrees
            )
        )
    )

    transformation = np.eye(4)

    transformation[
        :3,
        :3,
    ] = rotation

    transformation[
        :3,
        3,
    ] = np.asarray(
        translation,
        dtype=float,
    )

    return transformation


def crop_x(
    cloud: o3d.geometry.PointCloud,
    min_x: float | None,
    max_x: float | None,
) -> o3d.geometry.PointCloud:

    points = np.asarray(
        cloud.points
    )

    mask = np.ones(
        len(points),
        dtype=bool,
    )

    if min_x is not None:

        mask &= (
            points[:, 0]
            >= min_x
        )

    if max_x is not None:

        mask &= (
            points[:, 0]
            <= max_x
        )

    indices = np.where(
        mask
    )[0]

    return cloud.select_by_index(
        indices.tolist()
    )


def add_noise(
    cloud: o3d.geometry.PointCloud,
    std: float,
    rng: np.random.Generator,
) -> o3d.geometry.PointCloud:

    noisy = copy.deepcopy(
        cloud
    )

    if std <= 0.0:
        return noisy

    points = np.asarray(
        noisy.points
    ).copy()

    points += rng.normal(
        loc=0.0,
        scale=std,
        size=points.shape,
    )

    noisy.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )

    return noisy


def main() -> None:

    project_root = Path(
        __file__
    ).resolve().parent

    output_directory = (
        project_root
        / "data"
        / "multiway_test"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Reproduzierbarkeit
    # ---------------------------------------------------------

    seed = 1234

    np_rng = (
        np.random.default_rng(
            seed
        )
    )

    o3d.utility.random.seed(
        seed
    )

    # ---------------------------------------------------------
    # Dasselbe dentalartige Modell wie im Benchmark
    # ---------------------------------------------------------

    generator = (
        SyntheticDentalDataGenerator(
            number_of_points=30_000
        )
    )

    mesh = generator.mesh

    # ---------------------------------------------------------
    # Fragmentbereiche
    #
    # Es gibt starke Überlappung benachbarter Fragmente
    # und etwas Überlappung für Loop Closures.
    # ---------------------------------------------------------

    windows = [
        (
            None,
            -1.0,
        ),
        (
            -14.0,
            7.0,
        ),
        (
            -7.0,
            14.0,
        ),
        (
            1.0,
            None,
        ),
    ]

    # ---------------------------------------------------------
    # Künstliche Scan-Transformationen
    #
    # Diese bringen jedes Fragment absichtlich
    # in ein anderes lokales Koordinatensystem.
    # ---------------------------------------------------------

    acquisition_transforms = [
        np.eye(4),

        create_transform(
            rotation_degrees=(
                3.0,
                -5.0,
                8.0,
            ),
            translation=(
                2.0,
                -1.0,
                0.7,
            ),
        ),

        create_transform(
            rotation_degrees=(
                -4.0,
                7.0,
                -10.0,
            ),
            translation=(
                -2.5,
                2.0,
                1.2,
            ),
        ),

        create_transform(
            rotation_degrees=(
                6.0,
                -3.0,
                12.0,
            ),
            translation=(
                3.5,
                1.5,
                -0.8,
            ),
        ),
    ]

    fragments = []

    # ---------------------------------------------------------
    # Fragmente erzeugen
    # ---------------------------------------------------------

    for index, (
        window,
        transformation,
    ) in enumerate(
        zip(
            windows,
            acquisition_transforms,
        )
    ):

        full_cloud = (
            mesh.sample_points_uniformly(
                number_of_points=30_000
            )
        )

        fragment = crop_x(
            full_cloud,
            min_x=window[0],
            max_x=window[1],
        )

        fragment.transform(
            transformation
        )

        fragment = add_noise(
            fragment,
            std=0.02,
            rng=np_rng,
        )

        fragments.append(
            fragment
        )

        output_path = (
            output_directory
            / f"fragment_{index:02d}.ply"
        )

        o3d.io.write_point_cloud(
            str(output_path),
            fragment,
        )

        print(
            f"Fragment {index}: "
            f"{len(fragment.points):,} Punkte"
        )

        print(
            f"  gespeichert: {output_path}"
        )

    # ---------------------------------------------------------
    # Ground Truth
    # ---------------------------------------------------------
    #
    # acquisition_transform:
    #
    # Welt -> lokales Fragment
    #
    # Gesuchte Pose für den Pose Graph:
    #
    # lokales Fragment -> Welt / Fragment 0
    #
    # daher Inverse.
    # ---------------------------------------------------------

    ground_truth_poses = np.stack(
        [
            np.linalg.inv(
                transformation
            )
            for transformation
            in acquisition_transforms
        ]
    )

    np.save(
        output_directory
        / "ground_truth_poses.npy",
        ground_truth_poses,
    )

    np.save(
        output_directory
        / "acquisition_transforms.npy",
        np.stack(
            acquisition_transforms
        ),
    )

    # ---------------------------------------------------------
    # Vollständiges Referenzmodell speichern
    # ---------------------------------------------------------

    reference = (
        mesh.sample_points_uniformly(
            number_of_points=50_000
        )
    )

    o3d.io.write_point_cloud(
        str(
            output_directory
            / "reference_cloud.ply"
        ),
        reference,
    )

    print()
    print("=" * 80)
    print("MULTIWAY-TESTDATEN ERSTELLT")
    print("=" * 80)

    print(
        f"Ordner: {output_directory}"
    )


if __name__ == "__main__":
    main()