from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd

from registration3d.realistic_dataset_registry import (
    DENTAL_MODELS,
    JAW_NAMES,
    get_generated_directory,
    get_stl_path,
)


# ============================================================
# CONFIGURATION
# ============================================================

NUMBER_OF_REFERENCE_POINTS = 120_000

NUMBER_OF_FRAGMENTS = 4

NOMINAL_OVERLAP = 0.60

RANDOM_SEED = 42


# Acquisition transformations.
#
# Convention:
#
#     x_observed = A_i @ x_world
#
# Therefore the ground-truth pose that maps an observed
# fragment back into the common world/reference system is:
#
#     P_i = inv(A_i)
#
# Fragment 0 is the reference fragment.
ACQUISITION_TRANSFORMS = (
    {
        "rx": 0.0,
        "ry": 0.0,
        "rz": 0.0,
        "t": (0.0, 0.0, 0.0),
    },
    {
        "rx": 2.0,
        "ry": -3.0,
        "rz": 12.0,
        "t": (3.0, -1.5, 1.0),
    },
    {
        "rx": -2.0,
        "ry": 4.0,
        "rz": -10.0,
        "t": (-3.5, 2.0, 1.5),
    },
    {
        "rx": 3.0,
        "ry": -2.0,
        "rz": 15.0,
        "t": (4.5, 1.0, -1.0),
    },
)


# ============================================================
# HELPERS
# ============================================================

def build_transform(
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    translation: tuple[
        float,
        float,
        float,
    ],
) -> np.ndarray:

    rotation = (
        o3d.geometry.get_rotation_matrix_from_xyz(
            np.radians(
                [
                    rx_deg,
                    ry_deg,
                    rz_deg,
                ]
            )
        )
    )

    transform = np.eye(
        4,
        dtype=float,
    )

    transform[
        :3,
        :3,
    ] = rotation

    transform[
        :3,
        3,
    ] = np.asarray(
        translation,
        dtype=float,
    )

    return transform


def build_acquisition_transforms(
) -> np.ndarray:

    transforms = []

    for config in ACQUISITION_TRANSFORMS:

        transform = build_transform(
            rx_deg=config["rx"],
            ry_deg=config["ry"],
            rz_deg=config["rz"],
            translation=config["t"],
        )

        transforms.append(
            transform
        )

    return np.stack(
        transforms,
        axis=0,
    )


def deterministic_primary_pca_axis(
    points: np.ndarray,
) -> np.ndarray:

    centered = (
        points
        - np.mean(
            points,
            axis=0,
        )
    )

    covariance = np.cov(
        centered,
        rowvar=False,
    )

    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            covariance
        )
    )

    order = np.argsort(
        eigenvalues
    )[::-1]

    primary_axis = (
        eigenvectors[
            :,
            order[0],
        ]
        .astype(
            float
        )
    )

    # Eigenvector signs are mathematically arbitrary.
    # Fix the sign deterministically so repeated runs create
    # the same fragment ordering.
    largest_component = int(
        np.argmax(
            np.abs(
                primary_axis
            )
        )
    )

    if (
        primary_axis[
            largest_component
        ]
        < 0.0
    ):
        primary_axis *= -1.0

    primary_axis /= np.linalg.norm(
        primary_axis
    )

    return primary_axis


def calculate_fragment_windows(
    minimum: float,
    maximum: float,
    number_of_fragments: int,
    overlap: float,
) -> list[
    tuple[
        float,
        float,
    ]
]:

    """
    Create equally sized windows along one PCA coordinate.

    Adjacent windows overlap by the requested fraction.

    If window width is w and step is

        step = (1 - overlap) * w,

    then for N fragments

        total span = w + (N - 1) * step.
    """

    if not (
        0.0
        <= overlap
        < 1.0
    ):
        raise ValueError(
            "Overlap must satisfy 0 <= overlap < 1."
        )

    span = (
        maximum
        - minimum
    )

    denominator = (
        1.0
        + (
            number_of_fragments
            - 1
        )
        * (
            1.0
            - overlap
        )
    )

    window_width = (
        span
        / denominator
    )

    step = (
        window_width
        * (
            1.0
            - overlap
        )
    )

    windows = []

    for index in range(
        number_of_fragments
    ):

        start = (
            minimum
            + index
            * step
        )

        end = (
            start
            + window_width
        )

        # Avoid floating-point loss of the last boundary.
        if (
            index
            == number_of_fragments
            - 1
        ):
            end = maximum

        windows.append(
            (
                float(
                    start
                ),
                float(
                    end
                ),
            )
        )

    return windows


def extract_fragment(
    reference_cloud: o3d.geometry.PointCloud,
    mask: np.ndarray,
) -> o3d.geometry.PointCloud:

    indices = np.flatnonzero(
        mask
    ).tolist()

    fragment = (
        reference_cloud.select_by_index(
            indices
        )
    )

    return fragment


# ============================================================
# DATA GENERATION FOR ONE JAW
# ============================================================

def generate_jaw_dataset(
    project_root: Path,
    model_id: str,
    jaw: str,
    stl_path: Path,
) -> dict:

    output_directory = (
        get_generated_directory(
            project_root=project_root,
            model_id=model_id,
            jaw=jaw,
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 90)
    print(
        f"{model_id} / {jaw.upper()}"
    )
    print("=" * 90)

    print(
        f"Source mesh:\n"
        f"  {stl_path}"
    )

    print(
        f"Output:\n"
        f"  {output_directory}"
    )

    # --------------------------------------------------------
    # Load STL
    # --------------------------------------------------------

    if not stl_path.exists():

        raise FileNotFoundError(
            f"STL not found:\n{stl_path}"
        )

    mesh = (
        o3d.io.read_triangle_mesh(
            str(
                stl_path
            )
        )
    )

    if mesh.is_empty():

        raise RuntimeError(
            f"Could not load mesh:\n{stl_path}"
        )

    mesh.compute_vertex_normals()

    bbox = (
        mesh
        .get_axis_aligned_bounding_box()
    )

    extent = np.asarray(
        bbox.get_extent()
    )

    # --------------------------------------------------------
    # Uniform surface sampling
    # --------------------------------------------------------

    o3d.utility.random.seed(
        RANDOM_SEED
    )

    reference_cloud = (
        mesh.sample_points_uniformly(
            number_of_points=(
                NUMBER_OF_REFERENCE_POINTS
            )
        )
    )

    points = np.asarray(
        reference_cloud.points
    )

    print()
    print(
        "Reference cloud:"
    )

    print(
        f"  {len(points):,} points"
    )

    # --------------------------------------------------------
    # PCA fragmentation axis
    # --------------------------------------------------------

    primary_axis = (
        deterministic_primary_pca_axis(
            points
        )
    )

    center = np.mean(
        points,
        axis=0,
    )

    pca_coordinates = (
        (
            points
            - center
        )
        @ primary_axis
    )

    minimum = float(
        np.min(
            pca_coordinates
        )
    )

    maximum = float(
        np.max(
            pca_coordinates
        )
    )

    windows = (
        calculate_fragment_windows(
            minimum=minimum,
            maximum=maximum,
            number_of_fragments=(
                NUMBER_OF_FRAGMENTS
            ),
            overlap=(
                NOMINAL_OVERLAP
            ),
        )
    )

    # --------------------------------------------------------
    # Transform definitions
    # --------------------------------------------------------

    acquisition_transforms = (
        build_acquisition_transforms()
    )

    ground_truth_poses = np.linalg.inv(
        acquisition_transforms
    )

    # --------------------------------------------------------
    # Save reference
    # --------------------------------------------------------

    reference_path = (
        output_directory
        / "reference_cloud.ply"
    )

    success = (
        o3d.io.write_point_cloud(
            str(
                reference_path
            ),
            reference_cloud,
        )
    )

    if not success:

        raise RuntimeError(
            "Could not save reference cloud."
        )

    fragment_rows = []

    # --------------------------------------------------------
    # Generate fragments
    # --------------------------------------------------------

    for index, (
        window_start,
        window_end,
    ) in enumerate(
        windows
    ):

        if (
            index
            == NUMBER_OF_FRAGMENTS
            - 1
        ):

            mask = (
                (
                    pca_coordinates
                    >= window_start
                )
                &
                (
                    pca_coordinates
                    <= window_end
                )
            )

        else:

            mask = (
                (
                    pca_coordinates
                    >= window_start
                )
                &
                (
                    pca_coordinates
                    < window_end
                )
            )

        world_fragment = (
            extract_fragment(
                reference_cloud=(
                    reference_cloud
                ),
                mask=mask,
            )
        )

        observed_fragment = (
            copy.deepcopy(
                world_fragment
            )
        )

        observed_fragment.transform(
            acquisition_transforms[
                index
            ]
        )

        world_path = (
            output_directory
            / (
                f"world_fragment_"
                f"{index:02d}.ply"
            )
        )

        observed_path = (
            output_directory
            / (
                f"fragment_"
                f"{index:02d}.ply"
            )
        )

        success_world = (
            o3d.io.write_point_cloud(
                str(
                    world_path
                ),
                world_fragment,
            )
        )

        success_observed = (
            o3d.io.write_point_cloud(
                str(
                    observed_path
                ),
                observed_fragment,
            )
        )

        if (
            not success_world
            or not success_observed
        ):

            raise RuntimeError(
                f"Could not save fragment {index}."
            )

        point_count = len(
            world_fragment.points
        )

        print(
            f"Fragment {index}: "
            f"{point_count:,} points"
        )

        fragment_rows.append(
            {
                "Model":
                    model_id,

                "Jaw":
                    jaw,

                "Fragment":
                    index,

                "Points":
                    point_count,

                "PCA Window Start":
                    window_start,

                "PCA Window End":
                    window_end,

                "Nominal Overlap":
                    NOMINAL_OVERLAP,
            }
        )

    # --------------------------------------------------------
    # Save transformations
    # --------------------------------------------------------

    np.save(
        output_directory
        / "acquisition_transforms.npy",
        acquisition_transforms,
    )

    np.save(
        output_directory
        / "ground_truth_poses.npy",
        ground_truth_poses,
    )

    # --------------------------------------------------------
    # Save fragment metadata
    # --------------------------------------------------------

    fragment_df = pd.DataFrame(
        fragment_rows
    )

    fragment_df.to_csv(
        output_directory
        / "fragment_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Human-readable metadata
    # --------------------------------------------------------

    metadata = {
        "model_id":
            model_id,

        "jaw":
            jaw,

        "source_stl":
            stl_path.name,

        "reference_points":
            NUMBER_OF_REFERENCE_POINTS,

        "number_of_fragments":
            NUMBER_OF_FRAGMENTS,

        "nominal_overlap":
            NOMINAL_OVERLAP,

        "random_seed":
            RANDOM_SEED,

        "extent_x":
            float(
                extent[0]
            ),

        "extent_y":
            float(
                extent[1]
            ),

        "extent_z":
            float(
                extent[2]
            ),

        "pca_primary_axis":
            primary_axis.tolist(),

        "fragment_point_counts": [
            int(
                row["Points"]
            )
            for row
            in fragment_rows
        ],

        "transform_convention":
            (
                "x_observed = "
                "A_i @ x_world; "
                "ground_truth_pose = inv(A_i)"
            ),
    }

    with open(
        output_directory
        / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    return {
        "Model":
            model_id,

        "Jaw":
            jaw,

        "Source STL":
            stl_path.name,

        "Reference Points":
            NUMBER_OF_REFERENCE_POINTS,

        "Fragment 0":
            fragment_rows[0]["Points"],

        "Fragment 1":
            fragment_rows[1]["Points"],

        "Fragment 2":
            fragment_rows[2]["Points"],

        "Fragment 3":
            fragment_rows[3]["Points"],

        "Extent X":
            extent[0],

        "Extent Y":
            extent[1],

        "Extent Z":
            extent[2],
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    print()
    print("=" * 90)
    print(
        "REALISTIC FIVE-MODEL DATASET GENERATION"
    )
    print("=" * 90)

    print()
    print(
        f"Models:             "
        f"{len(DENTAL_MODELS)}"
    )

    print(
        f"Jaw meshes:         "
        f"{len(DENTAL_MODELS) * 2}"
    )

    print(
        f"Surface samples:    "
        f"{NUMBER_OF_REFERENCE_POINTS:,}"
    )

    print(
        f"Fragments per jaw:  "
        f"{NUMBER_OF_FRAGMENTS}"
    )

    print(
        f"Nominal overlap:    "
        f"{NOMINAL_OVERLAP:.0%}"
    )

    print(
        f"Random seed:        "
        f"{RANDOM_SEED}"
    )

    rows = []

    for model in (
        DENTAL_MODELS
    ):

        for jaw in (
            JAW_NAMES
        ):

            stl_path = (
                get_stl_path(
                    project_root=(
                        project_root
                    ),
                    model=model,
                    jaw=jaw,
                )
            )

            row = (
                generate_jaw_dataset(
                    project_root=(
                        project_root
                    ),
                    model_id=(
                        model.model_id
                    ),
                    jaw=jaw,
                    stl_path=(
                        stl_path
                    ),
                )
            )

            rows.append(
                row
            )

    # --------------------------------------------------------
    # Aggregate summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        rows
    )

    aggregate_directory = (
        project_root
        / "output"
        / "realistic_aggregate"
    )

    aggregate_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        aggregate_directory
        / "generated_dataset_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 90)
    print(
        "DATASET GENERATION COMPLETE"
    )
    print("=" * 90)

    print()

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    print()
    print(
        f"Summary saved:\n"
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()