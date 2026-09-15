from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd

from registration3d.realistic_dataset_registry import (
    DENTAL_MODELS,
    JAW_NAMES,
    get_stl_path,
)


def main() -> None:

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    rows = []

    print()
    print("=" * 90)
    print("REALISTIC DENTAL MODEL SUITE VALIDATION")
    print("=" * 90)

    for model in DENTAL_MODELS:

        for jaw in JAW_NAMES:

            path = get_stl_path(
                project_root=project_root,
                model=model,
                jaw=jaw,
            )

            print()
            print("-" * 90)
            print(
                f"{model.model_id} / {jaw}"
            )
            print(
                f"File: {path}"
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing STL:\n{path}"
                )

            mesh = (
                o3d.io.read_triangle_mesh(
                    str(path)
                )
            )

            if mesh.is_empty():
                raise RuntimeError(
                    f"Could not load STL:\n{path}"
                )

            vertices = np.asarray(
                mesh.vertices
            )

            triangles = np.asarray(
                mesh.triangles
            )

            bbox = (
                mesh
                .get_axis_aligned_bounding_box()
            )

            extent = np.asarray(
                bbox.get_extent()
            )

            diagonal = float(
                np.linalg.norm(
                    extent
                )
            )

            center = np.asarray(
                bbox.get_center()
            )

            try:
                surface_area = float(
                    mesh.get_surface_area()
                )
            except RuntimeError:
                surface_area = float(
                    "nan"
                )

            print(
                f"Vertices:      "
                f"{len(vertices):,}"
            )

            print(
                f"Triangles:     "
                f"{len(triangles):,}"
            )

            print(
                "Dimensions:     "
                f"{extent[0]:.3f} x "
                f"{extent[1]:.3f} x "
                f"{extent[2]:.3f}"
            )

            print(
                f"BBox diagonal: "
                f"{diagonal:.3f}"
            )

            print(
                f"Surface area:  "
                f"{surface_area:.3f}"
            )

            print(
                "Center:         "
                f"({center[0]:.3f}, "
                f"{center[1]:.3f}, "
                f"{center[2]:.3f})"
            )

            rows.append(
                {
                    "Model": model.model_id,
                    "Jaw": jaw,
                    "File": path.name,
                    "Vertices": len(vertices),
                    "Triangles": len(triangles),
                    "Extent X": extent[0],
                    "Extent Y": extent[1],
                    "Extent Z": extent[2],
                    "BBox Diagonal": diagonal,
                    "Surface Area": surface_area,
                }
            )

    df = pd.DataFrame(
        rows
    )

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print()

    print(
        df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    output_directory = (
        project_root
        / "output"
        / "realistic_aggregate"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "model_suite_summary.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    diagonals = (
        df[
            "BBox Diagonal"
        ]
        .to_numpy(
            dtype=float
        )
    )

    ratio = (
        diagonals.max()
        / diagonals.min()
    )

    print()
    print(
        "Largest/smallest bounding-box "
        f"diagonal ratio: {ratio:.3f}"
    )

    if ratio > 2.0:

        print()
        print(
            "WARNING:"
        )

        print(
            "The STL models appear to use "
            "substantially different scales."
        )

        print(
            "Do NOT run the benchmark with "
            "one common voxel size before "
            "checking the units."
        )

    else:

        print()
        print(
            "Scale check looks plausible."
        )

        print(
            "The models can likely be evaluated "
            "with the common registration "
            "configuration."
        )

    print()
    print(
        f"Saved summary:\n{output_path}"
    )


if __name__ == "__main__":
    main()