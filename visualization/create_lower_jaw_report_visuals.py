from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RenderConfig:
    width: int = 1400
    height: int = 1000

    point_size: float = 2.5

    # White report background
    background_color: tuple[float, float, float] = (
        1.0,
        1.0,
        1.0,
    )

    # Occlusal-like lower-jaw view
    front: tuple[float, float, float] = (
        0.0,
        0.0,
        -1.0,
    )

    up: tuple[float, float, float] = (
        0.0,
        -1.0,
        0.0,
    )

    zoom: float = 0.72


# ============================================================
# LOWER JAW REPORT RENDERER
# ============================================================

class LowerJawReportRenderer:

    def __init__(self) -> None:

        self.project_root = (
            Path(__file__)
            .resolve()
            .parent
        )

        self.config = RenderConfig()

        # ----------------------------------------------------
        # Input / reference
        # ----------------------------------------------------

        self.lower_stl_path = (
            self.project_root
            / "data"
            / "realistic_jaw"
            / "lower_jaw.stl"
        )

        self.lower_test_dir = (
            self.project_root
            / "data"
            / "realistic_jaw"
            / "lower_test"
        )

        # ----------------------------------------------------
        # Actual RANSAC multiway result
        # ----------------------------------------------------

        self.multiway_dir = (
            self.project_root
            / "output"
            / "realistic_multiway_ransac"
            / "lower"
        )

        # ----------------------------------------------------
        # Evaluated mesh results
        # ----------------------------------------------------

        self.poisson_mesh_path = (
            self.project_root
            / "output"
            / "realistic_mesh_validation"
            / "lower"
            / "poisson_mesh.ply"
        )

        # Final realistic BPA choice:
        #
        # B_baseline = radius factors
        # (1.5, 2.0, 3.0)
        self.bpa_mesh_path = (
            self.project_root
            / "output"
            / "realistic_bpa_parameter_study"
            / "lower"
            / "B_baseline.ply"
        )

        # ----------------------------------------------------
        # Report assets
        # ----------------------------------------------------

        self.output_dir = (
            self.project_root
            / "assets"
            / "marvin"
            / "lower"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Colors
        # ----------------------------------------------------

        self.fragment_colors = [
            [0.90, 0.28, 0.28],  # red
            [0.24, 0.58, 0.94],  # blue
            [0.29, 0.72, 0.35],  # green
            [0.86, 0.66, 0.19],  # orange
        ]

        self.original_mesh_color = [
            0.82,
            0.82,
            0.82,
        ]

        self.bpa_mesh_color = [
            0.72,
            0.84,
            0.96,
        ]

        self.poisson_mesh_color = [
            0.82,
            0.82,
            0.82,
        ]

    # ========================================================
    # MAIN
    # ========================================================

    def render_all(self) -> None:

        print()
        print("=" * 80)
        print(
            "LOWER JAW REPORT VISUALIZATION"
        )
        print("=" * 80)

        # ----------------------------------------------------
        # Load actual experiment data
        # ----------------------------------------------------

        original_mesh = (
            self.load_mesh(
                self.lower_stl_path
            )
        )

        displaced_fragments = (
            self.load_displaced_fragments()
        )

        registered_fragments = (
            self.load_registered_fragments()
        )

        poisson_mesh = (
            self.load_mesh(
                self.poisson_mesh_path
            )
        )

        bpa_mesh = (
            self.load_mesh(
                self.bpa_mesh_path
            )
        )

        # ----------------------------------------------------
        # Output paths
        # ----------------------------------------------------

        original_path = (
            self.output_dir
            / "lower_01_original_stl.png"
        )

        displaced_path = (
            self.output_dir
            / "lower_02_displaced_fragments.png"
        )

        registered_path = (
            self.output_dir
            / "lower_03_registered_fragments.png"
        )

        bpa_path = (
            self.output_dir
            / "lower_04_bpa_mesh.png"
        )

        poisson_path = (
            self.output_dir
            / "lower_05a_poisson.png"
        )

        bpa_comparison_path = (
            self.output_dir
            / "lower_05b_bpa.png"
        )

        comparison_path = (
            self.output_dir
            / "lower_05_poisson_vs_bpa.png"
        )

        overview_path = (
            self.output_dir
            / "lower_06_pipeline_overview.png"
        )

        # ----------------------------------------------------
        # Render
        # ----------------------------------------------------

        self.render_mesh(
            mesh=original_mesh,
            output_path=original_path,
            color=self.original_mesh_color,
        )

        self.render_point_clouds(
            clouds=displaced_fragments,
            output_path=displaced_path,
            point_size=2.5,
        )

        self.render_point_clouds(
            clouds=registered_fragments,
            output_path=registered_path,
            point_size=2.5,
        )

        self.render_mesh(
            mesh=bpa_mesh,
            output_path=bpa_path,
            color=self.bpa_mesh_color,
        )

        self.render_mesh(
            mesh=poisson_mesh,
            output_path=poisson_path,
            color=self.poisson_mesh_color,
        )

        self.render_mesh(
            mesh=bpa_mesh,
            output_path=bpa_comparison_path,
            color=self.bpa_mesh_color,
        )

        # ----------------------------------------------------
        # Combined figures
        # ----------------------------------------------------

        self.create_side_by_side(
            left_path=poisson_path,
            right_path=bpa_comparison_path,
            output_path=comparison_path,
        )

        self.create_2x2_overview(
            top_left=original_path,
            top_right=displaced_path,
            bottom_left=registered_path,
            bottom_right=bpa_path,
            output_path=overview_path,
        )

        print()
        print("=" * 80)
        print("DONE")
        print("=" * 80)

        print()
        print(
            "Final registration data:"
        )
        print(
            self.multiway_dir
        )

        print()
        print(
            "Final BPA mesh:"
        )
        print(
            self.bpa_mesh_path
        )

        print()
        print(
            "Figures saved to:"
        )
        print(
            self.output_dir
        )

    # ========================================================
    # LOADERS
    # ========================================================

    def load_mesh(
        self,
        path: Path,
    ) -> o3d.geometry.TriangleMesh:

        if not path.exists():
            raise FileNotFoundError(
                f"Mesh not found:\n{path}"
            )

        print(
            f"Loading mesh:\n"
            f"  {path}"
        )

        mesh = (
            o3d.io.read_triangle_mesh(
                str(path)
            )
        )

        if mesh.is_empty():
            raise RuntimeError(
                f"Could not load mesh:\n{path}"
            )

        mesh.compute_vertex_normals()

        print(
            f"  Vertices:  "
            f"{len(mesh.vertices):,}"
        )

        print(
            f"  Triangles: "
            f"{len(mesh.triangles):,}"
        )

        return mesh

    def load_cloud(
        self,
        path: Path,
    ) -> o3d.geometry.PointCloud:

        if not path.exists():
            raise FileNotFoundError(
                f"Point cloud not found:\n{path}"
            )

        print(
            f"Loading point cloud:\n"
            f"  {path}"
        )

        cloud = (
            o3d.io.read_point_cloud(
                str(path)
            )
        )

        if cloud.is_empty():
            raise RuntimeError(
                f"Could not load point cloud:\n{path}"
            )

        print(
            f"  Points: "
            f"{len(cloud.points):,}"
        )

        return cloud

    # ========================================================
    # DISPLACED FRAGMENTS
    # ========================================================

    def load_displaced_fragments(
        self,
    ) -> list[
        o3d.geometry.PointCloud
    ]:

        print()
        print(
            "Loading artificially displaced fragments..."
        )

        fragments = []

        for index in range(4):

            path = (
                self.lower_test_dir
                / f"fragment_{index:02d}.ply"
            )

            cloud = self.load_cloud(
                path
            )

            cloud.paint_uniform_color(
                self.fragment_colors[
                    index
                ]
            )

            fragments.append(
                cloud
            )

        return fragments

    # ========================================================
    # ACTUAL REGISTERED FRAGMENTS
    # ========================================================

    def load_registered_fragments(
        self,
    ) -> list[
        o3d.geometry.PointCloud
    ]:

        """
        Important:
        These are loaded directly from the output generated by
        realistic_multiway_ransac.py.

        No ground-truth poses are used here.
        """

        print()
        print(
            "Loading ACTUAL registered RANSAC fragments..."
        )

        fragments = []

        for index in range(4):

            path = (
                self.multiway_dir
                / (
                    f"registered_fragment_"
                    f"{index:02d}.ply"
                )
            )

            cloud = self.load_cloud(
                path
            )

            cloud.paint_uniform_color(
                self.fragment_colors[
                    index
                ]
            )

            fragments.append(
                cloud
            )

        return fragments

    # ========================================================
    # MESH RENDERING
    # ========================================================

    def render_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        output_path: Path,
        color,
    ) -> None:

        mesh_copy = copy.deepcopy(
            mesh
        )

        mesh_copy.paint_uniform_color(
            color
        )

        mesh_copy.compute_vertex_normals()

        self.render_geometries(
            geometries=[
                mesh_copy
            ],
            output_path=output_path,
            point_size=(
                self.config.point_size
            ),
        )

    # ========================================================
    # POINT CLOUD RENDERING
    # ========================================================

    def render_point_clouds(
        self,
        clouds: list[
            o3d.geometry.PointCloud
        ],
        output_path: Path,
        point_size: float,
    ) -> None:

        self.render_geometries(
            geometries=clouds,
            output_path=output_path,
            point_size=point_size,
        )

    # ========================================================
    # GENERIC OPEN3D RENDERER
    # ========================================================

    def render_geometries(
        self,
        geometries: Iterable,
        output_path: Path,
        point_size: float,
    ) -> None:

        geometries = list(
            geometries
        )

        if not geometries:
            raise ValueError(
                "No geometries supplied."
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print("-" * 80)

        print(
            f"Rendering:"
        )

        print(
            f"  {output_path}"
        )

        visualizer = (
            o3d.visualization.Visualizer()
        )

        created = (
            visualizer.create_window(
                window_name=(
                    "DentaVision Lower Jaw Render"
                ),
                width=int(
                    self.config.width
                ),
                height=int(
                    self.config.height
                ),
                left=50,
                top=50,
                visible=True,
            )
        )

        if not created:
            raise RuntimeError(
                "Open3D visualization "
                "window could not be created."
            )

        try:

            # -----------------------------------------------
            # Geometry
            # -----------------------------------------------

            for index, geometry in enumerate(
                geometries
            ):

                visualizer.add_geometry(
                    geometry,
                    reset_bounding_box=(
                        index == 0
                    ),
                )

            # -----------------------------------------------
            # Rendering options
            # -----------------------------------------------

            render_option = (
                visualizer
                .get_render_option()
            )

            render_option.background_color = (
                np.asarray(
                    self.config.background_color,
                    dtype=float,
                )
            )

            render_option.point_size = float(
                point_size
            )

            render_option.mesh_show_back_face = (
                True
            )

            render_option.light_on = True

            # -----------------------------------------------
            # Initialize renderer
            # -----------------------------------------------

            for _ in range(10):

                visualizer.poll_events()
                visualizer.update_renderer()

                time.sleep(
                    0.01
                )

            # -----------------------------------------------
            # Camera
            # -----------------------------------------------

            bounding_box = (
                self.get_combined_bbox(
                    geometries
                )
            )

            self.apply_camera(
                visualizer,
                bounding_box,
            )

            # -----------------------------------------------
            # Render enough frames for Windows/OpenGL
            # -----------------------------------------------

            for _ in range(20):

                visualizer.poll_events()
                visualizer.update_renderer()

                time.sleep(
                    0.02
                )

            # -----------------------------------------------
            # Screenshot
            # -----------------------------------------------

            success = (
                visualizer
                .capture_screen_image(
                    str(
                        output_path
                    ),
                    do_render=True,
                )
            )

            if success is False:
                raise RuntimeError(
                    "Screenshot capture failed:\n"
                    f"{output_path}"
                )

            print(
                f"Saved:"
            )

            print(
                f"  {output_path}"
            )

        finally:

            visualizer.destroy_window()

    # ========================================================
    # CAMERA
    # ========================================================

    def get_combined_bbox(
        self,
        geometries,
    ) -> (
        o3d.geometry
        .AxisAlignedBoundingBox
    ):

        mins = []
        maxs = []

        for geometry in geometries:

            bbox = (
                geometry
                .get_axis_aligned_bounding_box()
            )

            mins.append(
                np.asarray(
                    bbox.min_bound
                )
            )

            maxs.append(
                np.asarray(
                    bbox.max_bound
                )
            )

        minimum = np.min(
            np.vstack(
                mins
            ),
            axis=0,
        )

        maximum = np.max(
            np.vstack(
                maxs
            ),
            axis=0,
        )

        return (
            o3d.geometry
            .AxisAlignedBoundingBox(
                min_bound=minimum,
                max_bound=maximum,
            )
        )

    def apply_camera(
        self,
        visualizer:
            o3d.visualization.Visualizer,
        bbox:
            o3d.geometry.AxisAlignedBoundingBox,
    ) -> None:

        view_control = (
            visualizer
            .get_view_control()
        )

        center = (
            bbox.get_center()
        )

        front = self.normalize(
            np.asarray(
                self.config.front,
                dtype=float,
            )
        )

        up = self.normalize(
            np.asarray(
                self.config.up,
                dtype=float,
            )
        )

        view_control.set_lookat(
            center.tolist()
        )

        view_control.set_front(
            front.tolist()
        )

        view_control.set_up(
            up.tolist()
        )

        view_control.set_zoom(
            float(
                self.config.zoom
            )
        )

    @staticmethod
    def normalize(
        vector: np.ndarray,
    ) -> np.ndarray:

        length = float(
            np.linalg.norm(
                vector
            )
        )

        if length == 0:
            raise ValueError(
                "Camera vector has zero length."
            )

        return (
            vector
            / length
        )

    # ========================================================
    # IMAGE CROPPING
    # ========================================================

    @staticmethod
    def crop_background(
        image: Image.Image,
        padding: int = 25,
    ) -> Image.Image:

        """
        Automatically removes large white / nearly white
        borders from report screenshots.
        """

        image = image.convert(
            "RGB"
        )

        array = np.asarray(
            image
        )

        # Geometry is anything sufficiently darker
        # than the white background.
        mask = np.any(
            array < 245,
            axis=2,
        )

        coords = np.argwhere(
            mask
        )

        if len(coords) == 0:
            return image

        y_min, x_min = (
            coords.min(
                axis=0
            )
        )

        y_max, x_max = (
            coords.max(
                axis=0
            )
        )

        x_min = max(
            0,
            int(x_min) - padding,
        )

        y_min = max(
            0,
            int(y_min) - padding,
        )

        x_max = min(
            image.width,
            int(x_max) + padding,
        )

        y_max = min(
            image.height,
            int(y_max) + padding,
        )

        return image.crop(
            (
                x_min,
                y_min,
                x_max,
                y_max,
            )
        )

    # ========================================================
    # POISSON VS BPA
    # ========================================================

    def create_side_by_side(
        self,
        left_path: Path,
        right_path: Path,
        output_path: Path,
    ) -> None:

        left = Image.open(
            left_path
        ).convert(
            "RGB"
        )

        right = Image.open(
            right_path
        ).convert(
            "RGB"
        )

        left = self.crop_background(
            left
        )

        right = self.crop_background(
            right
        )

        cell_width = 750
        cell_height = 600

        left.thumbnail(
            (
                cell_width - 50,
                cell_height - 50,
            )
        )

        right.thumbnail(
            (
                cell_width - 50,
                cell_height - 50,
            )
        )

        canvas = Image.new(
            "RGB",
            (
                cell_width * 2,
                cell_height,
            ),
            "white",
        )

        left_x = (
            cell_width
            - left.width
        ) // 2

        left_y = (
            cell_height
            - left.height
        ) // 2

        right_x = (
            cell_width
            + (
                cell_width
                - right.width
            )
            // 2
        )

        right_y = (
            cell_height
            - right.height
        ) // 2

        canvas.paste(
            left,
            (
                left_x,
                left_y,
            ),
        )

        canvas.paste(
            right,
            (
                right_x,
                right_y,
            ),
        )

        canvas.save(
            output_path,
            dpi=(
                300,
                300,
            ),
        )

        print()
        print(
            f"Saved comparison:"
        )

        print(
            f"  {output_path}"
        )

    # ========================================================
    # 2 x 2 PIPELINE OVERVIEW
    # ========================================================

    def create_2x2_overview(
        self,
        top_left: Path,
        top_right: Path,
        bottom_left: Path,
        bottom_right: Path,
        output_path: Path,
    ) -> None:

        image_paths = [
            top_left,
            top_right,
            bottom_left,
            bottom_right,
        ]

        images = []

        for path in image_paths:

            image = Image.open(
                path
            ).convert(
                "RGB"
            )

            image = self.crop_background(
                image
            )

            images.append(
                image
            )

        cell_width = 700
        cell_height = 520

        prepared_images = []

        for image in images:

            image.thumbnail(
                (
                    cell_width - 50,
                    cell_height - 50,
                )
            )

            prepared_images.append(
                image
            )

        canvas = Image.new(
            "RGB",
            (
                cell_width * 2,
                cell_height * 2,
            ),
            "white",
        )

        positions = [
            (
                0,
                0,
            ),
            (
                cell_width,
                0,
            ),
            (
                0,
                cell_height,
            ),
            (
                cell_width,
                cell_height,
            ),
        ]

        for image, (
            base_x,
            base_y,
        ) in zip(
            prepared_images,
            positions,
        ):

            x = (
                base_x
                + (
                    cell_width
                    - image.width
                )
                // 2
            )

            y = (
                base_y
                + (
                    cell_height
                    - image.height
                )
                // 2
            )

            canvas.paste(
                image,
                (
                    x,
                    y,
                ),
            )

        canvas.save(
            output_path,
            dpi=(
                300,
                300,
            ),
        )

        print()
        print(
            f"Saved overview:"
        )

        print(
            f"  {output_path}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:

    renderer = (
        LowerJawReportRenderer()
    )

    renderer.render_all()


if __name__ == "__main__":
    main()