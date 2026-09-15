from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/realistic_jaw")

UPPER_TEST_DIR = (
    DATA_DIR
    / "upper_test"
)

REGISTERED_DIR = Path(
    "output/realistic_multiway_ransac/upper"
)

# Poisson result from the realistic mesh validation
POISSON_MESH_PATH = Path(
    "output/realistic_mesh_validation/"
    "upper/poisson_mesh.ply"
)

# FINAL realistic BPA configuration:
# radius factors = (1.5, 2.0, 3.0)
BPA_MESH_PATH = Path(
    "output/realistic_bpa_parameter_study/"
    "upper/B_baseline.ply"
)

OUTPUT_DIR = Path(
    "output/report_figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# COLORS
# ============================================================

FRAGMENT_COLORS = [
    [0.85, 0.25, 0.25],  # red
    [0.20, 0.55, 0.85],  # blue
    [0.25, 0.70, 0.35],  # green
    [0.85, 0.65, 0.20],  # yellow
]

ORIGINAL_MESH_COLOR = [
    0.80,
    0.80,
    0.80,
]

POINT_CLOUD_COLOR = [
    0.20,
    0.50,
    0.80,
]

BPA_COLOR = [
    0.70,
    0.82,
    0.95,
]

POISSON_COLOR = [
    0.82,
    0.82,
    0.82,
]


# ============================================================
# LOADING
# ============================================================

def load_cloud(
    path: Path,
) -> o3d.geometry.PointCloud:

    print(
        f"Loading point cloud: {path}"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    cloud = o3d.io.read_point_cloud(
        str(path)
    )

    if cloud.is_empty():
        raise RuntimeError(
            f"Could not load point cloud: {path}"
        )

    print(
        f"  -> {len(cloud.points):,} points"
    )

    return cloud


def load_mesh(
    path: Path,
) -> o3d.geometry.TriangleMesh:

    print(
        f"Loading mesh: {path}"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    mesh = o3d.io.read_triangle_mesh(
        str(path)
    )

    if mesh.is_empty():
        raise RuntimeError(
            f"Could not load mesh: {path}"
        )

    mesh.compute_vertex_normals()

    print(
        f"  -> {len(mesh.vertices):,} vertices, "
        f"{len(mesh.triangles):,} triangles"
    )

    return mesh


# ============================================================
# CAMERA / BOUNDS
# ============================================================

def get_combined_center(
    geometries,
) -> np.ndarray:

    min_bounds = []
    max_bounds = []

    for geometry in geometries:

        bbox = (
            geometry
            .get_axis_aligned_bounding_box()
        )

        min_bounds.append(
            np.asarray(
                bbox.min_bound
            )
        )

        max_bounds.append(
            np.asarray(
                bbox.max_bound
            )
        )

    minimum = np.min(
        np.vstack(min_bounds),
        axis=0,
    )

    maximum = np.max(
        np.vstack(max_bounds),
        axis=0,
    )

    return (
        minimum + maximum
    ) / 2.0


# ============================================================
# OPEN3D RENDERING
# ============================================================

def render_scene(
    geometries,
    output_path: Path,
    width: int = 1400,
    height: int = 1000,
    point_size: float = 3.0,
    zoom: float = 0.72,
):

    """
    Render one Open3D scene and save it as PNG.

    A visible window is deliberately used because this is
    generally more reliable with Open3D on Windows.
    """

    print()
    print("=" * 70)
    print(
        f"Rendering: {output_path.name}"
    )
    print("=" * 70)

    # Prevent the previous tuple problem.
    width = int(width)
    height = int(height)

    visualizer = (
        o3d.visualization.Visualizer()
    )

    success = visualizer.create_window(
        window_name=(
            "DentaVision Report Renderer"
        ),
        width=width,
        height=height,
        left=50,
        top=50,
        visible=True,
    )

    if not success:
        raise RuntimeError(
            "Could not create Open3D window."
        )

    # --------------------------------------------------------
    # Add geometry
    # --------------------------------------------------------

    for index, geometry in enumerate(
        geometries
    ):

        visualizer.add_geometry(
            geometry,
            reset_bounding_box=(
                index == 0
            ),
        )

    # --------------------------------------------------------
    # Render settings
    # --------------------------------------------------------

    render_option = (
        visualizer.get_render_option()
    )

    render_option.background_color = (
        np.array(
            [
                1.0,
                1.0,
                1.0,
            ]
        )
    )

    render_option.point_size = float(
        point_size
    )

    render_option.mesh_show_back_face = (
        True
    )

    # Let Open3D initialize the scene.
    for _ in range(10):

        visualizer.poll_events()
        visualizer.update_renderer()

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    center = get_combined_center(
        geometries
    )

    view_control = (
        visualizer.get_view_control()
    )

    view_control.set_lookat(
        center.tolist()
    )

    # Slightly oblique occlusal view.
    view_control.set_front(
        [
            0.0,
            -0.25,
            -1.0,
        ]
    )

    view_control.set_up(
        [
            0.0,
            -1.0,
            0.25,
        ]
    )

    view_control.set_zoom(
        float(zoom)
    )

    # Give Windows/OpenGL some render cycles.
    for _ in range(20):

        visualizer.poll_events()
        visualizer.update_renderer()

    # --------------------------------------------------------
    # Screenshot
    # --------------------------------------------------------

    success = (
        visualizer.capture_screen_image(
            str(output_path),
            do_render=True,
        )
    )

    visualizer.destroy_window()

    if success is False:
        raise RuntimeError(
            f"Screenshot failed: {output_path}"
        )

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# 1. ORIGINAL STL MODEL
# ============================================================

def render_original_mesh():

    mesh = load_mesh(
        DATA_DIR
        / "upper_jaw.stl"
    )

    mesh.paint_uniform_color(
        ORIGINAL_MESH_COLOR
    )

    render_scene(
        geometries=[
            mesh
        ],
        output_path=(
            OUTPUT_DIR
            / "01_original_upper_jaw.png"
        ),
        zoom=0.72,
    )


# ============================================================
# 2. SAMPLED REFERENCE POINT CLOUD
# ============================================================

def render_reference_cloud():

    cloud = load_cloud(
        UPPER_TEST_DIR
        / "reference_cloud.ply"
    )

    cloud.paint_uniform_color(
        POINT_CLOUD_COLOR
    )

    render_scene(
        geometries=[
            cloud
        ],
        output_path=(
            OUTPUT_DIR
            / "02_sampled_point_cloud.png"
        ),
        point_size=2.5,
        zoom=0.72,
    )


# ============================================================
# 3. OVERLAPPING FRAGMENTS IN ORIGINAL COORDINATES
# ============================================================

def render_overlapping_fragments():

    geometries = []

    for index in range(4):

        cloud = load_cloud(
            UPPER_TEST_DIR
            / (
                f"fragment_"
                f"{index:02d}_world.ply"
            )
        )

        cloud.paint_uniform_color(
            FRAGMENT_COLORS[
                index
            ]
        )

        geometries.append(
            cloud
        )

    render_scene(
        geometries=geometries,
        output_path=(
            OUTPUT_DIR
            / "03_overlapping_fragments.png"
        ),
        point_size=3.0,
        zoom=0.72,
    )


# ============================================================
# 4. ARTIFICIALLY DISPLACED FRAGMENTS
# ============================================================

def render_displaced_fragments():

    geometries = []

    for index in range(4):

        cloud = load_cloud(
            UPPER_TEST_DIR
            / (
                f"fragment_"
                f"{index:02d}.ply"
            )
        )

        cloud.paint_uniform_color(
            FRAGMENT_COLORS[
                index
            ]
        )

        geometries.append(
            cloud
        )

    render_scene(
        geometries=geometries,
        output_path=(
            OUTPUT_DIR
            / "04_displaced_fragments.png"
        ),
        point_size=3.0,
        zoom=0.62,
    )


# ============================================================
# 5. RANSAC + ICP REGISTERED FRAGMENTS
# ============================================================

def render_registered_fragments():

    geometries = []

    for index in range(4):

        cloud = load_cloud(
            REGISTERED_DIR
            / (
                f"registered_fragment_"
                f"{index:02d}.ply"
            )
        )

        cloud.paint_uniform_color(
            FRAGMENT_COLORS[
                index
            ]
        )

        geometries.append(
            cloud
        )

    render_scene(
        geometries=geometries,
        output_path=(
            OUTPUT_DIR
            / "05_registered_fragments.png"
        ),
        point_size=3.0,
        zoom=0.72,
    )


# ============================================================
# 6. FINAL REALISTIC BALL PIVOTING MESH
# ============================================================

def render_ball_pivoting_mesh():

    """
    Uses the realistic BPA parameter-study winner:

        B_baseline
        radius factors = (1.5, 2.0, 3.0)

    This replaces the former A_small visualization.
    """

    mesh = load_mesh(
        BPA_MESH_PATH
    )

    mesh.paint_uniform_color(
        BPA_COLOR
    )

    render_scene(
        geometries=[
            mesh
        ],
        output_path=(
            OUTPUT_DIR
            / "06_ball_pivoting_mesh.png"
        ),
        zoom=0.72,
    )


# ============================================================
# 7. POISSON MESH
# ============================================================

def render_poisson_mesh():

    mesh = load_mesh(
        POISSON_MESH_PATH
    )

    mesh.paint_uniform_color(
        POISSON_COLOR
    )

    render_scene(
        geometries=[
            mesh
        ],
        output_path=(
            OUTPUT_DIR
            / "07_poisson_mesh.png"
        ),
        zoom=0.72,
    )


# ============================================================
# WHITE-BORDER CROPPING
# ============================================================

def crop_white_border(
    image: Image.Image,
    padding: int = 20,
) -> Image.Image:

    image = image.convert(
        "RGB"
    )

    array = np.asarray(
        image
    )

    # Anything darker than nearly white counts
    # as actual image content.
    mask = np.any(
        array < 245,
        axis=2,
    )

    coordinates = np.argwhere(
        mask
    )

    if len(coordinates) == 0:

        print(
            "Warning: image appears completely white."
        )

        return image

    y_min, x_min = (
        coordinates.min(
            axis=0
        )
    )

    y_max, x_max = (
        coordinates.max(
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


# ============================================================
# REPORT WORKFLOW FIGURE
# ============================================================

def create_workflow_panel():

    """
    Creates the compact report workflow:

    Original STL
        ->
    Artificially displaced fragments
        ->
    RANSAC + ICP registration
        ->
    Ball Pivoting reconstruction
    """

    panels = [
        (
            "01_original_upper_jaw.png",
            "(a) Original STL model",
        ),
        (
            "04_displaced_fragments.png",
            "(b) Artificially displaced fragments",
        ),
        (
            "05_registered_fragments.png",
            "(c) FPFH + RANSAC + ICP registration",
        ),
        (
            "06_ball_pivoting_mesh.png",
            "(d) Ball Pivoting reconstruction",
        ),
    ]

    cell_width = 700
    cell_height = 540

    panel_images = []

    for filename, label in panels:

        image_path = (
            OUTPUT_DIR
            / filename
        )

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        image = crop_white_border(
            image
        )

        image.thumbnail(
            (
                cell_width - 40,
                cell_height - 90,
            )
        )

        canvas = Image.new(
            "RGB",
            (
                cell_width,
                cell_height,
            ),
            "white",
        )

        x = (
            cell_width
            - image.width
        ) // 2

        y = 10

        canvas.paste(
            image,
            (
                x,
                y,
            ),
        )

        draw = ImageDraw.Draw(
            canvas
        )

        draw.text(
            (
                20,
                cell_height - 50,
            ),
            label,
            fill="black",
        )

        panel_images.append(
            canvas
        )

    final_image = Image.new(
        "RGB",
        (
            cell_width * 2,
            cell_height * 2,
        ),
        "white",
    )

    for index, panel in enumerate(
        panel_images
    ):

        column = (
            index % 2
        )

        row = (
            index // 2
        )

        final_image.paste(
            panel,
            (
                column
                * cell_width,

                row
                * cell_height,
            ),
        )

    output_path = (
        OUTPUT_DIR
        / "realistic_validation_workflow.png"
    )

    final_image.save(
        output_path,
        dpi=(
            300,
            300,
        ),
    )

    print()
    print(
        f"Saved workflow figure: "
        f"{output_path}"
    )


# ============================================================
# POISSON VS BALL PIVOTING FIGURE
# ============================================================

def create_reconstruction_comparison():

    poisson = Image.open(
        OUTPUT_DIR
        / "07_poisson_mesh.png"
    ).convert(
        "RGB"
    )

    bpa = Image.open(
        OUTPUT_DIR
        / "06_ball_pivoting_mesh.png"
    ).convert(
        "RGB"
    )

    poisson = crop_white_border(
        poisson
    )

    bpa = crop_white_border(
        bpa
    )

    cell_width = 750
    cell_height = 580

    final_image = Image.new(
        "RGB",
        (
            cell_width * 2,
            cell_height,
        ),
        "white",
    )

    entries = [
        (
            poisson,
            "(a) Poisson Surface Reconstruction",
        ),
        (
            bpa,
            (
                "(b) Ball Pivoting "
                "(radius factors 1.5, 2.0, 3.0)"
            ),
        ),
    ]

    draw = ImageDraw.Draw(
        final_image
    )

    for index, (
        image,
        label,
    ) in enumerate(
        entries
    ):

        image.thumbnail(
            (
                cell_width - 40,
                cell_height - 90,
            )
        )

        x_offset = (
            index
            * cell_width
        )

        x = (
            x_offset
            + (
                cell_width
                - image.width
            )
            // 2
        )

        y = 10

        final_image.paste(
            image,
            (
                x,
                y,
            ),
        )

        draw.text(
            (
                x_offset + 20,
                cell_height - 50,
            ),
            label,
            fill="black",
        )

    output_path = (
        OUTPUT_DIR
        / "poisson_vs_ball_pivoting.png"
    )

    final_image.save(
        output_path,
        dpi=(
            300,
            300,
        ),
    )

    print(
        f"Saved reconstruction comparison: "
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print(
        "DENTAVISION REPORT VISUALIZATION GENERATOR"
    )
    print("=" * 75)

    print()
    print(
        "Final realistic registration method:"
    )
    print(
        "FPFH + RANSAC + Point-to-Plane ICP"
    )

    print()
    print(
        "Final realistic BPA radius factors:"
    )
    print(
        "(1.5, 2.0, 3.0)"
    )

    # --------------------------------------------------------
    # Individual figures
    # --------------------------------------------------------

    render_original_mesh()

    render_reference_cloud()

    render_overlapping_fragments()

    render_displaced_fragments()

    render_registered_fragments()

    render_ball_pivoting_mesh()

    render_poisson_mesh()

    # --------------------------------------------------------
    # Combined report figures
    # --------------------------------------------------------

    create_workflow_panel()

    create_reconstruction_comparison()

    print()
    print("=" * 75)
    print("DONE")
    print("=" * 75)

    print()
    print(
        "Figures saved to:"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    print()
    print(
        "Most relevant report figures:"
    )

    print(
        "  realistic_validation_workflow.png"
    )

    print(
        "  poisson_vs_ball_pivoting.png"
    )


if __name__ == "__main__":
    main()