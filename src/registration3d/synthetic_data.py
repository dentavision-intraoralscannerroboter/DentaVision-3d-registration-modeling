from dataclasses import dataclass
import copy

import numpy as np
import open3d as o3d

from registration3d.experiment_config import (
    ExperimentScenario,
)


@dataclass
class SyntheticDataset:
    source: o3d.geometry.PointCloud
    target: o3d.geometry.PointCloud
    ground_truth: np.ndarray


class SyntheticDentalDataGenerator:

    def __init__(
        self,
        number_of_points: int = 20_000,
    ):
        self.number_of_points = number_of_points
        self.mesh = self._create_dental_like_mesh()

    # ---------------------------------------------------------
    # Öffentliche Methode
    # ---------------------------------------------------------

    def generate(
    self,
    scenario: ExperimentScenario,
    seed: int,
    ) -> SyntheticDataset:

    # NumPy-Zufallszahlen reproduzierbar machen
     rng = np.random.default_rng(seed)

    # Open3D-Zufallszahlen reproduzierbar machen
     o3d.utility.random.seed(seed)

    # Zwei unabhängige, aber reproduzierbare
    # Scans desselben Modells erzeugen
     source_full = self.mesh.sample_points_uniformly(
        number_of_points=self.number_of_points
     )

     target_full = self.mesh.sample_points_uniformly(
        number_of_points=self.number_of_points
    )

     source, target = self._create_partial_clouds(
        source_full,
        target_full,
        scenario.overlap_fraction,
     )

     ground_truth = self._create_transform(
        rotation_degrees=scenario.rotation_degrees,
        translation_magnitude=scenario.translation_magnitude,
     )

     target_transformed = copy.deepcopy(
        target
     )

     target_transformed.transform(
        ground_truth
     )

     source = self._add_noise(
        source,
        scenario.noise_std,
        rng,
     )

     target_transformed = self._add_noise(
        target_transformed,
        scenario.noise_std,
        rng,
     )

     return SyntheticDataset(
        source=source,
        target=target_transformed,
        ground_truth=ground_truth,
     )

    # ---------------------------------------------------------
    # Dentalartiges Testmodell
    # ---------------------------------------------------------

    @staticmethod
    def _create_dental_like_mesh() -> (
        o3d.geometry.TriangleMesh
    ):

        combined_mesh = o3d.geometry.TriangleMesh()

        # Unterschiedliche Zahnformen.
        tooth_shapes = [
            (2.3, 2.1, 5.2),
            (2.1, 2.0, 5.7),
            (1.9, 1.8, 6.4),
            (1.8, 1.7, 6.8),
            (1.7, 1.6, 7.1),
            (1.8, 1.7, 6.7),
            (1.9, 1.9, 6.1),
            (2.0, 2.1, 5.8),
            (2.2, 2.3, 5.5),
            (2.5, 2.4, 5.0),
            (2.4, 2.6, 4.8),
        ]

        x_positions = np.linspace(
            -18.0,
            18.0,
            len(tooth_shapes),
        )

        for index, (
            x,
            shape,
        ) in enumerate(
            zip(
                x_positions,
                tooth_shapes,
            )
        ):

            scale_x, scale_y, scale_z = shape

            tooth = (
                o3d.geometry.TriangleMesh
                .create_sphere(
                    radius=1.0,
                    resolution=16,
                )
            )

            # Kugel -> Ellipsoid
            vertices = np.asarray(
                tooth.vertices
            ).copy()

            vertices[:, 0] *= scale_x
            vertices[:, 1] *= scale_y
            vertices[:, 2] *= scale_z

            tooth.vertices = (
                o3d.utility.Vector3dVector(
                    vertices
                )
            )

            # Leichter Zahnbogen.
            y = 0.018 * (x ** 2)

            # Unterschiedliche minimale Rotationen,
            # damit das Modell nicht perfekt symmetrisch ist.
            z_rotation = np.radians(
                (index - 5) * 1.8
            )

            rotation = (
                o3d.geometry
                .get_rotation_matrix_from_xyz(
                    [
                        0.0,
                        0.0,
                        z_rotation,
                    ]
                )
            )

            tooth.rotate(
                rotation,
                center=[0.0, 0.0, 0.0],
            )

            tooth.translate(
                [
                    x,
                    y,
                    scale_z,
                ]
            )

            combined_mesh += tooth

        # Kleine zusätzliche asymmetrische Struktur.
        marker = (
            o3d.geometry.TriangleMesh
            .create_sphere(
                radius=1.4,
                resolution=12,
            )
        )

        marker.translate(
            [
                11.5,
                4.5,
                9.0,
            ]
        )

        combined_mesh += marker

        combined_mesh.compute_vertex_normals()

        return combined_mesh

    # ---------------------------------------------------------
    # Teilwolken erzeugen
    # ---------------------------------------------------------

    @staticmethod
    def _create_partial_clouds(
        source_full: o3d.geometry.PointCloud,
        target_full: o3d.geometry.PointCloud,
        overlap_fraction: float,
    ) -> tuple[
        o3d.geometry.PointCloud,
        o3d.geometry.PointCloud,
    ]:

        if not 0.0 < overlap_fraction < 1.0:
            raise ValueError(
                "overlap_fraction muss zwischen 0 und 1 liegen."
            )

        source_points = np.asarray(
            source_full.points
        )

        target_points = np.asarray(
            target_full.points
        )

        all_x = np.concatenate(
            [
                source_points[:, 0],
                target_points[:, 0],
            ]
        )

        min_x = float(
            np.min(all_x)
        )

        max_x = float(
            np.max(all_x)
        )

        width = max_x - min_x

        # Beide Clouds decken mehr als die Hälfte ab.
        # Ihre Schnittmenge entspricht ungefähr
        # overlap_fraction des Gesamtobjekts.
        coverage_fraction = (
            1.0 + overlap_fraction
        ) / 2.0

        source_max_x = (
            min_x
            + coverage_fraction * width
        )

        target_min_x = (
            max_x
            - coverage_fraction * width
        )

        source_indices = np.where(
            source_points[:, 0]
            <= source_max_x
        )[0]

        target_indices = np.where(
            target_points[:, 0]
            >= target_min_x
        )[0]

        source = source_full.select_by_index(
            source_indices.tolist()
        )

        target = target_full.select_by_index(
            target_indices.tolist()
        )

        return source, target

    # ---------------------------------------------------------
    # Ground-Truth-Transformation
    # ---------------------------------------------------------

    @staticmethod
    def _create_transform(
        rotation_degrees: float,
        translation_magnitude: float,
    ) -> np.ndarray:

        # Feste, nicht-triviale Rotationsachse.
        axis = np.array(
            [
                0.35,
                -0.45,
                0.82,
            ],
            dtype=float,
        )

        axis /= np.linalg.norm(axis)

        axis_angle = (
            axis
            * np.radians(rotation_degrees)
        )

        rotation = (
            o3d.geometry
            .get_rotation_matrix_from_axis_angle(
                axis_angle
            )
        )

        # Feste Translationsrichtung.
        direction = np.array(
            [
                0.65,
                -0.60,
                0.46,
            ],
            dtype=float,
        )

        direction /= np.linalg.norm(
            direction
        )

        translation = (
            direction
            * translation_magnitude
        )

        transform = np.eye(4)

        transform[:3, :3] = rotation
        transform[:3, 3] = translation

        return transform

    # ---------------------------------------------------------
    # Rauschen
    # ---------------------------------------------------------

    @staticmethod
    def _add_noise(
        cloud: o3d.geometry.PointCloud,
        noise_std: float,
        rng: np.random.Generator,
    ) -> o3d.geometry.PointCloud:

        if noise_std <= 0.0:
            return copy.deepcopy(cloud)

        noisy_cloud = copy.deepcopy(cloud)

        points = np.asarray(
            noisy_cloud.points
        ).copy()

        noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=points.shape,
        )

        points += noise

        noisy_cloud.points = (
            o3d.utility.Vector3dVector(
                points
            )
        )

        return noisy_cloud