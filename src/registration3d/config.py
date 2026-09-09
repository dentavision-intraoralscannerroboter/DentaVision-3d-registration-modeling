from dataclasses import dataclass


@dataclass(frozen=True)
class RegistrationConfig:

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

    voxel_size: float = 0.5

    normal_radius_factor: float = 2.0
    max_nn: int = 30

    # ---------------------------------------------------------
    # Local ICP
    # ---------------------------------------------------------

    correspondence_distance_factor: float = 2.0
    max_iterations: int = 100

    # ---------------------------------------------------------
    # FPFH
    # ---------------------------------------------------------

    fpfh_radius_factor: float = 5.0
    fpfh_max_nn: int = 100

    # ---------------------------------------------------------
    # RANSAC
    # ---------------------------------------------------------

    ransac_distance_factor: float = 1.5
    ransac_n: int = 3
    ransac_max_iterations: int = 100_000
    ransac_confidence: float = 0.999

    # ---------------------------------------------------------
    # Fast Global Registration
    # ---------------------------------------------------------

    fgr_distance_factor: float = 0.5

    # ---------------------------------------------------------
    # ICP refinement after global registration
    # ---------------------------------------------------------

    refinement_distance_factor: float = 0.4

    # ---------------------------------------------------------
    # Einheitliche Evaluation aller Algorithmen
    # ---------------------------------------------------------

    evaluation_distance_factor: float = 2.0

    # ---------------------------------------------------------
    # Berechnete Parameter
    # ---------------------------------------------------------

    @property
    def normal_radius(self) -> float:
        return (
            self.voxel_size
            * self.normal_radius_factor
        )

    @property
    def max_correspondence_distance(self) -> float:
        return (
            self.voxel_size
            * self.correspondence_distance_factor
        )

    @property
    def fpfh_radius(self) -> float:
        return (
            self.voxel_size
            * self.fpfh_radius_factor
        )

    @property
    def ransac_distance(self) -> float:
        return (
            self.voxel_size
            * self.ransac_distance_factor
        )

    @property
    def fgr_distance(self) -> float:
        return (
            self.voxel_size
            * self.fgr_distance_factor
        )

    @property
    def refinement_distance(self) -> float:
        return (
            self.voxel_size
            * self.refinement_distance_factor
        )

    @property
    def evaluation_distance(self) -> float:
        return (
            self.voxel_size
            * self.evaluation_distance_factor
        )