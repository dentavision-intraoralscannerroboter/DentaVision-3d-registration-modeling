from dataclasses import dataclass
import time

import numpy as np
import open3d as o3d

from registration3d.algorithms.base import (
    RegistrationAlgorithm,
)
from registration3d.algorithms.fgr_fpfh_icp import (
    FgrFPFHICP,
)
from registration3d.algorithms.ransac_fpfh_icp import (
    RansacFPFHICP,
)
from registration3d.config import (
    RegistrationConfig,
)
from registration3d.result import (
    RegistrationResult,
)
from registration3d.robust_pairwise_config import (
    RobustPairwiseConfig,
)


@dataclass
class RobustRegistrationResult(
    RegistrationResult
):
    confidence: str = "low"
    consensus_size: int = 0
    method_support_count: int = 0
    selected_candidate: str = ""

    cycle_translation_error: float | None = None
    cycle_rotation_error_degrees: float | None = None


@dataclass
class _Candidate:
    label: str
    method: str
    seed: int
    result: RegistrationResult

    cycle_translation_error: float | None = None
    cycle_rotation_error_degrees: float | None = None


@dataclass
class _Hypothesis:
    center_index: int
    supporter_indices: list[int]
    method_support_count: int
    median_rmse: float
    median_fitness: float


class RobustPairwiseRegistration(
    RegistrationAlgorithm
):

    def __init__(
        self,
        registration_config: RegistrationConfig,
        robust_config: RobustPairwiseConfig,
    ) -> None:

        self.registration_config = (
            registration_config
        )

        self.robust_config = (
            robust_config
        )

        self.fgr = (
            FgrFPFHICP(
                registration_config
            )
        )

        self.ransac = (
            RansacFPFHICP(
                registration_config
            )
        )

    @property
    def name(self) -> str:
        return (
            "Multi-Start FGR/RANSAC Selector"
        )

    # =========================================================
    # NORMALE REGISTRIERUNG
    #
    # Wird für Odometry verwendet.
    # =========================================================

    def register(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        initial_transform: np.ndarray | None = None,
    ) -> RegistrationResult:

        start_time = (
            time.perf_counter()
        )

        candidates = (
            self._generate_candidates(
                source=source,
                target=target,
                initial_transform=(
                    initial_transform
                ),
            )
        )

        valid_indices = [
            index
            for index, candidate
            in enumerate(candidates)
            if self._is_valid_candidate(
                candidate.result
            )
        ]

        print()
        print(
            "    Best Multi-Start Candidates:"
        )

        self._print_best_candidates(
            candidates=candidates,
            valid_indices=valid_indices,
        )

        return self._select_result(
            candidates=candidates,
            candidate_indices=valid_indices,
            start_time=start_time,
        )

    # =========================================================
    # LOOP-CLOSURE-REGISTRIERUNG
    #
    # Unterschied:
    #
    # Alle Kandidaten werden VOR der Cluster-Auswahl gegen
    # die aus der Odometry vorhergesagte Transformation
    # geprüft.
    # =========================================================

    def register_with_cycle_consistency(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        predicted_transform: np.ndarray,
        translation_threshold: float,
        rotation_threshold_degrees: float,
    ) -> RegistrationResult:

        start_time = (
            time.perf_counter()
        )

        candidates = (
            self._generate_candidates(
                source=source,
                target=target,
                initial_transform=None,
            )
        )

        # -----------------------------------------------------
        # Grundsätzlich gültige Kandidaten
        # -----------------------------------------------------

        valid_indices = [
            index
            for index, candidate
            in enumerate(candidates)
            if self._is_valid_candidate(
                candidate.result
            )
        ]

        # -----------------------------------------------------
        # Cycle Difference für JEDEN Kandidaten
        # -----------------------------------------------------

        cycle_valid_indices = []

        for index in valid_indices:

            candidate = (
                candidates[index]
            )

            (
                translation_error,
                rotation_error,
            ) = (
                self._transformation_difference(
                    predicted_transform,
                    candidate.result
                    .transformation,
                )
            )

            candidate.cycle_translation_error = (
                translation_error
            )

            candidate.cycle_rotation_error_degrees = (
                rotation_error
            )

            cycle_ok = (
                translation_error
                <= translation_threshold

                and

                rotation_error
                <= rotation_threshold_degrees
            )

            if cycle_ok:

                cycle_valid_indices.append(
                    index
                )

        # -----------------------------------------------------
        # Diagnose
        # -----------------------------------------------------

        print()
        print(
            "    Cycle-Consistent Candidates:"
        )

        print(
            f"      Basic valid: "
            f"{len(valid_indices)} / "
            f"{len(candidates)}"
        )

        print(
            f"      Cycle valid: "
            f"{len(cycle_valid_indices)} / "
            f"{len(valid_indices)}"
        )

        self._print_cycle_candidates(
            candidates=(
                candidates
            ),

            valid_indices=(
                valid_indices
            ),

            cycle_valid_indices=(
                cycle_valid_indices
            ),
        )

        # -----------------------------------------------------
        # Nur noch cycle-konsistente Kandidaten dürfen
        # ausgewählt werden.
        # -----------------------------------------------------

        result = (
            self._select_result(
                candidates=candidates,

                candidate_indices=(
                    cycle_valid_indices
                ),

                start_time=(
                    start_time
                ),
            )
        )

        return result

    # =========================================================
    # Kandidaten erzeugen
    # =========================================================

    def _generate_candidates(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        initial_transform: np.ndarray | None,
    ) -> list[_Candidate]:

        candidates: list[
            _Candidate
        ] = []

        # -----------------------------------------------------
        # FGR Multi-Start
        # -----------------------------------------------------

        for run_index in range(
            self.robust_config.fgr_repeats
        ):

            seed = (
                self.robust_config.base_seed
                + run_index
            )

            o3d.utility.random.seed(
                seed
            )

            result = (
                self.fgr.register(
                    source,
                    target,
                    initial_transform,
                )
            )

            result = (
                self._evaluate_candidate(
                    result,
                    source,
                    target,
                )
            )

            candidates.append(
                _Candidate(
                    label=(
                        f"FGR seed {seed}"
                    ),

                    method="FGR",

                    seed=seed,

                    result=result,
                )
            )

        # -----------------------------------------------------
        # RANSAC Multi-Start
        # -----------------------------------------------------

        for run_index in range(
            self.robust_config.ransac_repeats
        ):

            seed = (
                self.robust_config.base_seed
                + run_index
            )

            o3d.utility.random.seed(
                seed
            )

            result = (
                self.ransac.register(
                    source,
                    target,
                    initial_transform,
                )
            )

            result = (
                self._evaluate_candidate(
                    result,
                    source,
                    target,
                )
            )

            candidates.append(
                _Candidate(
                    label=(
                        f"RANSAC seed {seed}"
                    ),

                    method="RANSAC",

                    seed=seed,

                    result=result,
                )
            )

        return candidates

    # =========================================================
    # Zentrales Auswahlverfahren
    # =========================================================

    def _select_result(
        self,
        candidates: list[_Candidate],
        candidate_indices: list[int],
        start_time: float,
    ) -> RobustRegistrationResult:

        # -----------------------------------------------------
        # Keine Kandidaten übrig
        # -----------------------------------------------------

        if not candidate_indices:

            runtime = (
                time.perf_counter()
                - start_time
            )

            print()
            print(
                "    -> Kein gültiger Kandidat vorhanden."
            )

            print(
                "    -> Confidence: LOW"
            )

            return RobustRegistrationResult(
                algorithm=self.name,

                transformation=(
                    np.eye(4)
                ),

                fitness=0.0,

                inlier_rmse=float(
                    "inf"
                ),

                runtime_seconds=runtime,

                confidence="low",

                consensus_size=0,

                method_support_count=0,

                selected_candidate="NONE",
            )

        # -----------------------------------------------------
        # Cluster bilden
        # -----------------------------------------------------

        hypotheses = (
            self._create_hypotheses(
                candidates=(
                    candidates
                ),

                valid_indices=(
                    candidate_indices
                ),
            )
        )

        # -----------------------------------------------------
        # Es gibt Kandidaten, aber keinen Cluster >= 2
        # -----------------------------------------------------

        if not hypotheses:

            fallback_index = (
                self._select_best_single_candidate(
                    candidates=(
                        candidates
                    ),

                    valid_indices=(
                        candidate_indices
                    ),
                )
            )

            fallback = (
                candidates[
                    fallback_index
                ]
            )

            runtime = (
                time.perf_counter()
                - start_time
            )

            print()
            print(
                "    Kein stabiler "
                "Transformationscluster gefunden."
            )

            print(
                f"    -> Fallback: "
                f"{fallback.label}"
            )

            print(
                "    -> Confidence: LOW"
            )

            return RobustRegistrationResult(
                algorithm=self.name,

                transformation=(
                    fallback.result
                    .transformation
                    .copy()
                ),

                fitness=(
                    fallback.result
                    .fitness
                ),

                inlier_rmse=(
                    fallback.result
                    .inlier_rmse
                ),

                runtime_seconds=runtime,

                confidence="low",

                consensus_size=1,

                method_support_count=1,

                selected_candidate=(
                    fallback.label
                ),

                cycle_translation_error=(
                    fallback
                    .cycle_translation_error
                ),

                cycle_rotation_error_degrees=(
                    fallback
                    .cycle_rotation_error_degrees
                ),
            )

        # -----------------------------------------------------
        # Gewinner-Cluster
        # -----------------------------------------------------

        best_hypothesis = (
            self._select_best_hypothesis(
                hypotheses
            )
        )

        representative_index = (
            self._select_cluster_representative(
                candidates=(
                    candidates
                ),

                supporter_indices=(
                    best_hypothesis
                    .supporter_indices
                ),
            )
        )

        representative = (
            candidates[
                representative_index
            ]
        )

        consensus_size = len(
            best_hypothesis
            .supporter_indices
        )

        method_support_count = (
            best_hypothesis
            .method_support_count
        )

        if method_support_count >= 2:

            confidence = "high"

        else:

            confidence = "medium"

        runtime = (
            time.perf_counter()
            - start_time
        )

        supporter_labels = [
            candidates[index].label
            for index
            in best_hypothesis
            .supporter_indices
        ]

        # -----------------------------------------------------
        # Ausgabe
        # -----------------------------------------------------

        print()
        print(
            "    Winning Multi-Start Cluster:"
        )

        print(
            f"      Size: "
            f"{consensus_size}"
        )

        print(
            f"      Methods: "
            f"{method_support_count}"
        )

        print(
            f"      Median RMSE: "
            f"{best_hypothesis.median_rmse:.6f}"
        )

        print(
            f"      Median Fitness: "
            f"{best_hypothesis.median_fitness:.6f}"
        )

        print(
            f"      Members: "
            f"{', '.join(supporter_labels)}"
        )

        print(
            f"    -> Selected: "
            f"{representative.label}"
        )

        print(
            f"    -> Fitness: "
            f"{representative.result.fitness:.6f}"
        )

        print(
            f"    -> RMSE: "
            f"{representative.result.inlier_rmse:.6f}"
        )

        if (
            representative
            .cycle_translation_error
            is not None
        ):

            print(
                f"    -> Cycle ΔT: "
                f"{representative.cycle_translation_error:.6f}"
            )

        if (
            representative
            .cycle_rotation_error_degrees
            is not None
        ):

            print(
                f"    -> Cycle ΔR: "
                f"{representative.cycle_rotation_error_degrees:.6f}°"
            )

        print(
            f"    -> Confidence: "
            f"{confidence.upper()}"
        )

        return RobustRegistrationResult(
            algorithm=self.name,

            transformation=(
                representative.result
                .transformation
                .copy()
            ),

            fitness=(
                representative.result
                .fitness
            ),

            inlier_rmse=(
                representative.result
                .inlier_rmse
            ),

            runtime_seconds=runtime,

            confidence=confidence,

            consensus_size=(
                consensus_size
            ),

            method_support_count=(
                method_support_count
            ),

            selected_candidate=(
                representative.label
            ),

            cycle_translation_error=(
                representative
                .cycle_translation_error
            ),

            cycle_rotation_error_degrees=(
                representative
                .cycle_rotation_error_degrees
            ),
        )

    # =========================================================
    # Open3D Evaluation
    # =========================================================

    def _evaluate_candidate(
        self,
        result: RegistrationResult,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
    ) -> RegistrationResult:

        evaluation = (
            o3d.pipelines.registration
            .evaluate_registration(
                source,
                target,

                self.registration_config
                .evaluation_distance,

                result.transformation,
            )
        )

        result.fitness = float(
            evaluation.fitness
        )

        result.inlier_rmse = float(
            evaluation.inlier_rmse
        )

        return result

    # =========================================================
    # Grundfilter
    # =========================================================

    def _is_valid_candidate(
        self,
        result: RegistrationResult,
    ) -> bool:

        if not np.isfinite(
            result.fitness
        ):
            return False

        if not np.isfinite(
            result.inlier_rmse
        ):
            return False

        if (
            result.fitness
            < self.robust_config
            .minimum_fitness
        ):
            return False

        if (
            result.inlier_rmse
            > self.robust_config
            .maximum_rmse
        ):
            return False

        return True

    # =========================================================
    # Transformationscluster
    # =========================================================

    def _create_hypotheses(
        self,
        candidates: list[_Candidate],
        valid_indices: list[int],
    ) -> list[_Hypothesis]:

        hypotheses: list[
            _Hypothesis
        ] = []

        for center_index in valid_indices:

            center = (
                candidates[
                    center_index
                ]
            )

            supporters = []

            for other_index in valid_indices:

                other = (
                    candidates[
                        other_index
                    ]
                )

                (
                    translation_difference,
                    rotation_difference,
                ) = (
                    self._transformation_difference(
                        center.result
                        .transformation,

                        other.result
                        .transformation,
                    )
                )

                if (
                    translation_difference
                    <= (
                        self.robust_config
                        .agreement_translation_threshold
                    )

                    and

                    rotation_difference
                    <= (
                        self.robust_config
                        .agreement_rotation_threshold_degrees
                    )
                ):

                    supporters.append(
                        other_index
                    )

            if (
                len(supporters)
                < (
                    self.robust_config
                    .minimum_cluster_size
                )
            ):

                continue

            methods = {
                candidates[index].method
                for index
                in supporters
            }

            rmse_values = [
                candidates[index]
                .result
                .inlier_rmse

                for index
                in supporters
            ]

            fitness_values = [
                candidates[index]
                .result
                .fitness

                for index
                in supporters
            ]

            hypotheses.append(
                _Hypothesis(
                    center_index=(
                        center_index
                    ),

                    supporter_indices=(
                        supporters
                    ),

                    method_support_count=(
                        len(methods)
                    ),

                    median_rmse=float(
                        np.median(
                            rmse_values
                        )
                    ),

                    median_fitness=float(
                        np.median(
                            fitness_values
                        )
                    ),
                )
            )

        return hypotheses

    # =========================================================
    # Gewinner-Cluster
    # =========================================================

    @staticmethod
    def _select_best_hypothesis(
        hypotheses: list[
            _Hypothesis
        ],
    ) -> _Hypothesis:

        return min(
            hypotheses,

            key=lambda hypothesis: (
                hypothesis.median_rmse,

                -hypothesis
                .method_support_count,

                -len(
                    hypothesis
                    .supporter_indices
                ),

                -hypothesis
                .median_fitness,
            ),
        )

    # =========================================================
    # Cluster-Repräsentant
    # =========================================================

    @staticmethod
    def _select_cluster_representative(
        candidates: list[_Candidate],
        supporter_indices: list[int],
    ) -> int:

        return min(
            supporter_indices,

            key=lambda index: (
                candidates[index]
                .result
                .inlier_rmse,

                -candidates[index]
                .result
                .fitness,
            ),
        )

    # =========================================================
    # Fallback
    # =========================================================

    @staticmethod
    def _select_best_single_candidate(
        candidates: list[_Candidate],
        valid_indices: list[int],
    ) -> int:

        return min(
            valid_indices,

            key=lambda index: (
                candidates[index]
                .result
                .inlier_rmse,

                -candidates[index]
                .result
                .fitness,
            ),
        )

    # =========================================================
    # Normale Top-Kandidaten ausgeben
    # =========================================================

    def _print_best_candidates(
        self,
        candidates: list[_Candidate],
        valid_indices: list[int],
    ) -> None:

        sorted_indices = sorted(
            valid_indices,

            key=lambda index: (
                candidates[index]
                .result
                .inlier_rmse
            ),
        )

        number_to_print = min(
            self.robust_config
            .print_top_candidates,

            len(sorted_indices),
        )

        for rank, index in enumerate(
            sorted_indices[
                :number_to_print
            ],
            start=1,
        ):

            candidate = (
                candidates[
                    index
                ]
            )

            print(
                f"      {rank:>2}. "
                f"{candidate.label:<16} "
                f"| Fitness="
                f"{candidate.result.fitness:.4f} "
                f"| RMSE="
                f"{candidate.result.inlier_rmse:.4f}"
            )

    # =========================================================
    # Cycle-Kandidaten ausgeben
    # =========================================================

    def _print_cycle_candidates(
        self,
        candidates: list[_Candidate],
        valid_indices: list[int],
        cycle_valid_indices: list[int],
    ) -> None:

        cycle_valid_set = set(
            cycle_valid_indices
        )

        sorted_indices = sorted(
            valid_indices,

            key=lambda index: (
                candidates[index]
                .cycle_rotation_error_degrees
                if (
                    candidates[index]
                    .cycle_rotation_error_degrees
                    is not None
                )
                else float("inf"),

                candidates[index]
                .cycle_translation_error
                if (
                    candidates[index]
                    .cycle_translation_error
                    is not None
                )
                else float("inf"),
            ),
        )

        number_to_print = min(
            self.robust_config
            .print_top_candidates,

            len(sorted_indices),
        )

        for rank, index in enumerate(
            sorted_indices[
                :number_to_print
            ],
            start=1,
        ):

            candidate = (
                candidates[index]
            )

            status = (
                "PASS"
                if index in cycle_valid_set
                else "FAIL"
            )

            print(
                f"      {rank:>2}. "
                f"{candidate.label:<16} "
                f"| {status:<4} "
                f"| ΔT="
                f"{candidate.cycle_translation_error:.4f} "
                f"| ΔR="
                f"{candidate.cycle_rotation_error_degrees:.4f}° "
                f"| Fitness="
                f"{candidate.result.fitness:.4f} "
                f"| RMSE="
                f"{candidate.result.inlier_rmse:.4f}"
            )

    # =========================================================
    # Transformationsunterschied
    # =========================================================

    @staticmethod
    def _transformation_difference(
        first: np.ndarray,
        second: np.ndarray,
    ) -> tuple[float, float]:

        difference = (
            np.linalg.inv(
                first
            )
            @ second
        )

        translation_difference = float(
            np.linalg.norm(
                difference[:3, 3]
            )
        )

        rotation = (
            difference[:3, :3]
        )

        cos_angle = (
            np.trace(
                rotation
            )
            - 1.0
        ) / 2.0

        cos_angle = np.clip(
            cos_angle,
            -1.0,
            1.0,
        )

        rotation_difference = float(
            np.degrees(
                np.arccos(
                    cos_angle
                )
            )
        )

        return (
            translation_difference,
            rotation_difference,
        )