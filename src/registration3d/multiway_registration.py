from dataclasses import dataclass
import copy

import numpy as np
import open3d as o3d

from registration3d.algorithms.base import (
    RegistrationAlgorithm,
)
from registration3d.config import (
    RegistrationConfig,
)
from registration3d.evaluator import (
    RegistrationEvaluator,
)
from registration3d.multiway_config import (
    MultiwayConfig,
)
from registration3d.preprocessing import (
    PointCloudPreprocessor,
)
from registration3d.result import (
    RegistrationResult,
)


@dataclass
class MultiwayRegistrationResult:

    pose_graph: (
        o3d.pipelines.registration.PoseGraph
    )

    combined_cloud: (
        o3d.geometry.PointCloud
    )

    transformed_clouds: list[
        o3d.geometry.PointCloud
    ]


class MultiwayRegistration:

    def __init__(
        self,
        registration_config: RegistrationConfig,
        multiway_config: MultiwayConfig,
        pairwise_algorithm: RegistrationAlgorithm,
    ) -> None:

        self.registration_config = (
            registration_config
        )

        self.multiway_config = (
            multiway_config
        )

        self.pairwise_algorithm = (
            pairwise_algorithm
        )

        self.preprocessor = (
            PointCloudPreprocessor(
                registration_config
            )
        )

    # =========================================================
    # Hauptregistrierung
    # =========================================================

    def register(
        self,
        clouds: list[
            o3d.geometry.PointCloud
        ],
    ) -> MultiwayRegistrationResult:

        if len(clouds) < 2:
            raise ValueError(
                "Für Multiway Registration werden "
                "mindestens zwei Punktwolken benötigt."
            )

        print()
        print("=" * 80)
        print("MULTIWAY REGISTRATION")
        print("=" * 80)

        # =====================================================
        # Preprocessing
        # =====================================================

        print()
        print(
            "Preprocessing der Fragmente ..."
        )

        processed_clouds = []

        for index, cloud in enumerate(
            clouds
        ):

            processed = (
                self.preprocessor.preprocess(
                    cloud
                )
            )

            processed_clouds.append(
                processed
            )

            print(
                f"  Fragment {index}: "
                f"{len(processed.points):,} Punkte"
            )

        # =====================================================
        # Pose Graph erstellen
        # =====================================================

        pose_graph = (
            self._build_pose_graph(
                processed_clouds
            )
        )

        # =====================================================
        # Global Optimization
        # =====================================================

        self._optimize_pose_graph(
            pose_graph
        )

        # =====================================================
        # Optimierte Posen anwenden
        # =====================================================

        print()
        print(
            "Wende optimierte Posen an ..."
        )

        transformed_clouds = []

        combined_cloud = (
            o3d.geometry.PointCloud()
        )

        for index, (
            cloud,
            node,
        ) in enumerate(
            zip(
                clouds,
                pose_graph.nodes,
            )
        ):

            transformed = (
                copy.deepcopy(
                    cloud
                )
            )

            transformed.transform(
                node.pose
            )

            transformed_clouds.append(
                transformed
            )

            combined_cloud += (
                transformed
            )

            print(
                f"  Fragment {index} transformiert"
            )

        # =====================================================
        # Gesamtwolke downsamplen
        # =====================================================

        if (
            self.multiway_config
            .merge_voxel_size
            > 0.0
        ):

            combined_cloud = (
                combined_cloud
                .voxel_down_sample(
                    voxel_size=(
                        self.multiway_config
                        .merge_voxel_size
                    )
                )
            )

        print()
        print(
            "Multiway Registration abgeschlossen."
        )

        print(
            f"Gesamtpunktwolke: "
            f"{len(combined_cloud.points):,} Punkte"
        )

        return MultiwayRegistrationResult(
            pose_graph=pose_graph,
            combined_cloud=(
                combined_cloud
            ),
            transformed_clouds=(
                transformed_clouds
            ),
        )

    # =========================================================
    # Pose Graph aufbauen
    # =========================================================

    def _build_pose_graph(
        self,
        clouds: list[
            o3d.geometry.PointCloud
        ],
    ) -> o3d.pipelines.registration.PoseGraph:

        pose_graph = (
            o3d.pipelines.registration
            .PoseGraph()
        )

        # =====================================================
        # Fragment 0 ist Referenz
        # =====================================================

        pose_graph.nodes.append(
            o3d.pipelines.registration
            .PoseGraphNode(
                np.eye(4)
            )
        )

        cumulative_transform = (
            np.eye(4)
        )

        # =====================================================
        # Odometry
        # =====================================================

        print()
        print("-" * 80)
        print("ODOMETRY EDGES")
        print("-" * 80)

        for source_id in range(
            len(clouds) - 1
        ):

            target_id = (
                source_id + 1
            )

            result = (
                self._register_pair(
                    source=(
                        clouds[source_id]
                    ),
                    target=(
                        clouds[target_id]
                    ),
                    source_id=source_id,
                    target_id=target_id,
                )
            )

            confidence = (
                self._get_confidence(
                    result
                )
            )

            if not (
                self._is_valid_odometry(
                    result
                )
            ):

                raise RuntimeError(
                    "\nUnsichere oder schlechte "
                    "Odometry-Registrierung "
                    f"{source_id}->{target_id}.\n"
                    f"Fitness: "
                    f"{result.fitness:.4f}\n"
                    f"RMSE: "
                    f"{result.inlier_rmse:.4f}\n"
                    f"Confidence: "
                    f"{confidence.upper()}"
                )

            information = (
                self._create_information_matrix(
                    source=(
                        clouds[source_id]
                    ),
                    target=(
                        clouds[target_id]
                    ),
                    transformation=(
                        result.transformation
                    ),
                )
            )

            # result.transformation:
            # source -> target
            #
            # cumulative_transform:
            # Fragment 0 -> aktuelles Fragment

            cumulative_transform = (
                result.transformation
                @ cumulative_transform
            )

            # PoseGraphNode.pose:
            # aktuelles Fragment -> Referenz / Fragment 0

            node_pose = (
                np.linalg.inv(
                    cumulative_transform
                )
            )

            pose_graph.nodes.append(
                o3d.pipelines.registration
                .PoseGraphNode(
                    node_pose
                )
            )

            pose_graph.edges.append(
                o3d.pipelines.registration
                .PoseGraphEdge(
                    source_id,
                    target_id,
                    result.transformation,
                    information,
                    uncertain=False,
                )
            )

            print(
                f"  ACCEPTED "
                f"{source_id}->{target_id} "
                f"| Fitness="
                f"{result.fitness:.4f} "
                f"| RMSE="
                f"{result.inlier_rmse:.4f} "
                f"| Confidence="
                f"{confidence.upper()}"
            )

        # =====================================================
        # Loop Closures
        # =====================================================

        if (
            self.multiway_config
            .use_loop_closures
        ):

            self._add_loop_closures(
                pose_graph=(
                    pose_graph
                ),
                clouds=(
                    clouds
                ),
            )

        return pose_graph

    # =========================================================
    # Loop Closures hinzufügen
    # =========================================================

    def _add_loop_closures(
        self,
        pose_graph: (
            o3d.pipelines.registration.PoseGraph
        ),
        clouds: list[
            o3d.geometry.PointCloud
        ],
    ) -> None:

        print()
        print("-" * 80)
        print(
            "LOOP CLOSURE CANDIDATES"
        )
        print("-" * 80)

        number_of_clouds = (
            len(clouds)
        )

        for source_id in range(
            number_of_clouds
        ):

            first_target = (
                source_id + 2
            )

            last_target = min(
                number_of_clouds,
                source_id
                + (
                    self.multiway_config
                    .loop_closure_max_gap
                )
                + 1,
            )

            for target_id in range(
                first_target,
                last_target,
            ):

                print(
                    f"  Registriere "
                    f"{source_id}->{target_id} ..."
                )

                # =============================================
                # Transformation aus der bereits aufgebauten
                # Odometry-Kette vorhersagen
                # =============================================

                predicted_transform = (
                    self._predict_transform_from_odometry(
                        pose_graph=(
                            pose_graph
                        ),
                        source_id=(
                            source_id
                        ),
                        target_id=(
                            target_id
                        ),
                    )
                )

                # =============================================
                # Multi-Start + Cycle Consistency
                # =============================================

                if hasattr(
                    self.pairwise_algorithm,
                    "register_with_cycle_consistency",
                ):

                    result = (
                        self.pairwise_algorithm
                        .register_with_cycle_consistency(
                            source=(
                                clouds[source_id]
                            ),
                            target=(
                                clouds[target_id]
                            ),
                            predicted_transform=(
                                predicted_transform
                            ),
                            translation_threshold=(
                                self.multiway_config
                                .loop_closure_cycle_translation_threshold
                            ),
                            rotation_threshold_degrees=(
                                self.multiway_config
                                .loop_closure_cycle_rotation_threshold_degrees
                            ),
                        )
                    )

                else:

                    # =========================================
                    # Fallback für normale Algorithmen
                    # =========================================

                    result = (
                        self._register_pair(
                            source=(
                                clouds[source_id]
                            ),
                            target=(
                                clouds[target_id]
                            ),
                            source_id=(
                                source_id
                            ),
                            target_id=(
                                target_id
                            ),
                        )
                    )

                    (
                        cycle_translation_error,
                        cycle_rotation_error,
                    ) = (
                        self._transformation_difference(
                            predicted=(
                                predicted_transform
                            ),
                            direct=(
                                result.transformation
                            ),
                        )
                    )

                    if not (
                        self._is_cycle_consistent(
                            translation_error=(
                                cycle_translation_error
                            ),
                            rotation_error=(
                                cycle_rotation_error
                            ),
                        )
                    ):

                        print(
                            f"  REJECTED "
                            f"{source_id}->{target_id} "
                            f"| Fitness="
                            f"{result.fitness:.4f} "
                            f"| RMSE="
                            f"{result.inlier_rmse:.4f} "
                            f"| Cycle ΔT="
                            f"{cycle_translation_error:.4f} "
                            f"| ΔR="
                            f"{cycle_rotation_error:.4f}° "
                            f"| Reason=CYCLE INCONSISTENT"
                        )

                        continue

                # =============================================
                # Einheitliche Evaluation
                # =============================================

                result = (
                    RegistrationEvaluator
                    .evaluate_result(
                        result=result,
                        source=(
                            clouds[source_id]
                        ),
                        target=(
                            clouds[target_id]
                        ),
                        config=(
                            self.registration_config
                        ),
                    )
                )

                confidence = (
                    self._get_confidence(
                        result
                    )
                )

                # =============================================
                # Finaler Quality Check
                # =============================================

                if not (
                    self._is_basic_valid_loop_closure(
                        result
                    )
                ):

                    print(
                        f"  REJECTED "
                        f"{source_id}->{target_id} "
                        f"| Fitness="
                        f"{result.fitness:.4f} "
                        f"| RMSE="
                        f"{result.inlier_rmse:.4f} "
                        f"| Confidence="
                        f"{confidence.upper()} "
                        f"| Reason=NO VALID "
                        f"CYCLE-CONSISTENT CONSENSUS"
                    )

                    continue

                information = (
                    self._create_information_matrix(
                        source=(
                            clouds[source_id]
                        ),
                        target=(
                            clouds[target_id]
                        ),
                        transformation=(
                            result.transformation
                        ),
                    )
                )

                pose_graph.edges.append(
                    o3d.pipelines.registration
                    .PoseGraphEdge(
                        source_id,
                        target_id,
                        result.transformation,
                        information,
                        uncertain=True,
                    )
                )

                cycle_translation_error = (
                    getattr(
                        result,
                        "cycle_translation_error",
                        None,
                    )
                )

                cycle_rotation_error = (
                    getattr(
                        result,
                        "cycle_rotation_error_degrees",
                        None,
                    )
                )

                print(
                    f"  ACCEPTED "
                    f"{source_id}->{target_id} "
                    f"| Fitness="
                    f"{result.fitness:.4f} "
                    f"| RMSE="
                    f"{result.inlier_rmse:.4f} "
                    f"| Confidence="
                    f"{confidence.upper()}"
                )

                if (
                    cycle_translation_error
                    is not None
                    and
                    cycle_rotation_error
                    is not None
                ):

                    print(
                        f"    Final Cycle Difference: "
                        f"ΔT="
                        f"{cycle_translation_error:.4f} "
                        f"| ΔR="
                        f"{cycle_rotation_error:.4f}°"
                    )

    # =========================================================
    # Transformation aus der Odometry vorhersagen
    # =========================================================

    @staticmethod
    def _predict_transform_from_odometry(
        pose_graph: (
            o3d.pipelines.registration.PoseGraph
        ),
        source_id: int,
        target_id: int,
    ) -> np.ndarray:

        source_pose = (
            np.asarray(
                pose_graph
                .nodes[source_id]
                .pose
            )
        )

        target_pose = (
            np.asarray(
                pose_graph
                .nodes[target_id]
                .pose
            )
        )

        # source_pose:
        # source -> reference
        #
        # target_pose:
        # target -> reference
        #
        # gesucht:
        # source -> target
        #
        # daher:
        # inv(target_pose) @ source_pose

        return (
            np.linalg.inv(
                target_pose
            )
            @ source_pose
        )

    # =========================================================
    # Cycle Difference
    # =========================================================

    @staticmethod
    def _transformation_difference(
        predicted: np.ndarray,
        direct: np.ndarray,
    ) -> tuple[float, float]:

        difference = (
            np.linalg.inv(
                predicted
            )
            @ direct
        )

        translation_error = float(
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

        rotation_error = float(
            np.degrees(
                np.arccos(
                    cos_angle
                )
            )
        )

        return (
            translation_error,
            rotation_error,
        )

    # =========================================================
    # Cycle Consistency prüfen
    # =========================================================

    def _is_cycle_consistent(
        self,
        translation_error: float,
        rotation_error: float,
    ) -> bool:

        translation_ok = (
            translation_error
            <= (
                self.multiway_config
                .loop_closure_cycle_translation_threshold
            )
        )

        rotation_ok = (
            rotation_error
            <= (
                self.multiway_config
                .loop_closure_cycle_rotation_threshold_degrees
            )
        )

        return (
            translation_ok
            and rotation_ok
        )

    # =========================================================
    # Pairwise Registrierung
    # =========================================================

    def _register_pair(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        source_id: int,
        target_id: int,
    ) -> RegistrationResult:

        print(
            f"  Registriere "
            f"{source_id}->{target_id} ..."
        )

        result = (
            self.pairwise_algorithm
            .register(
                source,
                target,
            )
        )

        result = (
            RegistrationEvaluator
            .evaluate_result(
                result=result,
                source=source,
                target=target,
                config=(
                    self.registration_config
                ),
            )
        )

        return result

    # =========================================================
    # Confidence
    # =========================================================

    @staticmethod
    def _get_confidence(
        result: RegistrationResult,
    ) -> str:

        return getattr(
            result,
            "confidence",
            "high",
        )

    # =========================================================
    # Odometry validieren
    # =========================================================

    def _is_valid_odometry(
        self,
        result: RegistrationResult,
    ) -> bool:

        confidence = (
            self._get_confidence(
                result
            )
        )

        confidence_ok = (
            confidence
            in {
                "high",
                "medium",
            }
        )

        fitness_ok = (
            result.fitness
            >= (
                self.multiway_config
                .odometry_min_fitness
            )
        )

        rmse_ok = (
            result.inlier_rmse
            <= (
                self.multiway_config
                .odometry_max_rmse
            )
        )

        return (
            confidence_ok
            and fitness_ok
            and rmse_ok
        )

    # =========================================================
    # Loop Closure validieren
    # =========================================================

    def _is_basic_valid_loop_closure(
        self,
        result: RegistrationResult,
    ) -> bool:

        confidence = (
            self._get_confidence(
                result
            )
        )

        confidence_ok = (
            confidence
            in {
                "high",
                "medium",
            }
        )

        fitness_ok = (
            result.fitness
            >= (
                self.multiway_config
                .loop_closure_min_fitness
            )
        )

        rmse_ok = (
            result.inlier_rmse
            <= (
                self.multiway_config
                .loop_closure_max_rmse
            )
        )

        return (
            confidence_ok
            and fitness_ok
            and rmse_ok
        )

    # =========================================================
    # Information Matrix
    # =========================================================

    def _create_information_matrix(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        transformation: np.ndarray,
    ) -> np.ndarray:

        return (
            o3d.pipelines.registration
            .get_information_matrix_from_point_clouds(
                source,
                target,
                self.registration_config
                .evaluation_distance,
                transformation,
            )
        )

    # =========================================================
    # Global Optimization
    # =========================================================

    def _optimize_pose_graph(
        self,
        pose_graph: (
            o3d.pipelines.registration.PoseGraph
        ),
    ) -> None:

        print()
        print("=" * 80)
        print(
            "GLOBAL POSE GRAPH OPTIMIZATION"
        )
        print("=" * 80)

        method = (
            o3d.pipelines.registration
            .GlobalOptimizationLevenbergMarquardt()
        )

        criteria = (
            o3d.pipelines.registration
            .GlobalOptimizationConvergenceCriteria()
        )

        option = (
            o3d.pipelines.registration
            .GlobalOptimizationOption(
                max_correspondence_distance=(
                    self.registration_config
                    .evaluation_distance
                ),
                edge_prune_threshold=(
                    self.multiway_config
                    .edge_prune_threshold
                ),
                preference_loop_closure=(
                    self.multiway_config
                    .preference_loop_closure
                ),
                reference_node=(
                    self.multiway_config
                    .reference_node
                ),
            )
        )

        (
            o3d.pipelines.registration
            .global_optimization(
                pose_graph,
                method,
                criteria,
                option,
            )
        )