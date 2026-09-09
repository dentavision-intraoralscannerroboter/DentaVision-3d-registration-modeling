from dataclasses import dataclass


@dataclass(frozen=True)
class MultiwayConfig:

    # =========================================================
    # Odometry
    # =========================================================

    odometry_min_fitness: float = 0.15

    odometry_max_rmse: float = 0.80

    # =========================================================
    # Loop Closures
    # =========================================================

    use_loop_closures: bool = True

    # Maximale Distanz zwischen Fragment-IDs.
    #
    # Beispiel:
    #
    # gap = 2
    #
    # erlaubt:
    # 0 -> 2
    # 1 -> 3
    #
    # aber nicht:
    # 0 -> 3
    #
    loop_closure_max_gap: int = 2

    loop_closure_min_fitness: float = 0.30

    loop_closure_max_rmse: float = 0.60

    # =========================================================
    # Cycle Consistency
    # =========================================================
    #
    # Die direkt registrierte Loop Closure muss ungefähr mit
    # der aus der Odometry-Kette vorhergesagten Transformation
    # übereinstimmen.
    # =========================================================

    loop_closure_cycle_translation_threshold: float = 1.0

    loop_closure_cycle_rotation_threshold_degrees: float = 5.0

    # =========================================================
    # Pose Graph Optimization
    # =========================================================

    edge_prune_threshold: float = 0.25

    preference_loop_closure: float = 0.10

    reference_node: int = 0

    # =========================================================
    # Zusammenführen
    # =========================================================

    merge_voxel_size: float = 0.25