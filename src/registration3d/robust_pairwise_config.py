from dataclasses import dataclass


@dataclass(frozen=True)
class RobustPairwiseConfig:

    # Anzahl Multi-Start-Läufe pro Algorithmus
    fgr_repeats: int = 20
    ransac_repeats: int = 20

    # Seeds:
    # base_seed=42 und 20 Runs -> Seeds 42 ... 61
    base_seed: int = 42

    # Zwei Transformationen gehören zum selben
    # Hypothesen-Cluster, wenn sie innerhalb
    # dieser Grenzen liegen.
    agreement_translation_threshold: float = 1.0
    agreement_rotation_threshold_degrees: float = 5.0

    # Mindestqualität eines Einzelkandidaten
    minimum_fitness: float = 0.30
    maximum_rmse: float = 0.80

    # Eine einzelne zufällige Hypothese reicht nicht.
    minimum_cluster_size: int = 2

    # Wie viele der besten Einzelkandidaten ausgegeben werden.
    print_top_candidates: int = 8