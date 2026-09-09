from dataclasses import dataclass


@dataclass(frozen=True)
class MeshConfig:

    # =========================================================
    # Allgemein / Normalenschätzung
    # =========================================================

    normal_radius_factor: float = 3.0

    normal_max_nn: int = 50

    # =========================================================
    # Poisson Surface Reconstruction
    # =========================================================

    poisson_depth: int = 9

    poisson_density_quantile: float = 0.02

    # =========================================================
    # Ball Pivoting
    #
    # Ergebnis unserer Parameterstudie:
    #
    # A_small = (1.0, 1.5, 2.0)
    #
    # Diese Konfiguration zeigte auf dem aktuellen
    # synthetischen Dentaldatensatz den niedrigsten
    # symmetrischen RMSE.
    # =========================================================

    ball_pivoting_radius_factors: tuple[
        float,
        ...
    ] = (
        1.0,
        1.5,
        2.0,
    )

    # =========================================================
    # Mesh Cleanup
    #
    # Im aktuellen synthetischen Datensatz:
    #
    # Hauptkomponente:
    #   40.020 Dreiecke
    #
    # Zweitgrößte Komponente:
    #   104 Dreiecke
    #
    # Daher werden Komponenten mit weniger als
    # 250 Dreiecken als kleine Artefakt-Komponenten entfernt.
    #
    # WICHTIG:
    # Dieser Wert ist momentan datensatzspezifisch.
    # =========================================================

    minimum_component_triangles: int = 250

    # =========================================================
    # Mesh Simplification
    #
    # Wird erst im nächsten Schritt verwendet.
    # =========================================================

    target_triangle_count: int = 100_000

    # =========================================================
    # Evaluation
    # =========================================================

    evaluation_sample_points: int = 100_000