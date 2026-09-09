from registration3d.experiment_config import (
    ExperimentScenario,
)


class ScenarioFactory:
    """
    Erstellt kontrollierte Versuchsreihen.

    Innerhalb einer Versuchsreihe wird immer nur
    genau eine Variable verändert.
    """

    # ---------------------------------------------------------
    # Gemeinsame Basiswerte
    # ---------------------------------------------------------

    BASE_OVERLAP = 0.60
    BASE_ROTATION = 10.0
    BASE_TRANSLATION = 3.0
    BASE_NOISE = 0.02

    # ---------------------------------------------------------
    # Overlap-Versuchsreihe
    # ---------------------------------------------------------

    @staticmethod
    def create_overlap_scenarios() -> list[ExperimentScenario]:

        overlap_values = [
            0.80,
            0.70,
            0.60,
            0.50,
            0.40,
            0.30,
            0.20,
        ]

        scenarios = []

        for overlap in overlap_values:

            scenarios.append(
                ExperimentScenario(
                    name=f"overlap_{int(overlap * 100)}",

                    series="overlap",

                    variable_name="Overlap",

                    variable_value=overlap,

                    overlap_fraction=overlap,

                    rotation_degrees=(
                        ScenarioFactory.BASE_ROTATION
                    ),

                    translation_magnitude=(
                        ScenarioFactory.BASE_TRANSLATION
                    ),

                    noise_std=(
                        ScenarioFactory.BASE_NOISE
                    ),
                )
            )

        return scenarios

    # ---------------------------------------------------------
    # Rotation-Versuchsreihe
    # ---------------------------------------------------------

    @staticmethod
    def create_rotation_scenarios() -> list[ExperimentScenario]:

        rotation_values = [
            0.0,
            5.0,
            10.0,
            20.0,
            30.0,
            40.0,
            60.0,
        ]

        scenarios = []

        for rotation in rotation_values:

            scenarios.append(
                ExperimentScenario(
                    name=f"rotation_{int(rotation)}",

                    series="rotation",

                    variable_name="Rotation [deg]",

                    variable_value=rotation,

                    overlap_fraction=(
                        ScenarioFactory.BASE_OVERLAP
                    ),

                    rotation_degrees=rotation,

                    translation_magnitude=(
                        ScenarioFactory.BASE_TRANSLATION
                    ),

                    noise_std=(
                        ScenarioFactory.BASE_NOISE
                    ),
                )
            )

        return scenarios

    # ---------------------------------------------------------
    # Translation-Versuchsreihe
    # ---------------------------------------------------------

    @staticmethod
    def create_translation_scenarios() -> list[ExperimentScenario]:

        translation_values = [
            0.0,
            1.0,
            2.0,
            4.0,
            6.0,
            8.0,
            12.0,
        ]

        scenarios = []

        for translation in translation_values:

            scenarios.append(
                ExperimentScenario(
                    name=f"translation_{translation:g}",

                    series="translation",

                    variable_name="Translation",

                    variable_value=translation,

                    overlap_fraction=(
                        ScenarioFactory.BASE_OVERLAP
                    ),

                    rotation_degrees=(
                        ScenarioFactory.BASE_ROTATION
                    ),

                    translation_magnitude=translation,

                    noise_std=(
                        ScenarioFactory.BASE_NOISE
                    ),
                )
            )

        return scenarios

    # ---------------------------------------------------------
    # Noise-Versuchsreihe
    # ---------------------------------------------------------

    @staticmethod
    def create_noise_scenarios() -> list[ExperimentScenario]:

        noise_values = [
            0.0,
            0.01,
            0.025,
            0.05,
            0.10,
            0.20,
        ]

        scenarios = []

        for noise in noise_values:

            scenarios.append(
                ExperimentScenario(
                    name=f"noise_{noise:g}",

                    series="noise",

                    variable_name="Noise Std",

                    variable_value=noise,

                    overlap_fraction=(
                        ScenarioFactory.BASE_OVERLAP
                    ),

                    rotation_degrees=(
                        ScenarioFactory.BASE_ROTATION
                    ),

                    translation_magnitude=(
                        ScenarioFactory.BASE_TRANSLATION
                    ),

                    noise_std=noise,
                )
            )

        return scenarios

    # ---------------------------------------------------------
    # Alle Szenarien
    # ---------------------------------------------------------

    @staticmethod
    def create_all() -> list[ExperimentScenario]:

        return (
            ScenarioFactory.create_overlap_scenarios()
            + ScenarioFactory.create_rotation_scenarios()
            + ScenarioFactory.create_translation_scenarios()
            + ScenarioFactory.create_noise_scenarios()
        )