from pathlib import Path

from registration3d.config import (
    RegistrationConfig,
)
from registration3d.experiment_config import (
    ExperimentConfig,
)
from registration3d.experiment_runner import (
    ExperimentRunner,
)
from registration3d.scenario_factory import (
    ScenarioFactory,
)


def main() -> None:

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    # ---------------------------------------------------------
    # Registrierungsparameter
    # ---------------------------------------------------------

    registration_config = (
        RegistrationConfig(
            voxel_size=0.5,

            normal_radius_factor=2.0,
            max_nn=30,

            correspondence_distance_factor=2.0,
            max_iterations=100,

            fpfh_radius_factor=5.0,
            fpfh_max_nn=100,

            ransac_distance_factor=1.5,
            ransac_n=3,
            ransac_max_iterations=100_000,
            ransac_confidence=0.999,

            fgr_distance_factor=0.5,

            refinement_distance_factor=0.4,

            evaluation_distance_factor=2.0,
        )
    )

    # ---------------------------------------------------------
    # Kontrollierte Szenarien erzeugen
    # ---------------------------------------------------------

    scenarios = (
        ScenarioFactory.create_all()
    )

    # ---------------------------------------------------------
    # Experimentparameter
    # ---------------------------------------------------------

    experiment_config = (
        ExperimentConfig(

            # Zunächst 3 zum Testen.
            # Später auf 10 oder 20 erhöhen.
            repeats=10,

            base_seed=42,

            success_translation_threshold=1.0,

            success_rotation_threshold_degrees=2.0,

            number_of_points=20_000,

            scenarios=scenarios,
        )
    )

    # ---------------------------------------------------------
    # Runner
    # ---------------------------------------------------------

    runner = ExperimentRunner(
        registration_config=(
            registration_config
        ),

        experiment_config=(
            experiment_config
        ),

        output_directory=(
            project_root
            / "output"
            / "controlled_experiments"
        ),
    )

    runner.run()


if __name__ == "__main__":
    main()