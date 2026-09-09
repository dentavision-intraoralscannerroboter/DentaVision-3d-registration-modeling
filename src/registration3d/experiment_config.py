from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentScenario:
    name: str

    # Zu welcher Versuchsreihe gehört das Szenario?
    series: str

    # Welche Variable wird in dieser Reihe verändert?
    variable_name: str

    # Wert dieser Variable
    variable_value: float

    # Tatsächliche Parameter des Szenarios
    overlap_fraction: float
    rotation_degrees: float
    translation_magnitude: float
    noise_std: float


@dataclass(frozen=True)
class ExperimentConfig:

    # Anzahl Wiederholungen pro Szenario
    repeats: int = 3

    # Seed für reproduzierbare Testdaten
    base_seed: int = 42

    # Erfolgskriterien
    success_translation_threshold: float = 1.0
    success_rotation_threshold_degrees: float = 2.0

    # Punktzahl des synthetischen Modells
    number_of_points: int = 20_000

    # Wird später von ScenarioFactory befüllt
    scenarios: list[ExperimentScenario] = field(
        default_factory=list
    )