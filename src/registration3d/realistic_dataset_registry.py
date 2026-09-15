from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DentalModel:
    model_id: str
    upper_filename: str
    lower_filename: str

    def upper_path(
        self,
        source_directory: Path,
    ) -> Path:
        return (
            source_directory
            / self.upper_filename
        )

    def lower_path(
        self,
        source_directory: Path,
    ) -> Path:
        return (
            source_directory
            / self.lower_filename
        )


DENTAL_MODELS: tuple[
    DentalModel,
    ...,
] = (
    DentalModel(
        model_id="model_01",
        upper_filename="upper_jaw.stl",
        lower_filename="lower_jaw.stl",
    ),
    DentalModel(
        model_id="model_02",
        upper_filename="upper_jaw2.stl",
        lower_filename="lower_jaw2.stl",
    ),
    DentalModel(
        model_id="model_03",
        upper_filename="upper_jaw3.stl",
        lower_filename="lower_jaw3.stl",
    ),
    DentalModel(
        model_id="model_04",
        upper_filename="upper_jaw4.stl",
        lower_filename="lower_jaw4.stl",
    ),
    DentalModel(
        model_id="model_05",
        upper_filename="upper_jaw5.stl",
        lower_filename="lower_jaw5.stl",
    ),
)


JAW_NAMES = (
    "upper",
    "lower",
)


def get_source_directory(
    project_root: Path,
) -> Path:
    return (
        project_root
        / "data"
        / "realistic_jaw"
    )


def get_generated_directory(
    project_root: Path,
    model_id: str,
    jaw: str,
) -> Path:
    return (
        project_root
        / "data"
        / "realistic_jaw"
        / "generated"
        / model_id
        / jaw
    )


def get_stl_path(
    project_root: Path,
    model: DentalModel,
    jaw: str,
) -> Path:

    source_directory = (
        get_source_directory(
            project_root
        )
    )

    if jaw == "upper":
        return model.upper_path(
            source_directory
        )

    if jaw == "lower":
        return model.lower_path(
            source_directory
        )

    raise ValueError(
        f"Unknown jaw: {jaw}"
    )