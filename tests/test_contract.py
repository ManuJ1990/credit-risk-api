"""Prueft, dass schemas.py und feature_mapping.json nicht auseinanderlaufen."""

import json
from pathlib import Path

from annotated_types import Ge, Le

from app.schemas import ApplicantInput

MAPPING = json.loads(
    (Path(__file__).parent.parent / "models" / "feature_mapping.json").read_text(
        encoding="utf-8"
    )
)


def bounds(field_name):
    """Untere und obere Grenze eines Feldes aus schemas.py."""
    metadata = ApplicantInput.model_fields[field_name].metadata
    lower = next((m.ge for m in metadata if isinstance(m, Ge)), None)
    upper = next((m.le for m in metadata if isinstance(m, Le)), None)
    return lower, upper


def test_fields_match_feature_order():
    assert list(ApplicantInput.model_fields) == MAPPING["feature_order"]


def test_categorical_bounds_match_mapping():
    for name, codes in MAPPING["categorical"].items():
        assert bounds(name) == (0, len(codes) - 1), name


def test_numeric_bounds_match_mapping():
    for name, limits in MAPPING["numeric"].items():
        assert bounds(name) == (limits["min"], limits["max"]), name