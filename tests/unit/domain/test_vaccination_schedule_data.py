"""예방접종 일정 데이터 로딩과 스키마 검증을 확인한다."""

import copy

import pytest

from app.domain.vaccination.schedule_data import load_schedule, validate_schedule


def test_bundled_schedule_passes_validation() -> None:
    schedule = load_schedule()
    assert schedule["version"] == "KDCA 2026-01-02"
    assert {vaccine["id"] for vaccine in schedule["vaccines"]} >= {"BCG", "HIB", "PCV", "RV", "FLU"}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda s: s.pop("version"), "version"),
        (lambda s: s["rules"].pop("delayGrace"), "delayGrace"),
        (lambda s: s["vaccines"][0].pop("short"), "short"),
        (lambda s: s["vaccines"][0]["variants"][0]["doses"][0].pop("to"), "to"),
        (lambda s: s["plans"].pop("HIB"), "plans"),
        (lambda s: s["vaccines"][6].update(defaultVariant="RV9"), "defaultVariant"),
    ],
)
def test_missing_required_field_is_rejected(mutate, message: str) -> None:
    schedule = copy.deepcopy(load_schedule())
    mutate(schedule)
    with pytest.raises(ValueError, match=message):
        validate_schedule(schedule)


@pytest.mark.parametrize("bad", ["2", "2mm", "m2", "1m+", "1M", "1m 1d", "", 2])
def test_bad_age_expression_is_rejected(bad) -> None:
    schedule = copy.deepcopy(load_schedule())
    schedule["vaccines"][1]["variants"][0]["doses"][1]["minInt"] = bad
    with pytest.raises(ValueError, match="나이 표현"):
        validate_schedule(schedule)


def test_combined_age_expression_is_accepted() -> None:
    schedule = copy.deepcopy(load_schedule())
    schedule["vaccines"][0]["variants"][0]["doses"][0]["to"] = "1y+2m+3d"
    validate_schedule(schedule)
