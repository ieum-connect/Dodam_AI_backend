"""예방접종 판정 엔진의 경계값을 확인한다."""

from datetime import date

import pytest

from app.domain.vaccination.engine import add_age, add_months, age_months, compute_schedule
from app.domain.vaccination.error import VaccinationError, VaccinationErrorCode
from app.domain.vaccination.schedule_data import load_schedule

BIRTH = date(2025, 5, 5)


def _compute(as_of: date, completed: list[str] | None = None, birth: date = BIRTH) -> dict:
    return compute_schedule(load_schedule(), birth, as_of, completed or [])


def _state(result: dict, dose_id: str) -> str | None:
    for vaccine in result["vaccines"]:
        for dose in vaccine["doses"]:
            if dose["doseId"] == dose_id:
                return dose["state"]
    return None


def test_add_months_clamps_to_month_end() -> None:
    assert add_months(date(2025, 1, 31), 1) == date(2025, 2, 28)
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert add_months(date(2025, 11, 30), 3) == date(2026, 2, 28)
    assert add_age(date(2025, 1, 31), "1m") == date(2025, 2, 28)


def test_add_age_applies_parts_in_order() -> None:
    assert add_age(BIRTH, "0") == BIRTH
    assert add_age(BIRTH, "15w") == date(2025, 8, 18)
    assert add_age(BIRTH, "8m+1d") == date(2026, 1, 6)
    assert add_age(BIRTH, "4y") == date(2029, 5, 5)


def test_age_months_counts_full_months() -> None:
    assert age_months(BIRTH, date(2026, 11, 4)) == 17
    assert age_months(BIRTH, date(2026, 11, 5)) == 18
    assert age_months(date(2025, 1, 31), date(2025, 2, 28)) == 1


def test_mmr_first_dose_delay_starts_exactly_on_delay_from() -> None:
    # MMR 1차: 12~16개월 → recTo 2026-09-04, delayFrom = 16개월 + 1개월 = 2026-10-05
    before = _compute(date(2026, 10, 4))
    on_day = _compute(date(2026, 10, 5))
    assert _state(before, "MMR-1") == "NOW"
    assert _state(on_day, "MMR-1") == "PAST"
    mmr = next(d for d in on_day["groups"]["past"] if d["doseId"] == "MMR-1")
    assert (mmr["recFrom"], mmr["recTo"], mmr["delayFrom"]) == (
        "2026-05-05",
        "2026-09-04",
        "2026-10-05",
    )


def test_rotavirus_first_dose_expires_on_fifteen_weeks() -> None:
    # 15주 = 2025-08-18. 전날은 포함, 당일부터 EXPIRED라 응답에서 빠진다.
    assert _state(_compute(date(2025, 8, 17)), "RV-1") == "NOW"
    assert _state(_compute(date(2025, 8, 18)), "RV-1") is None


def test_checked_expired_dose_stays_done() -> None:
    assert _state(_compute(date(2026, 11, 8), ["RV-1"]), "RV-1") == "DONE"


def _dose_ids(result: dict) -> list[str]:
    groups = result["groups"]
    return [d["doseId"] for key in ("now", "soon", "past") for d in groups[key]] + [
        d["doseId"] for v in result["vaccines"] for d in v["doses"]
    ]


def test_rotavirus_dropped_entirely_when_first_dose_missed_max_start() -> None:
    as_of = date(2025, 9, 5)
    result = _compute(as_of)
    assert not [d for d in _dose_ids(result) if d.startswith("RV-")]
    assert result["next"]["vaccineId"] != "RV"
    schedule = load_schedule()
    without_rv = {**schedule, "vaccines": [v for v in schedule["vaccines"] if v["id"] != "RV"]}
    assert result["counts"] == compute_schedule(without_rv, BIRTH, as_of, [])["counts"]


@pytest.mark.parametrize("as_of", [date(2025, 8, 18), date(2025, 8, 19)])
def test_rotavirus_dropped_from_max_start_day(as_of: date) -> None:
    assert not [d for d in _dose_ids(_compute(as_of)) if d.startswith("RV-")]


def test_rotavirus_later_doses_kept_when_first_dose_done() -> None:
    result = _compute(date(2025, 9, 5), ["RV-1"])
    assert _state(result, "RV-1") == "DONE"
    assert _state(result, "RV-2") is not None
    assert _state(result, "RV-3") is not None


@pytest.mark.parametrize("vaccine_id", ["HIB", "PCV"])
def test_hib_and_pcv_need_four_doses_without_records(vaccine_id: str) -> None:
    result = _compute(date(2026, 11, 8))
    vaccine = next(v for v in result["vaccines"] if v["vaccineId"] == vaccine_id)
    assert [d["doseId"] for d in vaccine["doses"]] == [f"{vaccine_id}-{n}" for n in range(1, 5)]


def test_checked_dose_is_done_and_reduces_remaining() -> None:
    before = _compute(date(2026, 11, 8))
    after = _compute(date(2026, 11, 8), ["BCG-1", "DTAP-4"])
    assert _state(after, "BCG-1") == "DONE"
    assert _state(after, "DTAP-4") == "DONE"
    assert after["counts"]["total"] == before["counts"]["total"]
    assert after["counts"]["done"] == 2
    assert after["counts"]["remaining"] == before["counts"]["remaining"] - 2
    assert after["counts"]["pastUnchecked"] == before["counts"]["pastUnchecked"] - 1
    assert all(d["doseId"] != "DTAP-4" for d in after["groups"]["now"])


def test_empty_and_duplicate_completed_doses() -> None:
    as_of = date(2026, 11, 8)
    assert _compute(as_of, [])["counts"]["done"] == 0
    assert _compute(as_of, ["BCG-1", "BCG-1"]) == _compute(as_of, ["BCG-1"])


def test_counts_match_groups() -> None:
    result = _compute(date(2026, 11, 8), ["BCG-1"])
    groups, counts = result["groups"], result["counts"]
    states = [d["state"] for v in result["vaccines"] for d in v["doses"]]
    assert counts["total"] == len(states)
    assert counts["done"] == states.count("DONE")
    assert counts["pastUnchecked"] == len(groups["past"]) == states.count("PAST")
    assert len(groups["now"]) == states.count("NOW")
    assert len(groups["soon"]) == states.count("SOON")
    assert groups["laterCount"] == states.count("LATER")


def test_influenza_is_excluded() -> None:
    result = _compute(date(2026, 11, 8))
    assert all(v["vaccineId"] != "FLU" for v in result["vaccines"])


def test_birth_after_as_of_is_rejected() -> None:
    with pytest.raises(VaccinationError) as error:
        _compute(date(2025, 5, 4))
    assert error.value.code is VaccinationErrorCode.BIRTH_AFTER_AS_OF


@pytest.mark.parametrize("dose_id", ["DTAP-7", "RV-4", "FLU-1", "dtap-1", ""])
def test_unknown_dose_id_is_rejected(dose_id: str) -> None:
    with pytest.raises(VaccinationError) as error:
        _compute(date(2026, 11, 8), [dose_id])
    assert error.value.code is VaccinationErrorCode.UNKNOWN_DOSE_ID


def test_unknown_dose_ids_are_collected_sorted_and_deduplicated() -> None:
    with pytest.raises(VaccinationError) as error:
        _compute(date(2026, 11, 8), ["RV-7", "BCG-1", "DTAP-9", "RV-7"])
    assert error.value.unknown_dose_ids == ["DTAP-9", "RV-7"]


def test_eighteen_month_example_next_is_now_with_earliest_rec_to() -> None:
    result = _compute(date(2026, 11, 8))
    now = result["groups"]["now"]
    assert result["ageMonths"] == 18
    assert [d["recTo"] for d in now] == sorted(d["recTo"] for d in now)
    # DTAP-4와 IPV-3 모두 recTo 2026-12-04 → recFrom이 빠른 IPV-3이 먼저
    assert [d["doseId"] for d in now[:2]] == ["IPV-3", "DTAP-4"]
    assert result["next"] == {**now[0], "group": "NOW"}
    assert result["next"]["recTo"] == "2026-12-04"


def test_next_falls_back_to_soon_then_later() -> None:
    newborn = _compute(BIRTH)
    assert newborn["next"]["group"] == "NOW"  # BCG·B형간염 1차
    done_all_now = [d["doseId"] for d in newborn["groups"]["now"]]
    soon = _compute(BIRTH, done_all_now)
    assert soon["next"]["group"] == "SOON"
    assert soon["next"]["doseId"] == soon["groups"]["soon"][0]["doseId"]


def test_schedule_version_comes_from_data_file() -> None:
    assert _compute(date(2026, 11, 8))["scheduleVersion"] == load_schedule()["version"]
