"""국가예방접종 일정 데이터(JSON)를 읽고 엔진이 기대하는 모양인지 검증한다."""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEDULE_PATH = Path(__file__).parent / "data" / "kdca_2026-01-02.json"

AGE_PATTERN = re.compile(r"^(0|\d+[dwmy](\+\d+[dwmy])*)$")
_DOSE_AGE_FIELDS = ("from", "to", "minAge", "minInt", "maxStart", "deadline", "recInt")


@lru_cache(maxsize=1)
def load_schedule() -> dict[str, Any]:
    """포함된 일정 데이터를 한 번 읽고 검증한다. 형식이 틀리면 ValueError로 기동을 막는다."""
    schedule = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
    validate_schedule(schedule)
    return schedule


def validate_schedule(schedule: dict[str, Any]) -> None:
    """필수 필드와 나이 표현 형식을 확인하고, 처음 발견한 문제 위치를 ValueError로 알린다."""
    _require(schedule, "version", "schedule")
    rules = _require(schedule, "rules", "schedule")
    _check_age(_require(rules, "delayGrace", "rules"), "rules.delayGrace")
    plans = schedule.get("plans", {})
    for vaccine in _require(schedule, "vaccines", "schedule"):
        vaccine_id = _require(vaccine, "id", "vaccine")
        where = f"vaccine {vaccine_id}"
        _require(vaccine, "name", where)
        _require(vaccine, "short", where)
        variants = _require(vaccine, "variants", where)
        if not variants:
            raise ValueError(f"{where}: variants가 비어 있습니다.")
        variant_ids = {_require(variant, "id", where) for variant in variants}
        default_variant = vaccine.get("defaultVariant")
        if default_variant is not None and default_variant not in variant_ids:
            raise ValueError(f"{where}: defaultVariant {default_variant}가 variants에 없습니다.")
        if vaccine.get("plan") and vaccine_id not in plans:
            raise ValueError(f"{where}: plan 백신인데 plans[{vaccine_id}]가 없습니다.")
        for variant in variants:
            for dose in _require(variant, "doses", f"{where}/{variant['id']}"):
                _check_dose(dose, f"{where}/{variant['id']}")


def _check_dose(dose: dict[str, Any], where: str) -> None:
    n = _require(dose, "n", where)
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"{where}: 차수 n이 1 이상의 정수가 아닙니다: {n!r}")
    where = f"{where} dose {n}"
    for field in ("from", "to", "minAge"):
        _require(dose, field, where)
    for field in _DOSE_AGE_FIELDS:
        if field in dose:
            _check_age(dose[field], f"{where}.{field}")
    for extra in dose.get("extra", []):
        _require(extra, "dose", f"{where}.extra")
        _check_age(_require(extra, "int", f"{where}.extra"), f"{where}.extra.int")


def _require(container: dict[str, Any], key: str, where: str) -> Any:
    if key not in container or container[key] is None:
        raise ValueError(f"{where}: 필수 필드 {key}가 없습니다.")
    return container[key]


def _check_age(value: Any, where: str) -> None:
    if not isinstance(value, str) or not AGE_PATTERN.match(value):
        raise ValueError(f"{where}: 나이 표현 형식이 올바르지 않습니다: {value!r}")
