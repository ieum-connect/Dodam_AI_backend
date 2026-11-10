"""예방접종 일정 판정 엔진. 원본 VaccEngine.compute(도담 예방접종 관리.html)의 규칙을 옮겼다.

원본과의 차이: 접종 기록(날짜) 대신 체크한 차수 id만 받는다. 체크한 차수는 DONE이고,
날짜가 필요한 규칙(최소 간격 경고, 이전 접종일 기준 일정 조정, blocked)은 계산하지 않는다.
"""

import calendar
from collections.abc import Iterable
from datetime import date, timedelta
from typing import Any

from app.domain.vaccination.error import VaccinationError, VaccinationErrorCode

SOON_DAYS = 90

DONE = "DONE"
PAST = "PAST"
NOW = "NOW"
SOON = "SOON"
LATER = "LATER"


def add_months(start: date, months: int) -> date:
    """월을 더하고, 그 달에 같은 날이 없으면 말일로 맞춘다(1/31 + 1개월 = 2/28)."""
    index = start.month - 1 + months
    year, month = start.year + index // 12, index % 12 + 1
    return date(year, month, min(start.day, calendar.monthrange(year, month)[1]))


def add_age(start: date, expr: str) -> date:
    """'0', '15w', '8m+1d' 같은 나이 표현을 앞에서부터 차례로 더한다."""
    if expr == "0":
        return start
    result = start
    for part in expr.split("+"):
        n, unit = int(part[:-1]), part[-1]
        if unit == "d":
            result += timedelta(days=n)
        elif unit == "w":
            result += timedelta(weeks=n)
        elif unit == "m":
            result = add_months(result, n)
        elif unit == "y":
            result = add_months(result, 12 * n)
        else:
            raise ValueError(f"나이 표현 형식이 올바르지 않습니다: {expr!r}")
    return result


def age_months(birth: date, at: date) -> int:
    """만 나이(개월). add_months(birth, n) <= at 인 가장 큰 n."""
    n = max(0, (at.year - birth.year) * 12)
    while add_months(birth, n + 1) <= at:
        n += 1
    while n > 0 and add_months(birth, n) > at:
        n -= 1
    return n


def _plan_total(plans: dict[str, list[list[int]]], vaccine_id: str, age: int) -> int:
    for under_months, total in plans[vaccine_id]:
        if age < under_months:
            return total
    return 0


def _default_variant(vaccine: dict[str, Any]) -> dict[str, Any]:
    variant_id = vaccine.get("defaultVariant") or vaccine["variants"][0]["id"]
    return next(v for v in vaccine["variants"] if v["id"] == variant_id)


def known_dose_ids(schedule: dict[str, Any]) -> set[str]:
    """응답에 나올 수 있는 차수 id(기본 variant 기준, 인플루엔자 제외)."""
    return {
        f"{vaccine['id']}-{dose['n']}"
        for vaccine in schedule["vaccines"]
        if not vaccine.get("info")
        for dose in _default_variant(vaccine)["doses"]
    }


def compute_schedule(
    schedule: dict[str, Any], birth: date, as_of: date, completed_doses: Iterable[str]
) -> dict[str, Any]:
    """생년월일·기준일·체크한 차수로 묶음, 다음 접종, 개수를 계산해 API 응답 dict로 돌려준다."""
    if birth > as_of:
        raise VaccinationError(VaccinationErrorCode.BIRTH_AFTER_AS_OF)
    completed = set(completed_doses)
    unknown = sorted(completed - known_dose_ids(schedule))
    if unknown:
        raise VaccinationError(VaccinationErrorCode.UNKNOWN_DOSE_ID, unknown)

    grace = schedule["rules"]["delayGrace"]
    vaccines: list[dict[str, Any]] = []
    states = (DONE, PAST, NOW, SOON, LATER)
    doses_by_state: dict[str, list[dict[str, Any]]] = {state: [] for state in states}

    for vaccine in schedule["vaccines"]:
        if vaccine.get("info"):
            continue
        doses = _default_variant(vaccine)["doses"]
        first = doses[0]
        if (
            first.get("maxStart")
            and f"{vaccine['id']}-{first['n']}" not in completed
            and as_of >= add_age(birth, first["maxStart"])
        ):
            continue  # 1차 시작 시한이 지났고 1차 미접종: 백신 전체 EXPIRED
        total = len(doses)
        if vaccine.get("plan"):
            # 기록이 없으므로 첫 권장 시기에 제때 시작했다고 보고 필요 횟수를 정한다.
            first_age = age_months(birth, add_age(birth, doses[0]["from"]))
            total = _plan_total(schedule["plans"], vaccine["id"], first_age)

        vaccine_doses: list[dict[str, Any]] = []
        for dose in doses:
            dose_id = f"{vaccine['id']}-{dose['n']}"
            done = dose_id in completed
            if not done and vaccine.get("plan") and dose["n"] > total:
                continue  # NOT_REQUIRED
            rec_from = add_age(birth, dose["from"])
            to = add_age(birth, dose["to"])
            delay_from = add_age(to, grace)
            if not done and _expired(dose, birth, as_of):
                continue  # EXPIRED
            state = _state(done, as_of, rec_from, delay_from)
            item = {
                "doseId": dose_id,
                "vaccineId": vaccine["id"],
                "name": vaccine["short"],
                "label": dose.get("label") or f"{dose['n']}차",
                "recFrom": rec_from.isoformat(),
                "recTo": (to - timedelta(days=1)).isoformat(),
                "delayFrom": delay_from.isoformat(),
            }
            doses_by_state[state].append(item)
            vaccine_doses.append({"doseId": dose_id, "label": item["label"], "state": state})
        if vaccine_doses:
            vaccines.append(
                {"vaccineId": vaccine["id"], "name": vaccine["name"], "doses": vaccine_doses}
            )

    now = sorted(doses_by_state[NOW], key=lambda d: (d["recTo"], d["recFrom"]))
    soon = sorted(doses_by_state[SOON], key=lambda d: d["recFrom"])
    past = sorted(doses_by_state[PAST], key=lambda d: d["recFrom"])
    later = sorted(doses_by_state[LATER], key=lambda d: d["recFrom"])

    total_count = sum(len(items) for items in doses_by_state.values())
    done_count = len(doses_by_state[DONE])
    return {
        "scheduleVersion": schedule["version"],
        "asOf": as_of.isoformat(),
        "ageMonths": age_months(birth, as_of),
        "next": _next(now, soon, later),
        "groups": {"now": now, "soon": soon, "past": past, "laterCount": len(later)},
        "counts": {
            "total": total_count,
            "done": done_count,
            "remaining": total_count - done_count,
            "pastUnchecked": len(past),
        },
        "vaccines": vaccines,
    }


def _expired(dose: dict[str, Any], birth: date, as_of: date) -> bool:
    return bool(dose.get("deadline")) and as_of >= add_age(birth, dose["deadline"])


def _state(done: bool, as_of: date, rec_from: date, delay_from: date) -> str:
    if done:
        return DONE
    if as_of >= delay_from:
        return PAST
    if as_of >= rec_from:
        return NOW
    if (rec_from - as_of).days <= SOON_DAYS:
        return SOON
    return LATER


def _next(
    now: list[dict[str, Any]], soon: list[dict[str, Any]], later: list[dict[str, Any]]
) -> dict[str, Any] | None:
    for group, items in ((NOW, now), (SOON, soon), (LATER, later)):
        if items:
            return {**items[0], "group": group}
    return None
