"""What a wrong extraction does to a Wajib client.

Ported from the Wajib rules engine, /home/kiritohaze/Projects/ClaudeCode/wajib at commit
9e040a9: `src/lib/rules/uae-compliance-rules.ts` (DUE_SOON_DAYS, URGENT_DAYS, PENALTY,
the trade-licence rule and `ctRegistrationDeadlineByMonth`) and `src/lib/rules/engine.ts`
(`deriveStatus`).

Rules applied here:
- The trade licence renewal is due on the licence expiry date. Alerts fire when the
  obligation turns due-soon (30 days before), urgent (7 days before) and due (the day).
- The Corporate Tax registration deadline follows the month of licence issue
  (FTA Decision No. 3 of 2024). The table is applied to every licence, as the benchmark
  contract says, although the Wajib engine uses it only for companies incorporated before
  1 March 2024.

Per licence:
- wrong_day: the predicted expiry is a real date that differs from gold, so all three
  alerts fire on the wrong day. This includes a predicted expiry on a licence whose gold
  expiry is null.
- missed: gold has an expiry and the prediction is null or not a real date, so no
  renewal alert exists at all.
- ct_deadline_wrong: the Corporate Tax deadline derived from the predicted issue date is
  not the one derived from gold (a missing predicted issue date counts). Two issue months
  that share a deadline (January and February) cause nothing and are not counted.

A document routed to review is corrected by a person, so after review it causes nothing.
The AED figure is Wajib's own indicative amount for operating on an expired licence, not a
measured loss.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from bench.normalize import parse_date
from eval.metrics import DocResult

DUE_SOON_DAYS = 30
URGENT_DAYS = 7
ALERTS_PER_LICENCE = 3
LICENCE_OPERATING_EXPIRED_AED = 5_000
EXPOSURE_LABEL = (
    "Wajib's own indicative figure for operating on an expired licence "
    "(PENALTY.LICENCE_OPERATING_EXPIRED in its rules engine). Not a measured loss."
)

# Issue month (1 to 12) -> (year, month, day) of the registration deadline. The TypeScript
# table is keyed 0 to 11; this one is shifted by one.
CT_DEADLINE_BY_MONTH: dict[int, tuple[int, int, int]] = {
    1: (2024, 5, 31),
    2: (2024, 5, 31),
    3: (2024, 6, 30),
    4: (2024, 6, 30),
    5: (2024, 7, 31),
    6: (2024, 8, 31),
    7: (2024, 9, 30),
    8: (2024, 10, 31),
    9: (2024, 10, 31),
    10: (2024, 11, 30),
    11: (2024, 11, 30),
    12: (2024, 12, 31),
}


def ct_registration_deadline(issue_date: date) -> date:
    return date(*CT_DEADLINE_BY_MONTH[issue_date.month])


def renewal_alert_dates(expiry: date) -> dict[str, date]:
    return {
        "due_soon": expiry - timedelta(days=DUE_SOON_DAYS),
        "urgent": expiry - timedelta(days=URGENT_DAYS),
        "due": expiry,
    }


def _date_or_none(value: object) -> date | None:
    return parse_date(value) if isinstance(value, str) else None


def classify_licence(r: DocResult) -> dict[str, bool]:
    """Flags for one licence, before any review."""
    gold_expiry = _date_or_none(r.gold_value("expiry_date"))
    pred_expiry = _date_or_none(r.pred("expiry_date"))
    wrong_day = pred_expiry is not None and pred_expiry != gold_expiry
    missed = gold_expiry is not None and pred_expiry is None
    late = missed or (
        gold_expiry is not None and pred_expiry is not None and pred_expiry > gold_expiry
    )
    gold_issue = _date_or_none(r.gold_value("issue_date"))
    pred_issue = _date_or_none(r.pred("issue_date"))
    gold_ct = ct_registration_deadline(gold_issue) if gold_issue else None
    pred_ct = ct_registration_deadline(pred_issue) if pred_issue else None
    return {
        "wrong_day": wrong_day,
        "missed": missed,
        "late": late,
        "ct_deadline_wrong": gold_ct != pred_ct,
    }


def _per_100(count: int, n: int) -> float | None:
    return 100 * count / n if n else None


def licence_consequences(results: list[DocResult]) -> dict[str, Any]:
    """Counts over the trade licences in `results`, without and after review."""
    licences = [r for r in results if r.doc_type == "trade_licence"]
    n = len(licences)
    flags = [(r, classify_licence(r)) for r in licences]

    def total(key: str, after_review: bool) -> int:
        return sum(1 for r, f in flags if f[key] and not (after_review and r.routed))

    wrong_day = total("wrong_day", False)
    missed = total("missed", False)
    ct_wrong = total("ct_deadline_wrong", False)
    wrong_day_ar = total("wrong_day", True)
    missed_ar = total("missed", True)
    ct_wrong_ar = total("ct_deadline_wrong", True)
    late = total("late", False)
    late_ar = total("late", True)
    return {
        "licences": n,
        "wrong_day_licences": wrong_day,
        "missed_licences": missed,
        "ct_deadline_wrong": ct_wrong,
        "wrong_day_alerts": wrong_day * ALERTS_PER_LICENCE,
        "missed_alerts": missed * ALERTS_PER_LICENCE,
        "wrong_day_licences_after_review": wrong_day_ar,
        "missed_licences_after_review": missed_ar,
        "ct_deadline_wrong_after_review": ct_wrong_ar,
        "wrong_day_alerts_after_review": wrong_day_ar * ALERTS_PER_LICENCE,
        "missed_alerts_after_review": missed_ar * ALERTS_PER_LICENCE,
        "per_100_clients": {
            "wrong_day": _per_100(wrong_day, n),
            "missed": _per_100(missed, n),
            "ct_deadline_wrong": _per_100(ct_wrong, n),
            "wrong_day_after_review": _per_100(wrong_day_ar, n),
            "missed_after_review": _per_100(missed_ar, n),
            "ct_deadline_wrong_after_review": _per_100(ct_wrong_ar, n),
        },
        "indicative_exposure": {
            "label": EXPOSURE_LABEL,
            "per_licence_aed": LICENCE_OPERATING_EXPIRED_AED,
            "late_licences": late,
            "aed": late * LICENCE_OPERATING_EXPIRED_AED,
            "late_licences_after_review": late_ar,
            "aed_after_review": late_ar * LICENCE_OPERATING_EXPIRED_AED,
        },
    }
