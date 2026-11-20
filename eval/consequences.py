"""What a wrong extraction does to a Wajib client.

Ported from the Wajib checkout, /home/kiritohaze/Projects/ClaudeCode/wajib at commit
9e040a9:
- `src/lib/alerts/schedule.ts`: ALERT_OFFSETS = [90, 60, 30, 7, 1] and
  ALERT_CHANNELS = ["WHATSAPP", "EMAIL"]. `buildAlertSchedule` sends one reminder per
  offset per channel, scheduled `subDays(dueDate, offset)`.
- `src/lib/rules/uae-compliance-rules.ts`: the trade-licence rule (due on the expiry
  date), PENALTY.LICENCE_OPERATING_EXPIRED, `ctRegistrationDeadlineByMonth` and the
  juridical-person Corporate Tax registration branch.

Rules applied here:
- The trade licence renewal is due on the licence expiry date. Wajib schedules a renewal
  reminder 90, 60, 30, 7 and 1 days before that date, each on WhatsApp and on email:
  10 reminders per licence. No reminder fires on the due date itself.
- Corporate Tax registration deadline (FTA Decision No. 3 of 2024): a company
  incorporated on or after 1 March 2024 registers within 3 months of incorporation
  (date-fns addMonths, so 30 November plus 3 months is 28 or 29 February); an earlier one
  follows the month table keyed by the licence issue month. The benchmark extracts only
  the licence issue date, so it stands in for the incorporation date in the cutoff test.

Per licence:
- wrong_day: the predicted expiry is a real date that differs from gold, so all 10
  reminders fire on the wrong days. This includes a predicted expiry on a licence whose
  gold expiry is null.
- missed: gold has an expiry and the prediction is null or not a real date, so none of the
  10 reminders exist.
- ct_deadline_wrong: the Corporate Tax deadline derived from the predicted issue date is
  not the one derived from gold (a missing predicted issue date counts). Two pre-cutoff
  issue months that share a deadline (January and February) cause nothing and are not
  counted.

The licence counts (wrong_day_licences, missed_licences and their after-review values) are
the published numbers; each *_alerts value is the licence count times 10. A document routed
to review is corrected by a person, so after review it causes nothing. The AED figure is
Wajib's own indicative amount for operating on an expired licence, not a measured loss.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from dateutil.relativedelta import relativedelta

from bench.normalize import parse_date
from eval.metrics import DocResult

# src/lib/alerts/schedule.ts at 9e040a9: days before the due date, and delivery channels.
ALERT_OFFSETS: tuple[int, ...] = (90, 60, 30, 7, 1)
ALERT_CHANNELS: tuple[str, ...] = ("whatsapp", "email")
ALERTS_PER_LICENCE = len(ALERT_OFFSETS) * len(ALERT_CHANNELS)
# uae-compliance-rules.ts: `new Date(2024, 2, 1)`, incorporated on or after it -> 3 months.
CT_CUTOFF = date(2024, 3, 1)
CT_MONTHS_AFTER_INCORPORATION = 3
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
    """On or after 1 March 2024: issue date plus 3 months, clamped to the end of a shorter
    month as date-fns addMonths does. Before it: the month table."""
    if issue_date >= CT_CUTOFF:
        return issue_date + relativedelta(months=CT_MONTHS_AFTER_INCORPORATION)
    return date(*CT_DEADLINE_BY_MONTH[issue_date.month])


def renewal_alert_dates(expiry: date) -> dict[int, date]:
    """{offset in days: the day Wajib sends that reminder}, on each of ALERT_CHANNELS."""
    return {offset: expiry - timedelta(days=offset) for offset in ALERT_OFFSETS}


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
        "alerts_per_licence": ALERTS_PER_LICENCE,
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
