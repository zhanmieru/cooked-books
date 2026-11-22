"""Turn a Case into text: Russian labels, ~100 columns, optional ANSI colours.

Money is formatted from int minor units with integer arithmetic only
(`divmod`), a space between thousands and a decimal comma. Rates are printed
as stored. Timestamps are printed as stored, cut to minutes.
"""

from __future__ import annotations

import json
import re
import textwrap
from datetime import timedelta, timezone
from typing import Any

from cookedbooks.xray.case import CHANGE_TABLES, Case, Entry, EpisodeView, Ref, parse_instant

TYPE_NAMES = {
    "duplicate_invoice": "дубль счёта",
    "unallocated_payment": "пропущенное или частичное распределение платежа",
    "fx_error": "неверная дата или курс конвертации",
    "cutoff_error": "ошибка отнесения к отчётному периоду",
}
MONEY_FIELDS = frozenset({"debit_minor", "credit_minor", "total_minor", "amount_minor", "unit_price_minor"})
CHILD_CAP = 6
TWIN_CHILD_CAP = 3
TWIN_ENTRY_CAP = 3
RULE = "─" * 100
_TIME = re.compile(r"(\d{4}-\d\d-\d\d)T(\d\d:\d\d)(?::\d\d(?:\.\d+)?)?(Z|[+-]\d\d:\d\d)?")
_OFFSET = re.compile(r"([+-])(\d\d):(\d\d)")

_ANSI = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "magenta": "35", "cyan": "36"}


class Style:
    def __init__(self, color: bool) -> None:
        self.color = color

    def __call__(self, text: str, *styles: str) -> str:
        if not self.color or not styles or not text:
            return text
        codes = ";".join(_ANSI[s] for s in styles)
        return f"\x1b[{codes}m{text}\x1b[0m"


# --- values -----------------------------------------------------------------------------


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def fmt_money(amount: object, currency: object, exponents: dict) -> str:
    """`10 211,30 USD`; exponent 0 has no fraction; unknown exponent: raw integer + «мин. ед.»."""
    cur = show_id(currency) if currency is not None else "?"
    if not _is_int(amount):
        return f"{show_value(amount)} {cur}"
    exponent = exponents.get(currency) if isinstance(currency, str) else None
    sign = "-" if amount < 0 else ""
    if not _is_int(exponent) or exponent < 0:
        return f"{amount} мин. ед. {cur}"
    whole, frac = divmod(abs(amount), 10 ** exponent)
    digits = str(whole)
    groups = []
    while len(digits) > 3:
        groups.insert(0, digits[-3:])
        digits = digits[:-3]
    groups.insert(0, digits)
    text = " ".join(groups)
    if exponent:
        text += "," + str(frac).rjust(exponent, "0")
    return f"{sign}{text} {cur}"


def show_id(value: object) -> str:
    """An ID as is when it is plain; quoted with escapes when it has spaces or invisible characters."""
    if not isinstance(value, str):
        return show_value(value)
    if value and value.isprintable() and " " not in value and '"' not in value:
        return value
    return _quote(value)


def _quote(value: str) -> str:
    out = []
    for ch in value:
        if ch in '"\\':
            out.append("\\" + ch)
        elif ch.isprintable():
            out.append(ch)
        elif ord(ch) < 0x10000:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(f"\\U{ord(ch):08x}")
    return '"' + "".join(out) + '"'


def show_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return show_id(value)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return "[" + ", ".join(show_id(v) for v in value) + "]"
    if isinstance(value, (bool, int)):
        return json.dumps(value)
    return _escape_json(json.dumps(value, ensure_ascii=False, sort_keys=True))


def id_list(ids: list[str]) -> str:
    """The exact list of IDs as JSON (copy-pasteable), invisible characters escaped."""
    return _escape_json(json.dumps(ids, ensure_ascii=False))


def _escape_json(text: str) -> str:
    return "".join(ch if ch.isprintable() else f"\\u{ord(ch):04x}" for ch in text)


def fmt_time(value: object) -> str:
    if isinstance(value, str):
        match = _TIME.fullmatch(value)
        if match:
            date, hm, offset = match.groups()
            return f"{date} {hm}" + (f" {offset}" if offset else "")
    return show_value(value)


def company_tz(manifest: dict) -> timezone | None:
    match = _OFFSET.fullmatch(manifest.get("utc_offset") or "") if isinstance(manifest.get("utc_offset"), str) else None
    if not match:
        return None
    sign, hours, minutes = match.groups()
    delta = timedelta(hours=int(hours), minutes=int(minutes))
    try:
        return timezone(-delta if sign == "-" else delta)
    except ValueError:
        return None


def local_time(value: object, manifest: dict) -> str | None:
    """Local time by the company offset, only when it is written differently from the stored text."""
    moment, tz = parse_instant(value), company_tz(manifest)
    if moment is None or tz is None:
        return None
    local = moment.astimezone(tz)
    text = local.strftime("%Y-%m-%d %H:%M") + " " + manifest["utc_offset"]
    return None if text == fmt_time(value) else text


# --- records ----------------------------------------------------------------------------


def record_line(view: EpisodeView, ref: Ref) -> list[str]:
    return record_text(view, ref[0], view.record(ref))


def record_text(view: EpisodeView, table: str, r: dict) -> list[str]:
    """Key fields of one record: one line, two for an invoice."""
    ex = view.exponents
    if table == "ledger":
        debit, credit = r.get("debit_minor"), r.get("credit_minor")
        if _is_int(debit) and debit > 0 and not (_is_int(credit) and credit > 0):
            side = "Дт " + fmt_money(debit, r.get("currency"), ex)
        elif _is_int(credit) and credit > 0 and not (_is_int(debit) and debit > 0):
            side = "Кт " + fmt_money(credit, r.get("currency"), ex)
        else:
            side = f"Дт {fmt_money(debit, r.get('currency'), ex)} / Кт {fmt_money(credit, r.get('currency'), ex)}"
        return [f"строка {show_id(r.get('row_id'))}  {fmt_time(r.get('timestamp'))}  "
                f"{show_value(r.get('account'))}  {side}"]
    if table == "invoices":
        first = (f"счёт {show_id(r.get('doc_id'))} · {show_id(r.get('invoice_id'))} · "
                 f"клиент {show_id(r.get('entity_id'))} · {fmt_money(r.get('total_minor'), r.get('currency'), ex)}")
        second = "выставлен " + fmt_time(r.get("issued_at"))
        local = local_time(r.get("issued_at"), view.manifest)
        if local:
            second += f" (местное {local})"
        if r.get("po_ref") is not None:
            second += f" · заказ {show_value(r.get('po_ref'))}"
        if r.get("service_period") is not None:
            second += f" · услуга {show_value(r.get('service_period'))}"
        return [first, second]
    if table == "payments":
        refs = r.get("invoice_refs")
        return [f"платёж {show_id(r.get('doc_id'))} · {show_id(r.get('payment_id'))} · "
                f"{fmt_time(r.get('timestamp'))} · {fmt_money(r.get('amount_minor'), r.get('currency'), ex)} · "
                f"счета: {show_value(refs) if refs is not None else 'нет (null)'}"]
    return [f"курс {show_id(r.get('id'))}: 1 {show_value(r.get('source_currency'))} = "
            f"{show_value(r.get('rate'))} {show_value(r.get('target_currency'))}"]


def _tag(ref: Ref, view: EpisodeView, affected: list[Ref], evidence: list[Ref], st: Style) -> str:
    a = st("З", "red", "bold") if ref in affected else " "
    e = st("Д", "green", "bold") if ref in evidence else " "
    mark = view.changed.get(ref, " ")
    return a + e + (st(mark, "magenta", "bold") if mark != " " else " ")


def _record_block(view: EpisodeView, ref: Ref, affected: list[Ref], evidence: list[Ref],
                  st: Style, cap: int, shown: list[Ref]) -> list[str]:
    lines = record_line(view, ref)
    out = [f"{_tag(ref, view, affected, evidence, st)} {lines[0]}"] + [f"      {line}" for line in lines[1:]]
    shown.append(ref)
    children = view.children(ref)
    for i, child in enumerate(children[:cap]):
        branch = "└" if i == len(children) - 1 else "├"
        out.append(f"{_tag(child, view, affected, evidence, st)}   {branch} {record_line(view, child)[0]}")
        shown.append(child)
    if len(children) > cap:
        out.append(f"      └ … ещё {len(children) - cap}")
    return out


def _entries_block(view: EpisodeView, entries: list[Entry], affected: list[Ref], evidence: list[Ref],
                   st: Style, cap: int) -> list[str]:
    out: list[str] = []
    shown: list[Ref] = []
    for entry in entries:
        if not entry.refs:
            out.append(f"    ID {show_id(entry.id)} — не найден в открытых файлах")
            continue
        if entry.ambiguous:
            out.append(st(f"    ID {show_id(entry.id)} неоднозначен: его носят {len(entry.refs)} записи "
                          "(контракт, раздел 7); показаны все", "yellow"))
        elif entry.refs[0] in shown:
            continue  # already listed under its document
        for ref in entry.refs:
            out.extend(_record_block(view, ref, affected, evidence, st, cap, shown))
    return out


# --- the case file ----------------------------------------------------------------------


def type_label(issue_type: object) -> str:
    name = TYPE_NAMES.get(issue_type) if isinstance(issue_type, str) else None
    return f"{name or 'неизвестный тип'} ({show_value(issue_type)})"


def render_case(view: EpisodeView, case: Case, *, color: bool = False) -> str:
    st = Style(color)
    issue, m = case.issue, view.manifest
    lines = [st(RULE, "dim"),
             st(f"Дело {show_id(issue['issue_id'])} · {type_label(issue['type'])}", "bold")]
    lines.append(f"эпизод {show_value(m.get('episode_id'))} · базовая валюта {show_value(m.get('base_currency'))}"
                 f" · смещение UTC {show_value(m.get('utc_offset'))}")
    lines.append(f"период {fmt_time(m.get('period_start'))} — {fmt_time(m.get('period_end'))}")
    lines.append("сумма по эталону: " + st(fmt_money(issue["expected_amount_minor"], issue["currency"],
                                                     view.exponents), "bold"))
    inj = case.injection
    if inj is not None:
        if isinstance(inj.get("description"), str):
            lines.append(_wrap("что внесено: " + inj["description"], ""))
        if inj.get("variant") is not None:
            lines.append("вид: " + show_value(inj.get("variant")))
        if inj.get("scenario") is not None:
            together = ", ".join(show_id(s) for s in case.siblings) or "—"
            lines.append(f"сценарий {show_value(inj.get('scenario'))}: одна правка вносит и эту ошибку, и {together}")
    for warning in issue["warnings"]:
        lines.append(st("предупреждение эталона: " + warning, "yellow"))

    lines += ["", st("1. Цепочка доказательств", "cyan", "bold"),
              st("   З — в affected_ids · Д — в наборе доказательств · + добавлена ошибкой · ~ изменена ошибкой",
                 "dim")]
    lines += _entries_block(view, case.entries, case.affected, case.evidence, st, CHILD_CAP)
    lines.append("   набор доказательств: " + (id_list(case.evidence_set) if case.evidence_set else "нет"))

    lines += ["", st("2. До и после", "cyan", "bold")]
    if case.changes is None:
        lines.append("   " + (case.history_note or "история изменений недоступна"))
    elif not case.changes:
        lines.append("   в provenance.json у этой ошибки нет правок")
    for change in case.changes or []:
        where = f"{show_value(change.file)} {show_id(change.key)}"
        if change.op == "add":
            lines.append(f" {st('+', 'magenta', 'bold')} добавлена: {_brief(change, view)}")
        elif change.op == "remove":
            lines.append(f" − удалена: {_brief(change, view)}")
        elif change.op == "modify":
            for fc in change.fields or []:
                lines.append(f" {st('~', 'magenta', 'bold')} {where}: {fc.name}: "
                             f"{_value(fc.name, fc.before, change, view)} → {_value(fc.name, fc.after, change, view)}")
            if not change.fields:
                lines.append(f" ~ {where}: изменена, но поля не различаются")
        else:
            lines.append(f" ? {where}: правка {show_value(change.op)}")

    lines += ["", st("3. Двойник: не ошибка", "cyan", "bold")]
    if case.twin is None:
        lines.append("   " + (case.twin_note or "двойника нет"))
    else:
        twin = case.twin
        lines.append(f"   {show_value(twin.kind)}: {id_list([e.id for e in twin.entries])}")
        lines += _entries_block(view, twin.entries[:TWIN_ENTRY_CAP], [], [], st, TWIN_CHILD_CAP)
        if len(twin.entries) > TWIN_ENTRY_CAP:
            lines.append(f"    … ещё {len(twin.entries) - TWIN_ENTRY_CAP} ID из списка выше")
        lines.append(_wrap("   почему не ошибка: " + _twin_reason(view, twin), "   "))
    lines.append(st(RULE, "dim"))
    return "\n".join(lines) + "\n"


def _value(name: str, value: Any, change: Any, view: EpisodeView) -> str:
    if name in MONEY_FIELDS and _is_int(value):
        return fmt_money(value, change.currency, view.exponents)
    if name in ("timestamp", "issued_at", "due_at"):
        return fmt_time(value)
    if name == "line_items":
        return f"{len(value)} поз." if isinstance(value, list) else show_value(value)
    return show_value(value)


def _brief(change: Any, view: EpisodeView) -> str:
    target = CHANGE_TABLES.get(change.file) if isinstance(change.file, str) else None
    if target is None or not isinstance(change.record, dict):
        return f"{show_value(change.file)} {show_id(change.key)}"
    return record_text(view, target[0], change.record)[0]


def _wrap(text: str, indent: str) -> str:
    return textwrap.fill(text, width=100, subsequent_indent=indent, break_long_words=False,
                         break_on_hyphens=False)


def _twin_reason(view: EpisodeView, twin: Any) -> str:
    d = twin.detail
    if twin.kind == "recurring_amount" and d:
        if not d["diff"]:
            return (f"счета {show_id(d['a'])} и {show_id(d['b'])} совпадают по всему ключу duplicate_key; "
                    "это надо проверить вручную")
        diff = "; ".join(f"{k}: {show_value(x)} и {show_value(y)}" for k, x, y in d["diff"])
        same = ", ".join(d["same"]) or "—"
        return (f"счета {show_id(d['a'])} и {show_id(d['b'])} совпадают по {same}, но отличаются ({diff}). "
                "По ключу duplicate_key это разные обязательства, не дубль.")
    if twin.kind == "overpayment" and d:
        return (f"платёж {show_id(d['payment'])} больше счёта {show_id(d['invoice'])} на "
                f"{fmt_money(d['excess'], d['currency'], view.exponents)}, остаток счёта "
                f"{fmt_money(d['open'], d['currency'], view.exponents)}. " + twin.reason)
    if twin.kind == "advance" and d:
        later = ", ".join(show_id(i) for i in d["later"])
        return twin.reason + f" Счёт {later} выставлен позже платежа, а в такие счета аванс не засчитывается."
    return twin.reason or "правило для этого вида двойника не описано"


# --- list mode --------------------------------------------------------------------------


def render_list(view: EpisodeView, *, color: bool = False) -> str:
    st = Style(color)
    m = view.manifest
    lines = [st(f"эпизод {show_value(m.get('episode_id'))} · базовая валюта {show_value(m.get('base_currency'))}"
                f" · период {fmt_time(m.get('period_start'))} — {fmt_time(m.get('period_end'))}", "bold")]
    if not view.issues:
        lines.append("ошибок нет")
    for issue in view.issues:
        lines.append(f"  {show_id(issue['issue_id'])}  {show_value(issue['type'])}  "
                     f"{fmt_money(issue['expected_amount_minor'], issue['currency'], view.exponents)}")
    if view.provenance is None:
        lines.append("двойники: история изменений недоступна (" + (view.provenance_note or "") + ")")
    else:
        counts: dict[str, int] = {}
        for d in view.distractors:
            kind = show_value(d["kind"])
            counts[kind] = counts.get(kind, 0) + 1
        lines.append("двойники: " + (", ".join(f"{k} ×{n}" for k, n in counts.items()) or "нет"))
    return "\n".join(lines) + "\n"
