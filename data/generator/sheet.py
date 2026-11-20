"""One row model feeds both the HTML template and GoldDoc.text.

A document is a list of visual rows. Each row is one printed line (cells never wrap; the
renderer shrinks a cell that would overflow and fails if it still does), so GoldDoc.text is
simply one line per visible row, cells joined by a tab, in reading order. Cells are listed
in reading order too: for an Arabic page the first cell sits on the right.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from data.generator.pools import FOOTER, Pair

CELL_SEP = "\t"

# Grid columns per row family. Values are CSS grid-template-columns.
KV_COLS = "50mm 1fr"
BI_KV_COLS = "46mm 1fr 46mm"
TOTAL_COLS = "1fr 38mm"
BI_TOTAL_COLS = "1fr 38mm 1fr"
HALVES = "1fr 1fr"


def strong_direction(ch: str) -> str:
    """ "R" for Arabic letters, "L" for Latin letters and every digit (digit runs are laid out
    left to right in both scripts), "N" for spaces and punctuation."""
    cls = unicodedata.bidirectional(ch)
    if cls in ("R", "AL"):
        return "R"
    if cls in ("L", "EN", "AN"):
        return "L"
    return "N"


def is_rtl_text(text: str) -> bool:
    """A cell is set right to left when Arabic letters outnumber Latin letters. The renderer
    and the PDF text check both use this rule, so they agree on every cell's direction."""
    rtl = sum(1 for ch in text if strong_direction(ch) == "R")
    ltr = sum(1 for ch in text if unicodedata.bidirectional(ch) == "L")
    return rtl > ltr


_TRAILING = "،؛,:;."


def _tokens(text: str) -> list[str]:
    """Words and whitespace, with trailing punctuation split off a word so that "2379،" in
    Arabic text keeps its comma on the Arabic side of the number."""
    tokens = []
    for token in re.findall(r"\s+|\S+", text):
        core = token.rstrip(_TRAILING)
        if core and core != token and any(strong_direction(ch) != "N" for ch in core):
            tokens += [core, token[len(core) :]]
        else:
            tokens.append(token)
    return tokens


def bidi_runs(text: str) -> list[tuple[str, str]]:
    """Split a cell into pieces of one direction, as (direction, text); concatenated they
    give back the text exactly.

    direction is "rtl" or "ltr" for words with Arabic or Latin letters or digits. Neutral
    pieces (spaces, "|", trailing punctuation) take the direction of the words on both
    sides when those agree, and are "" (left to the cell's own direction) when they do not.
    The template wraps every piece whose direction differs from the cell's in <bdi>.
    """
    tokens = _tokens(text)
    strong = []
    for token in tokens:
        kinds = {strong_direction(ch) for ch in token} - {"N"}
        strong.append("rtl" if "R" in kinds else ("ltr" if kinds else ""))
    resolved = []
    for i, direction in enumerate(strong):
        if not direction:
            before = next((d for d in reversed(strong[:i]) if d), "")
            after = next((d for d in strong[i + 1 :] if d), "")
            direction = before if before == after else ""
        resolved.append(direction)
    pieces: list[tuple[str, str]] = []
    for token, direction in zip(tokens, resolved, strict=True):
        if pieces and pieces[-1][0] == direction:
            pieces[-1] = (direction, pieces[-1][1] + token)
        else:
            pieces.append((direction, token))
    return pieces


@dataclass
class Cell:
    text: str
    role: str = "value"  # label | value | num | head | title
    align: str = "start"  # start | end | center, relative to the page direction

    @property
    def rtl(self) -> bool:
        return is_rtl_text(self.text)


@dataclass
class Row:
    kind: str  # title | brand | authority | sub | section | kv | thead | trow | total | note
    cells: list[Cell]
    cols: str = "1fr"
    style: str = ""  # extra CSS classes: strong, low_contrast, white_text, fake_field
    hidden: bool = False  # white text: in the PDF but not visible, so not in GoldDoc.text
    tag: str = ""  # anchor for inserting injected rows: "total", "footer"

    def line(self) -> str:
        return CELL_SEP.join(c.text for c in self.cells if c.text)


@dataclass
class Sheet:
    """Turns labelled values into rows for one layout: ar, en or bi."""

    layout: str
    rows: list[Row] = field(default_factory=list)

    def pick(self, pair: Pair) -> str:
        return pair[1] if self.layout == "ar" else pair[0]

    def heading(self, kind: str, pair: Pair, align: str = "start") -> None:
        if self.layout == "bi":
            cells = [Cell(pair[0], "title"), Cell(pair[1], "title", "end")]
            self.rows.append(Row(kind, cells, HALVES))
        else:
            self.rows.append(Row(kind, [Cell(self.pick(pair), "title", align)]))

    def section(self, label: Pair) -> None:
        self.heading("section", label)

    def kv(self, label: Pair, value: str | Pair, role: str = "value") -> None:
        v_en, v_ar = (value, value) if isinstance(value, str) else value
        if self.layout == "bi":
            cells = [Cell(label[0], "label"), Cell(v_en, role), Cell(label[1], "label", "end")]
            self.rows.append(Row("kv", cells, BI_KV_COLS))
        else:
            text = v_ar if self.layout == "ar" else v_en
            self.rows.append(
                Row("kv", [Cell(self.pick(label), "label"), Cell(text, role)], KV_COLS)
            )

    def name(self, label: Pair, label_en: Pair, label_ar: Pair, en: str, ar: str) -> None:
        """A name in the layout's script, or in both scripts on two rows for bi."""
        if self.layout == "bi":
            self.kv(label_en, en, "name")
            self.kv(label_ar, ar, "name")
        else:
            self.kv(label, (en, ar), "name")

    def table(self, columns: list[tuple[Pair, str, str]], body: list[list[str | Pair]]) -> None:
        """columns: (label, width, align). Body cells are text, or (English, Arabic) pairs."""
        cols = " ".join(width for _, width, _ in columns)
        heads = [[Cell(label[0], "head", align) for label, _, align in columns]]
        if self.layout == "ar":
            heads = [[Cell(label[1], "head", align) for label, _, align in columns]]
        elif self.layout == "bi":
            heads.append([Cell(label[1], "head", align) for label, _, align in columns])
        for cells in heads:
            self.rows.append(Row("thead", cells, cols))
        for values in body:
            cells = []
            for value, (_, _, align) in zip(values, columns, strict=True):
                text = value if isinstance(value, str) else self.pick(value)
                cells.append(Cell(text, "num" if align == "end" else "value", align))
            self.rows.append(Row("trow", cells, cols))

    def total(self, label: Pair, value: str, strong: bool = False, tag: str = "") -> None:
        style = "strong" if strong else ""
        if self.layout == "bi":
            cells = [
                Cell(label[0], "label"),
                Cell(value, "num", "end"),
                Cell(label[1], "label", "end"),
            ]
            self.rows.append(Row("total", cells, BI_TOTAL_COLS, style, tag=tag))
        else:
            cells = [Cell(self.pick(label), "label"), Cell(value, "num", "end")]
            self.rows.append(Row("total", cells, TOTAL_COLS, style, tag=tag))

    def note(self, pair: Pair, kind: str = "note", align: str = "start") -> None:
        if self.layout == "bi":
            self.rows.append(Row(kind, [Cell(pair[0]), Cell(pair[1], align="end")], HALVES))
        else:
            self.rows.append(Row(kind, [Cell(self.pick(pair), align=align)]))

    def line(self, text: str, kind: str = "note") -> None:
        """A one-cell row in a single language, whatever the layout."""
        self.rows.append(Row(kind, [Cell(text)]))

    def footer(self) -> None:
        self.rows.append(Row("footer", [Cell(FOOTER, align="center")], tag="footer"))


def visible_text(rows: list[Row]) -> str:
    return "\n".join(row.line() for row in rows if not row.hidden)


def hidden_text(rows: list[Row]) -> str:
    return "\n".join(row.line() for row in rows if row.hidden)
