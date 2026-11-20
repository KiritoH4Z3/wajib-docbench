"""The four bundled fonts and the characters they can draw.

The renderer's fontconfig lists only these files, so a character outside their cmaps would
print as a box (tofu). The check reads the cmaps directly to prove that cannot happen.
"""

from __future__ import annotations

import struct
from functools import cache
from pathlib import Path

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_FILES = {
    "sans_regular": "NotoSans-Regular.ttf",
    "sans_bold": "NotoSans-Bold.ttf",
    "naskh_regular": "NotoNaskhArabic-Regular.ttf",
    "naskh_bold": "NotoNaskhArabic-Bold.ttf",
}
# PostScript names as they appear in the PDF (after the subset prefix).
PDF_FONT_NAMES = {
    "NotoSans-Regular",
    "NotoSans-Bold",
    "NotoNaskhArabic-Regular",
    "NotoNaskhArabic-Bold",
}


def _table(data: bytes, tag: bytes) -> int:
    count = struct.unpack(">H", data[4:6])[0]
    for i in range(count):
        record = data[12 + 16 * i : 28 + 16 * i]
        if record[:4] == tag:
            return struct.unpack(">I", record[8:12])[0]
    raise ValueError(f"font has no {tag.decode()} table")


def _format4(data: bytes, offset: int) -> set[int]:
    segments = struct.unpack(">H", data[offset + 6 : offset + 8])[0] // 2
    ends_at = offset + 14
    starts_at = ends_at + 2 * segments + 2
    deltas_at = starts_at + 2 * segments
    ranges_at = deltas_at + 2 * segments
    codepoints = set()
    for s in range(segments):
        end, start, delta, range_offset = (
            struct.unpack(">H", data[base + 2 * s : base + 2 * s + 2])[0]
            for base in (ends_at, starts_at, deltas_at, ranges_at)
        )
        for code in range(start, end + 1):
            if code == 0xFFFF:
                continue
            if range_offset == 0:
                glyph = (code + delta) & 0xFFFF
            else:
                at = ranges_at + 2 * s + range_offset + 2 * (code - start)
                glyph = struct.unpack(">H", data[at : at + 2])[0]
                glyph = (glyph + delta) & 0xFFFF if glyph else 0
            if glyph:
                codepoints.add(code)
    return codepoints


def _format12(data: bytes, offset: int) -> set[int]:
    groups = struct.unpack(">I", data[offset + 12 : offset + 16])[0]
    codepoints = set()
    for g in range(groups):
        at = offset + 16 + 12 * g
        start, end, _ = struct.unpack(">III", data[at : at + 12])
        codepoints.update(range(start, end + 1))
    return codepoints


@cache
def codepoints(path: Path) -> frozenset[int]:
    """Characters the font maps to a real glyph (Windows Unicode cmap, format 12 or 4)."""
    data = path.read_bytes()
    cmap = _table(data, b"cmap")
    count = struct.unpack(">H", data[cmap + 2 : cmap + 4])[0]
    subtables = {}
    for i in range(count):
        platform, encoding, offset = struct.unpack(
            ">HHI", data[cmap + 4 + 8 * i : cmap + 12 + 8 * i]
        )
        subtables[(platform, encoding)] = cmap + offset
    for key, reader in (((3, 10), _format12), ((3, 1), _format4)):
        if key in subtables:
            offset = subtables[key]
            fmt = struct.unpack(">H", data[offset : offset + 2])[0]
            if (fmt, reader) in ((12, _format12), (4, _format4)):
                return frozenset(reader(data, offset))
    raise ValueError(f"{path.name}: no Windows Unicode cmap in format 4 or 12")


def covered() -> frozenset[int]:
    """Characters at least one bundled font can draw."""
    result: set[int] = set()
    for name in FONT_FILES.values():
        result |= codepoints(FONT_DIR / name)
    return frozenset(result)


def uncovered(text: str) -> set[str]:
    have = covered()
    return {ch for ch in text if not ch.isspace() and ord(ch) not in have}
