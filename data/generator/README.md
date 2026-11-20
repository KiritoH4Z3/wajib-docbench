# Dataset generator

Writes the 85 synthetic UAE business documents of `docs/CONTRACTS.md` section 1 and their
ground truth.

```
uv run python -m data.generator                  # all 85: JSON, PDFs and scan JPEGs (about 12 s)
uv run python -m data.generator --skip-render    # ground truth only, no browser (under 1 s)
uv run python -m data.generator --only inv-ar-03 --only inv-ar-03.poison
uv run python -m data.generator --check          # validate data/, exit 1 on any failure
uv run python -m data.generator --check --skip-render   # ground truth only
```

Rendering needs Playwright's Chromium: `uv run playwright install chromium`.

## How it is built

| Module | Job |
|---|---|
| `plan.py` | Which document gets which hard case, scan and injection. Fixed tables and rules on (type, layout, index), so the section 1.3 minimums hold by construction. |
| `content.py` | Seeded values per document type and the rows that print them. Gold values come from the same variables. |
| `sheet.py` | The row model. One list of visual rows renders the HTML and writes `GoldDoc.text` (one line per row, cells joined by a tab, reading order). |
| `fmt.py` | Every printed number: digit style, date styles, Hijri (Umm al-Qura via hijridate), amounts, masked TRNs, amounts in words. |
| `render.py` | Jinja2 template to PDF through headless Chromium, and the scanned-look JPEG. |
| `pdftext.py` | Reads a rendered PDF back from pypdfium2 glyph positions. |
| `check.py` | Everything `--check` verifies, one function per silent failure. |
| `pools.py` | Labels, authorities, legal forms, names, activities, line items. |

Randomness comes only from `random.Random(f"docbench-v1:{doc_id}:{stream}")` with the streams
`values`, `layout`, `hard_cases`, `scan` and `poison`. Variants use their own id for the `scan`
and `poison` streams and their base document's id for everything else. The scan settings,
including the noise seed, are stored in `GoldDoc.scan`.

Hard-case counts over the 60 base documents: `arabic_indic_digits` 16, `hijri_date` 8,
`written_month` 20, `iso_date` 8, `aed_marker` 18, `arabic_currency_marker` 12,
`absent_field` 25, `masked_trn` 7, `thousands_commas` 28 (every amount of 1,000 or more is
grouped, which `--check` verifies on the printed text).

## Decisions worth knowing

- **Private fontconfig.** Chromium runs with `FONTCONFIG_FILE` pointing at a config that lists
  only `fonts/`. With the host's fontconfig, web-font glyph advances were snapped to whole
  pixels, and pdfium then split English words into one line per letter. The private config
  fixes that and means no host font can stand in for a missing glyph. `--check` also proves
  every printed character is in the bundled cmaps and every PDF glyph uses a bundled font.
- **Arabic-Indic digit groups are forced left to right.** Under the Unicode bidi algorithm,
  "١٠٠-١٢٣٤-٥٦٧٨-٩١٢٣" or a spaced phone number in Arabic-Indic digits is laid out with its
  groups right to left. An OCR pass reading left to right would then get the TRN wrong. Runs
  with Arabic-Indic digits carry `unicode-bidi: isolate-override`.
- **No parentheses inside Arabic text or on rows with Arabic.** pdfium mirrors them on
  mixed-direction lines, so labels say "Customer Name in Arabic", not "(Arabic)".
- **One line per row.** Cells never wrap. The renderer shrinks a cell that overflows by up to
  28% and fails the run if it still does not fit, or if the page runs past one A4 sheet.
- Labels follow the research notes. They are not confirmed against official specimens
  (DET, ADDED, SEDD licences, FTA certificates); verify before claiming authority-exact layouts.

## Known limitation: pdfium's own text assembly

`--check` must show that every gold value is in each PDF's text layer as read by pypdfium2.
pdfium's `get_text_range()` cannot do this on Chromium's Arabic output: it merges rows,
reverses some Arabic words and not others, and turns "INV-2026-0042" inside a right-to-left
row into "-2026-0042INV". Measured on this dataset, it misses 93 of the 501 printed gold values
(including 8 dates, 12 ids and 6 amounts). The PDFs themselves are fine: poppler reads them
correctly.

So `pdftext.py` rebuilds the text from pypdfium2's per-glyph data instead: glyph origins and
boxes for position, and the index order of space characters for word breaks (pdfium places
space glyphs at unreliable x positions, and Noto Naskh's gap after a non-joining letter is as
wide as some word spaces). It then puts each cell back into reading order. With that, all 501
values are found, and the check stays strict for numbers, dates, TRNs, ids and Arabic text
alike.

What the rebuilt text still gets wrong, all in rows that hold no gold value: 84 of 1,364
lines differ from `GoldDoc.text` by a word space (missing between two Arabic words, or added
before an Arabic comma) or by the position of a "|" separator. Long Arabic phrases are matched
with spaces removed by `bench.grounding`, so this does not affect the check.
