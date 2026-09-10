# NLP course notes (Obsidian vault)

This is an Obsidian vault of personal study notes for a university NLP course — not a codebase. The job here is to help the user *understand* course material and turn it into good notes, not to write or maintain software.

## Layout

- `appunti_nlp/` — the Obsidian vault. This is where all notes live.
  - `appunti_nlp/assets/YYYY-MM-DD/` — images (mostly slide screenshots) cropped out of lecture PDFs, one subfolder per session date.
  - `appunti_nlp/.obsidian/` — vault config, don't touch.
- `docs/` — source material per lecture: the slide deck PDF, and sometimes an auto-downloaded Panopto subtitle file (`.srt`) with the same base name plus `-subtitles`.
- `subtitles/` — raw Panopto-exported `.srt` files not yet matched/renamed to a lecture PDF (named after the Panopto recording, e.g. `NLP Fall262026-09-08 08-52.srt`). When working on a lecture, check here for a subtitle file covering that date if none is paired in `docs/` yet.
- `lab/` — lab session PDFs.

The user will keep dropping new PDFs/subtitles into `docs/`, `subtitles/`, or `lab/` and ask for notes to be written or extended in `appunti_nlp/`.

## Note conventions (match existing files exactly)

- Filename: `YYYY-MM-DD - Title.md`, placed directly in `appunti_nlp/`.
- Written in a mix of English and Italian depending on the note (follow whatever language the specific note/topic is already in; new notes on the same lecture as an existing note should match its language).
- Structure: `##`/`###` headings per topic/subtopic, sometimes prefixed with slide/page references, e.g. `## Pag. 24 — Matrice TF-IDF` or `### Historical framing`.
- Heavy use of **bold** for key terms, `-` bullet lists, and Markdown tables for side-by-side comparisons.
- Blockquotes (`>`) for verbatim definitions/quotes from slides or the lecturer.
- Math in LaTeX: inline `$...$`, display `$$...$$` — Obsidian renders both natively.
- Images embedded with Obsidian wikilink syntax: `![[assets/YYYY-MM-DD/filename.png]]`.
- Don't invent structure/sections beyond what's needed to capture the material — no filler, no restating the obvious.

## Workflow for turning a PDF into notes

1. Convert relevant slide pages to images with `pdftoppm` (e.g. `pdftoppm -png -r 150 -f <page> -l <page> "docs/NN - Title.pdf" appunti_nlp/assets/YYYY-MM-DD/prefix`).
2. Trim whitespace borders with ImageMagick: `convert in.png -trim +repage out.png` (or `mogrify -trim` in place). Use a short, descriptive prefix per lecture (e.g. `nn-primer-024.png`); append `-crop` for a further-cropped variant of the same page when only part of a slide is relevant.
3. Only save/embed images that add real value (diagrams, tables, formulas) — don't screenshot pure bullet-text slides that are easily transcribed.
4. If a matching `-subtitles.srt` exists, `grep` it for the surrounding timestamp/topic to catch things the lecturer said but that aren't on the slide (asides, extra examples, emphasis, things glossed over quickly) — fold those into the note as prose or bullets, not just slide transcription.
5. Write/extend the note in `appunti_nlp/`, following the conventions above.

## What "helping" means here

- Explain concepts clearly, correctly, and at the right level — the user is using this as a learning tool, not just a transcription service.
- When slides are terse or the lecturer's phrasing is ambiguous, add clarifying explanation rather than copying verbatim.
- Flag when something in the slides/subtitles seems inconsistent or when you're inferring beyond what's stated.
- No commit/push automation assumed — this is not a git-tracked review workflow; treat file edits like any other content edit (still confirm before anything destructive).
