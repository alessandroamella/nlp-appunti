# NLP course notes (Obsidian vault)

This is an Obsidian vault of personal study notes for a university NLP course — not a codebase. The job here is to help the user *understand* course material and turn it into good notes, not to write or maintain software.

## Layout

- `appunti_nlp/` — the Obsidian vault. This is where all notes live.
  - `appunti_nlp/assets/YYYY-MM-DD/` — images (mostly slide screenshots) cropped out of lecture PDFs, one subfolder per session date.
  - `appunti_nlp/.obsidian/` — vault config, don't touch.
- `docs/` — source material per lecture: the slide deck PDF, and sometimes an auto-downloaded Panopto subtitle file (`.srt`) with the same base name plus `-subtitles`.
- `subtitles/` — raw Panopto-exported `.srt` files not yet matched/renamed to a lecture PDF (named after the Panopto recording, e.g. `NLP Fall262026-09-08 08-52.srt`). When working on a lecture, check here for a subtitle file covering that date if none is paired in `docs/` yet. Each `.srt` has a `.json` sidecar next to it with the recording's identity (`viewerUrl`, title, duration) — that's what makes timestamp links possible, see below.
- `lab/` — lab session PDFs.

The repo root is a git repo; PDFs and images are tracked with **git LFS** by extension (see `.gitattributes`), transcripts and sidecars stay plain text. Drop new files in and they're handled automatically. Note the vault root is `appunti_nlp/`, one level below the repo root.

The user will keep dropping new PDFs/subtitles into `docs/`, `subtitles/`, or `lab/` and ask for notes to be written or extended in `appunti_nlp/`.

## Before writing anything

**A topic is not a file.** One topic is often spread over more than one deck and more than one recording, and the numbering doesn't line up across sources. Before writing, gather *everything* that touches it — `ls docs/ | grep -i <topic>`, `ls subtitles/`, `ls lab/` — and match by **topic, not by number or date**. Don't assume the first match is the whole story.

**Check what already exists.** `ls appunti_nlp/` and grep for the topic before starting a new note. If it's already covered, extend that note. Two notes that each half-cover the same thing are worse than one that covers it.

`appunti_nlp/000 - Indice.md` is the map of the course — one line per topic. Add the note to it when you create one.

## Note conventions (match existing files exactly)

- Filename: `YYYY-MM-DD - Title.md`, placed directly in `appunti_nlp/`.
- **Language: lecture notes are in English.** That's what the substantial ones use (`Introduction to NLP`, `Neural Networks and Word Embeddings`), and the exam is in English, so terminology is English regardless. Don't switch an existing note's language — when extending one, match what's already in the file.
  - The Italian files are a different kind of thing: `Robe miste` is scratch, `Sessione gruppo progetto` is personal project notes. Lecture conventions don't apply to them; leave them in Italian and don't reformat them.
  - When a note is genuinely new and unrelated to any existing one, default to English.
- Structure: `##`/`###` headings per topic/subtopic, sometimes prefixed with slide/page references, e.g. `## Pag. 24 — Matrice TF-IDF` or `### Historical framing`.
- Heavy use of **bold** for key terms, `-` bullet lists, and Markdown tables for side-by-side comparisons.
- Blockquotes (`>`) for verbatim definitions/quotes from slides or the lecturer.
- Math in LaTeX: inline `$...$`, display `$$...$$` — Obsidian renders both natively.
- Images embedded with Obsidian wikilink syntax: `![[assets/YYYY-MM-DD/filename.png]]`.
- **Links between notes**: `[[2026-08-25 - Introduction to NLP]]` whenever a concept is properly defined in another note. Don't redefine it — link it. This is a vault, not a stack of PDFs, and at revision time the connections are half the value.
- Don't invent structure/sections beyond what's needed to capture the material — no filler, no restating the obvious.

## Timestamped references to the recording

The `.srt` files carry timestamps: when the lecturer says something worth hearing again **in their own voice**, the note should link that exact moment. Re-listening to ninety seconds is the most efficient thing you can do while revising — but only if the link takes you there in one click. Nobody scrubs through a 90-minute recording by hand.

**Building the link.** The `.srt` has no session id; that's in the `.json` next to it. Take `viewerUrl` from there, convert the cue timestamp to seconds, append `&start=<seconds>`:

```
00:38:12 in the .srt   ->   38*60 + 12 = 2292
[38:12](https://au.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=<guid>&start=2292)
```

Start the link a few seconds **before** the sentence begins — a link that lands mid-word makes you rewind by hand, which is exactly what it's meant to avoid.

**When to cite — only what isn't on the slides.** The test is: *if this were already on the deck, the link wouldn't be needed*. So yes for:

- anything about the exam ("I will ask this", "this is out of syllabus", "watch out for…"), which should be marked prominently anyway;
- a **correction** or a qualification to what the slide says;
- a worked example, an intuition or an analogy the deck doesn't have;
- a subtle point explained far better aloud than in writing;
- a derivation done live, where the recording is the only source.

And no for everything else. "This sentence came from the `.srt` rather than the slide" is not a reason. Expect **a handful per lecture**, not dozens — past that they stop meaning "stop here and listen again", which is their entire value.

**Form.** For a verbatim quote, a blockquote with the link at the end:

> «Attention is really just a weighted average — the weights are the interesting part.» — [38:12](https://…&start=2292)

For a pointer inside the text, a bare link at the end of the sentence: `… (see [38:12](…&start=2292))`. No `.json` available? Then a plain `[38:12]` and nothing more — never invent a guid, and never write a link that doesn't go where it claims.

## Workflow for turning a PDF into notes

1. Convert relevant slide pages to images with `pdftoppm` (e.g. `pdftoppm -png -r 150 -f <page> -l <page> "docs/NN - Title.pdf" appunti_nlp/assets/YYYY-MM-DD/prefix`).
2. Trim whitespace borders with ImageMagick: `convert in.png -trim +repage out.png` (or `mogrify -trim` in place). Use a short, descriptive prefix per lecture (e.g. `nn-primer-024.png`); append `-crop` for a further-cropped variant of the same page when only part of a slide is relevant.
3. Only save/embed images that add real value (diagrams, tables, formulas) — don't screenshot pure bullet-text slides that are easily transcribed.
4. If a matching `-subtitles.srt` exists, `grep` it for the surrounding timestamp/topic to catch things the lecturer said but that aren't on the slide (asides, extra examples, emphasis, things glossed over quickly) — fold those into the note as prose or bullets, not just slide transcription. `grep -i -n -B2 -A4 '<term>' docs/*.srt subtitles/*.srt`; the `-B2` keeps the cue's timestamp in view, which is what you need for a timestamped link.
5. `cat` the matching `.json` sidecar for the `viewerUrl` when you're going to link a moment.
6. Write/extend the note in `appunti_nlp/`, following the conventions above.

## When a note is finished

Not when you've covered every slide. When:

- [ ] Every concept has a **why**, not just a what. The slides are terse; the note should be more explanatory than the original, not a copy of it.
- [ ] What the lecturer said aloud and isn't on the slides is in there, and the moments worth re-hearing carry a **timestamped link**.
- [ ] Terms defined in another note are linked with `[[...]]`, not redefined.
- [ ] It closes with a short **`## Da ricordare`** section: the 3–8 things you'd re-check the night before the exam — exact definitions, formulas, conditions, traps. It's the most-read part of the note, so keep it tight.
- [ ] Anything the lecturer said about the exam is marked prominently.
- [ ] It's listed in `appunti_nlp/000 - Indice.md`.

(Scratch and project notes are exempt — this applies to lecture notes.)

## What "helping" means here

- Explain concepts clearly, correctly, and at the right level — the user is using this as a learning tool, not just a transcription service.
- When slides are terse or the lecturer's phrasing is ambiguous, add clarifying explanation rather than copying verbatim.
- Flag when something in the slides/subtitles seems inconsistent or when you're inferring beyond what's stated.
- If you didn't understand something from the sources, **say so** rather than filling the gap with something plausible. A note with an honest question mark is worth more than a paragraph that turns out to be wrong in the exam hall.
- The repo is git-tracked now, but there's no commit/push automation: treat file edits like any other content edit, commit only when asked, and still confirm before anything destructive.
