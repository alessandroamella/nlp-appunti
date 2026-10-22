# NLP course notes (Obsidian vault)

This is an Obsidian vault of personal study notes for a university NLP course — not a codebase. The job here is to help the user *understand* course material and turn it into good notes, not to write or maintain software.

## Layout

The **repo root is the Obsidian vault**, so the notes can link the slides, exams and labs directly.

- `appunti/` — where all notes live.
  - `appunti/assets/YYYY-MM-DD/` — images (mostly slide screenshots) cropped out of lecture PDFs, one subfolder per session date.
- `.obsidian/` — vault config, don't touch.
- `docs/` — the slide deck PDF of each lecture.
- `vids/` — the lecture recordings and their transcripts, produced by `vids/get_vids.py` (the user runs it; it downloads every `[Video]` on the Brightspace course page and transcribes it with AssemblyAI, prompted with that lecture's slides).
  - `vids/subs/` — one `.srt` per recording, named after it (e.g. `NLP Fall262026-09-08 08-52.srt`), each with a `.json` sidecar holding the recording's identity (`viewerUrl`, title, duration, `lecture`, `slides`) — that's what makes timestamp links possible, see below.
  - `vids/index.json` — maps each recording to its date, lecture title and slide deck: the quickest way to find the transcript for a lecture.
  - `vids/prompts/` — the context each transcription was prompted with. `vids/videos/` holds the `.mp4`s (gitignored, GBs).
- `lab/` — lab session PDFs.
- `ta_sessions/` — **symlink** to a separate git clone of the TA's course repo (`../webdev_projects/nlp-2026`). `Exercises/NN - Topic/` holds the lab notebooks by topic, `Exam/mock_exam.ipynb` a mock exam, `Project/` the project specs. It's someone else's repo: read it, never edit or commit into it (pull to update). Treat it as one more source when gathering a topic; the mock exam counts alongside `exams/`.
- `exams/` — past exam papers with solutions (e.g. `exam_2025_with_solutions.md`: 10 MCQs, one correct option each, with per-option explanations). Per the note at its top, the **2026 exam is MCQ-only, and longer**. Use these to gauge what gets asked and at what depth: when writing a note on a topic a past question covers, make sure the note answers it and put the trap in `## Da ricordare`.

The repo root is a git repo; PDFs and images are tracked with **git LFS** by extension (see `.gitattributes`), transcripts and sidecars stay plain text. Drop new files in and they're handled automatically.

The user will keep dropping new PDFs/subtitles/exams into `docs/`, `lab/`, or `exams/` and ask for notes to be written or extended in `appunti/`.

## Before writing anything

**A topic is not a file.** One topic is often spread over more than one deck and more than one recording, and the numbering doesn't line up across sources. Before writing, gather *everything* that touches it — `ls docs/ | grep -i <topic>`, `cat vids/index.json`, `ls lab/`, `ls ta_sessions/Exercises/`, `grep -il <topic> exams/* ta_sessions/Exam/*` — and match by **topic, not by number or date**. Don't assume the first match is the whole story.

**Check what already exists.** `ls appunti/` and grep for the topic before starting a new note. If it's already covered, extend that note. Two notes that each half-cover the same thing are worse than one that covers it.

`appunti/000 - Indice.md` is the map of the course — one line per topic. Add the note to it when you create one.

## Note conventions (match existing files exactly)

- Filename: `YYYY-MM-DD - Title.md`, placed directly in `appunti/`.
- **Language: lecture notes are in English.** That's what the substantial ones use (`Introduction to NLP`, `Neural Networks and Word Embeddings`), and the exam is in English, so terminology is English regardless. Don't switch an existing note's language — when extending one, match what's already in the file.
  - When a note is genuinely new and unrelated to any existing one, default to English.
- Structure: `##`/`###` headings per topic/subtopic, sometimes prefixed with slide/page references, e.g. `## Pag. 24 — Matrice TF-IDF` or `### Historical framing`.
- Heavy use of **bold** for key terms, `-` bullet lists, and Markdown tables for side-by-side comparisons.
- Blockquotes (`>`) for verbatim definitions/quotes from slides or the lecturer.
- Math in LaTeX: inline `$...$`, display `$$...$$` — Obsidian renders both natively.
- Images embedded with Obsidian wikilink syntax, path from the vault (= repo) root: `![[appunti/assets/YYYY-MM-DD/filename.png]]`.
- **Links between notes**: `[[2026-08-25 - Introduction to NLP]]` whenever a concept is properly defined in another note. Don't redefine it — link it. This is a vault, not a stack of PDFs, and at revision time the connections are half the value.
- **MCQs are written out in full.** Whenever a note comments on the answer to a question (past exam, mock exam, lab MCQ), it quotes the question and **every option**, marks the correct one (✅/❌ per option, with the reason next to each distractor), and names the source (e.g. `2025 Q5 = mock exam`, `lab MCQ 6.3`). The note must be readable without opening the exam or the notebook. `## Da ricordare` bullets can stay short, as long as the full question is in the body.
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

1. Convert relevant slide pages to images with `pdftoppm` (e.g. `pdftoppm -png -r 150 -f <page> -l <page> "docs/NN - Title.pdf" appunti/assets/YYYY-MM-DD/prefix`).
2. Trim whitespace borders with ImageMagick: `convert in.png -trim +repage out.png` (or `mogrify -trim` in place). Use a short, descriptive prefix per lecture (e.g. `nn-primer-024.png`); append `-crop` for a further-cropped variant of the same page when only part of a slide is relevant.
3. Only save/embed images that add real value (diagrams, tables, formulas) — don't screenshot pure bullet-text slides that are easily transcribed.
4. If the lecture's transcript exists in `vids/subs/` (look it up in `vids/index.json`), `grep` it for the surrounding timestamp/topic to catch things the lecturer said but that aren't on the slide (asides, extra examples, emphasis, things glossed over quickly) — fold those into the note as prose or bullets, not just slide transcription. `grep -i -n -B2 -A4 '<term>' vids/subs/*.srt`; the `-B2` keeps the cue's timestamp in view, which is what you need for a timestamped link.
5. `cat` the matching `.json` sidecar for the `viewerUrl` when you're going to link a moment.
6. Write/extend the note in `appunti/`, following the conventions above.

## When a note is finished

Not when you've covered every slide. When:

- [ ] Every concept has a **why**, not just a what. The slides are terse; the note should be more explanatory than the original, not a copy of it.
- [ ] What the lecturer said aloud and isn't on the slides is in there, and the moments worth re-hearing carry a **timestamped link**.
- [ ] Terms defined in another note are linked with `[[...]]`, not redefined.
- [ ] It closes with a short **`## Da ricordare`** section: the 3–8 things you'd re-check the night before the exam — exact definitions, formulas, conditions, traps. It's the most-read part of the note, so keep it tight.
- [ ] Anything the lecturer said about the exam is marked prominently.
- [ ] It's listed in `appunti/000 - Indice.md`.

(Scratch and project notes are exempt — this applies to lecture notes.)

## What "helping" means here

- Explain concepts clearly, correctly, and at the right level — the user is using this as a learning tool, not just a transcription service.
- When slides are terse or the lecturer's phrasing is ambiguous, add clarifying explanation rather than copying verbatim.
- Flag when something in the slides/subtitles seems inconsistent or when you're inferring beyond what's stated.
- If you didn't understand something from the sources, **say so** rather than filling the gap with something plausible. A note with an honest question mark is worth more than a paragraph that turns out to be wrong in the exam hall.
- The repo is git-tracked now, but there's no commit/push automation: treat file edits like any other content edit, commit only when asked, and still confirm before anything destructive.
