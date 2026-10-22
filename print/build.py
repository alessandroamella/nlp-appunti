"""Render the notes in appunti/ to printable PDFs.

    python print/build.py

Writes print/pdf/<note>.pdf for every note, plus print/pdf/NLP - all notes.pdf
(index first, then the lecture notes in date order). Obsidian syntax (image
embeds, [[wikilinks]], > [!callouts]) is rewritten into plain Markdown, pandoc
turns it into HTML with MathML, and headless Chromium prints it.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTES = ROOT / "appunti"
OUT = ROOT / "print" / "pdf"
CSS = ROOT / "print" / "print.css"
# CI runners ship google-chrome, not chromium-browser
CHROME = os.environ.get("CHROME", "chromium-browser")

EMBED = re.compile(r"!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")
WIKILINK = re.compile(r"\[\[([^\]|#]+)(#[^\]|]*)?(?:\|([^\]]*))?\]\]")
CALLOUT = re.compile(r"^> \[!(\w+)\][+-]? ?(.*)$")


def slug(name):
    return "note-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def convert_inline(text, link_targets):
    """Rewrite embeds and wikilinks, leaving `code spans` untouched."""
    parts = re.split(r"(`[^`]*`)", text)
    for i, part in enumerate(parts):
        if i % 2:
            continue
        part = EMBED.sub(lambda m: f"![]({ROOT / m.group(1)})", part)

        def link(m):
            target, heading, alias = m.group(1).strip(), m.group(2), m.group(3)
            label = alias or (target + (f" › {heading[1:]}" if heading else ""))
            if target in link_targets:
                return f"[{label}](#{slug(target)})"
            return f"*{label}*"

        parts[i] = WIKILINK.sub(link, part)
    return "".join(parts)


def convert(md, link_targets):
    out, in_fence, callout = [], False, False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if in_fence or line.lstrip().startswith("```"):
            out.append(line)
            continue
        m = CALLOUT.match(line)
        if m:
            kind, title = m.group(1).lower(), m.group(2) or m.group(1).capitalize()
            out += ["", f"::: {{.callout .{kind}}}", f"[{convert_inline(title, link_targets)}]{{.callout-title}}", ""]
            callout = True
            continue
        if callout:
            if line.startswith(">"):
                out.append(convert_inline(line[1:].removeprefix(" "), link_targets))
                continue
            out += [":::", ""]
            callout = False
        out.append(convert_inline(line, link_targets))
    if callout:
        out.append(":::")
    return "\n".join(out) + "\n"


def note_markdown(path, link_targets):
    name = path.stem
    md = path.read_text()
    # every note but the index starts without a title: add one
    if not md.lstrip().startswith("# "):
        md = f"# {name}\n\n{md}"
    md = re.sub(r"\A\s*# (.*)", lambda m: f"# {m.group(1)} {{#{slug(name)}}}", md, count=1)
    return convert(md, link_targets)


def render(markdown, title, pdf):
    with tempfile.TemporaryDirectory() as tmp:
        src, html = Path(tmp) / "in.md", Path(tmp) / "out.html"
        src.write_text(markdown)
        subprocess.run(
            ["pandoc", str(src), "-f", "markdown+tex_math_dollars+lists_without_preceding_blankline-implicit_figures",
             "-t", "html5", "--standalone", "--mathml", "--embed-resources",
             "--css", str(CSS), "--metadata", f"pagetitle={title}", "-o", str(html)],
            check=True)
        subprocess.run(
            [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={pdf}", html.as_uri()],
            check=True, capture_output=True)
    print("wrote", pdf.relative_to(ROOT))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    notes = sorted(NOTES.glob("*.md"))
    targets = {p.stem for p in notes}

    # one PDF per note: links to other notes become plain italic text
    for p in notes:
        render(note_markdown(p, set()), p.stem, OUT / f"{p.stem}.pdf")

    # everything in one PDF, with working cross-links
    combined = "\n\n".join(
        f'<div class="note">\n\n{note_markdown(p, targets)}\n\n</div>' for p in notes)
    render(combined, "NLP — all notes", OUT / "NLP - all notes.pdf")

    # drop PDFs left over from notes that were renamed or deleted
    keep = {f"{p.stem}.pdf" for p in notes} | {"NLP - all notes.pdf"}
    for pdf in OUT.glob("*.pdf"):
        if pdf.name not in keep:
            pdf.unlink()
            print("removed", pdf.relative_to(ROOT))


if __name__ == "__main__":
    sys.exit(main())
