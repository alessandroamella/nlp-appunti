#!/usr/bin/env python3
"""Download every [Video] linked on the Brightspace course page and subtitle it
with AssemblyAI universal-3-5-pro, using the lecture's slides as context.

Pipeline, per video (several videos in parallel):
  1. download   Panopto links via panopto-dl, Brightspace .mp4 links directly
  2. context    slides PDF -> pdftotext -> OpenAI -> short prompt + keyterms
  3. audio      ffmpeg -> mono 16 kHz opus (a few tens of MB instead of GB)
  4. subtitles  AssemblyAI -> subs/<name>.srt + subs/<name>.json sidecar

Every step is skipped when its output already exists, so re-running only does
what's missing. Delete a file to redo that step (e.g. a .srt to re-transcribe).

Setup (all gitignored):
  .env                     ASSEMBLYAI_API_KEY=..., OPENAI_API_KEY=...
  cookies/brightspace.json cookie export from a logged-in brightspace.au.dk tab
  cookies/panopto.json     same for au.cloud.panopto.eu (optional: without it
                           panopto-dl reads the cookie from Firefox)
Cookie files are JSON lists of {"name": ..., "value": ...}, as exported by
Cookie-Editor & co. Brightspace session cookies expire: when the script says
so, export them again.
"""

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import requests

HERE = Path(__file__).resolve().parent
VIDEOS = HERE / "videos"
AUDIO = HERE / "audio"
SUBS = HERE / "subs"
PROMPTS = HERE / "prompts"
SLIDES = HERE / "slides"
LOGS = HERE / "logs"
INDEX = HERE / "index.json"
DOCS = HERE.parent / "docs"  # slides already in the repo, reused before downloading

BRIGHTSPACE = "https://brightspace.au.dk"
COURSE_ID = 219812
UNIT_ID = 2863627  # "Natural Language Processing (NLP)" unit, the one with the schedule table
COURSE_NAME = "Natural Language Processing (NLP) at Aarhus University"

ASR_MODEL = "universal-3-5-pro"
ASR_PRICE_PER_HOUR = 0.21
LLM_MODEL = "gpt-6-astra"
LLM_PRICE_PER_M = {"input": 10.0, "output": 50.0}  # $ per 1M tokens, Standard tier
KEYTERMS_MAX_WORDS = 900  # API cap is 1000 words, minus some headroom for tokenization

print_lock = threading.Lock()


def log(*args):
    with print_lock:
        print(*args, flush=True)


def load_env():
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*?)\s*$", line)
            if m and not os.environ.get(m[1]):
                os.environ[m[1]] = m[2].strip("'\"")


def cookie_header(name):
    path = HERE / "cookies" / f"{name}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if isinstance(data, dict):  # some exporters wrap the list
        data = data.get("cookies", [])
    return "; ".join(f"{c['name']}={c['value']}" for c in data)


# ---------------------------------------------------------------- course page


def fetch_schedule(session):
    """Return the [Video] links of the course page, each with its slides and title."""
    r = session.get(
        f"{BRIGHTSPACE}/d2l/api/le/1.75/{COURSE_ID}/content/toc", allow_redirects=False
    )
    if r.status_code != 200 or "json" not in r.headers.get("content-type", ""):
        sys.exit(
            f"Brightspace answered {r.status_code}: the cookies in cookies/brightspace.json "
            "have probably expired. Export them again from a logged-in tab."
        )
    module = find_module(r.json()["Modules"], UNIT_ID)
    if module is None:
        sys.exit(f"unit {UNIT_ID} not found in the course TOC")
    base_path = module["DefaultPath"]  # /content/enforced/219812-LR60743/
    page = module["Description"]["Html"]

    def file_url(href):
        # quickLink.d2l?...&type=coursefile&fileId=lab_videos%2flab_session1.mp4
        m = re.search(r"fileId=([^&\"]+)", html.unescape(href))
        if not m:
            return None
        return BRIGHTSPACE + base_path + unquote(m[1].replace("+", " "))

    videos = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        date = text_of(cells[1]) if len(cells) > 1 else ""
        for cell in cells:
            links = re.findall(r'<a [^>]*href="([^"]+)"[^>]*>(.*?)</a>', cell, re.S)
            for href, label in links:
                if not text_of(label).startswith("[Video"):
                    continue
                href = html.unescape(href)
                slides = next(
                    (file_url(h) for h, l in links if text_of(l).startswith("[Slides")), None
                )
                title = text_of(re.split(r"<a ", cell, maxsplit=1)[0])
                if "panopto" in href:
                    kind, url = "panopto", href
                else:
                    kind, url = "brightspace", file_url(href)
                videos.append(
                    {"kind": kind, "url": url, "slides": slides, "title": title, "date": date}
                )
    return videos


def find_module(modules, unit_id):
    for m in modules:
        if m["ModuleId"] == unit_id:
            return m
        found = find_module(m.get("Modules", []), unit_id)
        if found:
            return found
    return None


def text_of(fragment):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


# ---------------------------------------------------------------- naming


def resolve_name(video, session, index):
    """File base name, cached in index.json so later runs don't have to ask again."""
    if video["url"] in index:
        return index[video["url"]]["name"]
    if video["kind"] == "panopto":
        # Same name panopto-dl will save the file under.
        out = run(["panopto-dl", "--list", video["url"]], capture=True)
        names = [
            m[1] for m in (re.match(r"(.+) \(\d+:\d\d:\d\d\)$", l) for l in out.splitlines()) if m
        ]
        if not names:
            raise RuntimeError(f"panopto-dl --list gave no recording:\n{out}")
        name = names[0]
    else:
        # Raw uploads have no session name: build one like Panopto's, from the
        # recording time in the file's metadata (ffprobe only reads the header).
        out = run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
                "-of", "default=nw=1:nk=1", "-headers", f"Cookie: {session.headers['Cookie']}\r\n",
                video["url"],
            ],
            capture=True,
        ).strip()
        when = datetime.fromisoformat(out.replace("Z", "+00:00")).astimezone()
        prefix = "NLP Fall26 Lab " if "lab" in video["url"].lower() else "NLP Fall26"
        name = prefix + when.strftime("%Y-%m-%d %H-%M")
    return name


# ---------------------------------------------------------------- steps


def download(video, name, session):
    dest = VIDEOS / f"{name}.mp4"
    if dest.exists():
        return "video already there"
    if video["kind"] == "panopto":
        # panopto-dl skips what exists and also writes Panopto's own .srt and
        # the .json sidecar (with the viewer URL) next to the video.
        logfile = LOGS / f"{name}.download.log"
        with open(logfile, "w") as f:
            code = subprocess.call(
                ["panopto-dl", "--flat", "-c", "1", "-o", str(VIDEOS), video["url"]],
                stdout=f, stderr=subprocess.STDOUT, cwd=HERE,
            )
        if code != 0 or not dest.exists():
            raise RuntimeError(f"panopto-dl failed, see {logfile}")
    else:
        fetch_file(session, video["url"], dest)
    return f"downloaded ({dest.stat().st_size / 1e9:.1f} GB)"


def fetch_file(session, url, dest):
    """Plain HTTP download that resumes from a .part file."""
    part = dest.with_name(dest.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with session.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:  # .part already complete
            pass
        elif r.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {r.status_code} for {url} (cookies expired?)")
        else:
            mode = "ab" if r.status_code == 206 else "wb"
            with open(part, mode) as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    part.rename(dest)


def slides_text(video, session):
    if not video["slides"]:
        return None
    filename = video["slides"].rsplit("/", 1)[1]
    pdf = DOCS / filename
    if not pdf.exists():
        pdf = SLIDES / filename
        if not pdf.exists():
            fetch_file(session, video["slides"], pdf)
    return run(["pdftotext", "-q", str(pdf), "-"], capture=True)


def asr_context(video, name, session):
    """Prompt + keyterms for AssemblyAI, written by an LLM from the slides."""
    cache = PROMPTS / f"{name}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    slides = slides_text(video, session)
    lecture = f"{video['title']} ({video['date']})"
    if not slides or not os.environ.get("OPENAI_API_KEY"):
        ctx = {
            "prompt": f"Recorded university lecture of the course {COURSE_NAME}: {lecture}.",
            "keyterms": [],
        }
    else:
        ctx = ask_llm(lecture, slides[:150_000])
    ctx["keyterms"] = cap_keyterms(ctx["keyterms"])
    ctx.update(lecture=lecture, slides=video["slides"], model=LLM_MODEL if slides else None)
    cache.write_text(json.dumps(ctx, indent=2, ensure_ascii=False) + "\n")
    return ctx


LLM_INSTRUCTIONS = f"""\
You prepare context for a speech-to-text model that will transcribe the recording \
of a university lecture. You get the lecture title and the text extracted from its \
slide deck. The course is {COURSE_NAME}; the lecture is in English.

Return two things:

1. "prompt": a description of the recording in 20 to 50 words, written as plain \
complete sentences in one short block. Say it is a recorded university lecture, \
name the course and the lecture topic, and mention the main subjects covered. \
Describe the audio; do not give transcription instructions and do not pack it with \
keyword lists.

2. "keyterms": the words and phrases from the slides that the lecturer is likely to \
say aloud and that a speech recognizer could get wrong or misspell: technical terms, \
model and architecture names (e.g. "BERT", "word2vec", "seq2seq"), acronyms, method \
names, datasets, benchmarks, libraries, paper authors and other people's names, and \
distinctive mathematical vocabulary. Write each as it should appear in a transcript, \
with the right capitalization and hyphenation. At most 6 words per phrase, at most \
250 entries, most important first. Leave out everyday English words, slide \
boilerplate (page numbers, URLs, "Outline", "Questions?") and duplicates."""


def ask_llm(lecture, slides):
    from openai import OpenAI

    schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "keyterms": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["prompt", "keyterms"],
        "additionalProperties": False,
    }
    response = OpenAI().responses.create(
        model=LLM_MODEL,
        reasoning={"effort": "low"},
        instructions=LLM_INSTRUCTIONS,
        input=f"Lecture: {lecture}\n\nSlide text:\n\n{slides}",
        text={"format": {"type": "json_schema", "name": "asr_context", "strict": True, "schema": schema}},
    )
    ctx = json.loads(response.output_text)
    usage = response.usage
    ctx["cost_usd"] = round(
        (usage.input_tokens * LLM_PRICE_PER_M["input"] + usage.output_tokens * LLM_PRICE_PER_M["output"]) / 1e6, 4
    )
    log(f"[context] {lecture}: {len(ctx['keyterms'])} keyterms, ${ctx['cost_usd']:.3f}")
    return ctx


def cap_keyterms(terms):
    out, seen, words = [], set(), 0
    for t in terms:
        t = " ".join(t.split())
        n = len(t.split())
        if not t or n > 6 or t.lower() in seen:
            continue
        if words + n > KEYTERMS_MAX_WORDS:
            break
        out.append(t)
        seen.add(t.lower())
        words += n
    return out


def extract_audio(name):
    dest = AUDIO / f"{name}.opus"
    if dest.exists():
        return dest
    tmp = dest.with_name(dest.stem + ".tmp.opus")
    run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(VIDEOS / f"{name}.mp4"),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "32k", str(tmp),
        ]
    )
    tmp.rename(dest)
    return dest


def transcribe(video, name, session):
    srt = SUBS / f"{name}.srt"
    if srt.exists():
        return "subtitles already there"

    from assemblyai import TranscriptStatus
    from assemblyai.prerecorded import v2 as aai

    ctx = asr_context(video, name, session)
    audio = extract_audio(name)
    config = {"speech_models": [ASR_MODEL], "language_code": "en", "prompt": ctx["prompt"]}
    if ctx["keyterms"]:
        config["keyterms_prompt"] = ctx["keyterms"]
    transcriber = aai.Transcriber(
        api_key=os.environ["ASSEMBLYAI_API_KEY"], config=aai.TranscriptionConfig(**config)
    )
    transcript = transcriber.transcribe(str(audio))
    if transcript.status == TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI: {transcript.error}")

    write_sidecar(video, name, transcript, ctx)
    tmp = srt.with_name(srt.name + ".tmp")
    tmp.write_text(transcript.export_subtitles_srt())
    tmp.rename(srt)  # last, so a crash never leaves a .srt that looks finished
    hours = (transcript.audio_duration or 0) / 3600
    return f"subtitled ({hours:.1f} h, ~${hours * ASR_PRICE_PER_HOUR:.2f})"


def write_sidecar(video, name, transcript, ctx):
    """Same shape as panopto-dl's sidecar: viewerUrl + &start=<s> is the timestamp link."""
    panopto_meta = VIDEOS / f"{name}.json"
    if panopto_meta.exists():
        meta = json.loads(panopto_meta.read_text())
    else:
        meta = {
            "sessionId": None,
            "title": name,
            "folder": None,
            "durationSeconds": transcript.audio_duration,
            "viewerUrl": video["url"],
            "deepLink": video["url"] + "#t=<seconds-from-start-of-recording>",
        }
    meta.update(
        captions=f"{name}.srt",
        lecture=ctx["lecture"],
        slides=video["slides"],
        source=f"captions by AssemblyAI {ASR_MODEL}, transcript id {transcript.id}, "
        f"prompted with prompts/{name}.json",
    )
    (SUBS / f"{name}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- plumbing


def run(cmd, capture=False):
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {(p.stderr or p.stdout).strip()[-800:]}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--downloads", type=int, default=3, help="parallel downloads (default 3)")
    ap.add_argument("-t", "--transcriptions", type=int, default=4, help="parallel transcriptions (default 4)")
    ap.add_argument("-l", "--list", action="store_true", help="only show what's on the page and what's missing")
    ap.add_argument("--no-subs", action="store_true", help="only download the videos")
    args = ap.parse_args()

    load_env()
    for d in (VIDEOS, AUDIO, SUBS, PROMPTS, SLIDES, LOGS):
        d.mkdir(exist_ok=True)
    if not args.no_subs and not args.list and not os.environ.get("ASSEMBLYAI_API_KEY"):
        sys.exit("ASSEMBLYAI_API_KEY missing (put it in vids/.env)")
    for tool in ("panopto-dl", "ffmpeg", "ffprobe", "pdftotext"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH")

    brightspace = cookie_header("brightspace")
    if not brightspace:
        sys.exit("cookies/brightspace.json missing")
    session = requests.Session()
    session.headers["Cookie"] = brightspace
    panopto = cookie_header("panopto")
    if panopto:
        os.environ["PANOPTO_COOKIE"] = panopto  # otherwise panopto-dl falls back to Firefox

    index = json.loads(INDEX.read_text()) if INDEX.exists() else {}
    videos = fetch_schedule(session)
    for v in videos:
        v["name"] = resolve_name(v, session, index)
        index[v["url"]] = {k: v[k] for k in ("name", "date", "title", "kind", "slides")}
    INDEX.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")

    for v in videos:
        have_video = (VIDEOS / f"{v['name']}.mp4").exists()
        have_subs = (SUBS / f"{v['name']}.srt").exists()
        log(f"{'V' if have_video else '-'}{'S' if have_subs else '-'}  {v['date']:<12} {v['name']}  ({v['title']})")
    if args.list:
        return

    failures = []
    started = time.time()
    with ThreadPoolExecutor(args.downloads) as dl, ThreadPoolExecutor(args.transcriptions) as tr:
        pending = {}
        for v in videos:
            # A finished .srt needs nothing else, not even the video.
            if not args.no_subs and (SUBS / f"{v['name']}.srt").exists():
                continue
            pending[dl.submit(download, v, v["name"], session)] = ("download", v)
        while pending:
            for fut in as_completed(list(pending)):
                step, v = pending.pop(fut)
                try:
                    log(f"[{step}] {v['name']}: {fut.result()}")
                    if step == "download" and not args.no_subs:
                        pending[tr.submit(transcribe, v, v["name"], session)] = ("subtitles", v)
                except Exception as e:  # keep going with the other videos
                    failures.append(v["name"])
                    log(f"[{step}] {v['name']}: FAILED - {e}")
                break  # re-enter as_completed so newly queued transcriptions are watched

    log(f"done in {(time.time() - started) / 60:.0f} min" + (f", {len(failures)} failed: {failures}" if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
