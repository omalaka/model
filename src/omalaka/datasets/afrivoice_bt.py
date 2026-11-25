"""Afrivoice Oshindonga sentences back-translated into English by base MADLAD-400, for English -> Oshindonga.

The Oshindonga side is human (Afrivoice V2 transcripts, data/afrivoice, CC BY
4.0) and only the English side is machine-made: the usual back-translation set
for translating *into* Oshindonga. Never use these pairs for Oshindonga ->
English, where the machine English would be the target; prepare refuses this
kind for that reason.

Always the published base model (Apache-2.0), never a model fine-tuned here:
a fine-tune on closed data would launder that data into this set.

Each row keeps its clip key, `ref` = "<clip>#<n>": the n-th sentence of that
clip's transcript, split the same way as omalaka.net's "translate into English"
task (omalaka-web tools/afrivoice_sentences.py), so it equals the site's
`source_ref`. omalaka.net shows these sentences to speakers; a sentence that
gets a human translation there must not also be trained on as a synthetic
pair, or it could leak into dev and test. Pass the site's export with
--exclude (community/website/ng-afrivoice.tsv, whose note column names the
key) and filter again when training, since more get translated over time.
Matching is by key, not text: speakers can correct a sentence's spelling.

Needs a GPU: book it (scripts/gpu.sh). Resumable; with OMALAKA_GPU_RUN set it
checks between chunks whether to yield the GPU to a reservation, and stops.
Run with --limit for a trial.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

from . import write_source
from ..common import derailed, load_jsonl

MODEL = "google/madlad400-3b-mt"

INFO = {
    "name": "afrivoice-bt",
    "title": "Afrivoice V2 Oshindonga transcripts back-translated into English with MADLAD-400 3B",
    "kind": "backtranslated",
    "langs": ["ng"],
    "homepage": "https://huggingface.co/datasets/DigitalUmuganda/Afrivoice_V2",
    "licence": "CC-BY-4.0",
    "licence_note": "Oshindonga: Afrivoice V2 transcripts by Digital Umuganda, CC BY 4.0 (see data/afrivoice). "
                    "English: output of google/madlad400-3b-mt (Apache-2.0), which places no restriction "
                    "on its output.",
    "checked": "2026-11-25",
    "status": "experimental quality; English -> Oshindonga only",
    "size": "as many as generated (--limit)",
}

# Same split and spacing fixes as omalaka.net, so the keys match its source_ref.
SPLIT = re.compile(r"(?<=[.!?])\s*(?=[A-Z])")
KEY = re.compile(r"\b([A-Za-z0-9]+#\d+)\b")
MIN_WORDS, MAX_WORDS = 3, 40


def sentences(rows):
    """{ref, text, category} for each complete sentence, first copy only."""
    seen = set()
    for r in rows:
        if not r.get("clip"):
            continue
        for n, s in enumerate(SPLIT.split(r["text"].strip()), 1):
            s = re.sub(r"\s+([.,!?;:])", r"\1", " ".join(s.split()))
            s = re.sub(r",(?=\w)", ", ", s)
            if not s or not (MIN_WORDS <= len(s.split()) <= MAX_WORDS) or s[-1] not in ".!?" or not s[0].isupper():
                continue
            if s.lower() in seen:
                continue
            seen.add(s.lower())
            yield {"ref": f"{r['clip']}#{n}", "text": s, "category": r.get("category")}


def excluded(paths):
    """Clip keys named in omalaka.net exports (the note column of ng-afrivoice.tsv)."""
    keys = set()
    for p in paths:
        for line in open(p, encoding="utf-8"):
            keys.update(KEY.findall(line))
    return keys


def still_oshindonga(src: str, hyp: str) -> bool:
    """Half the output's words copied from the input: an untranslated or half-translated line."""
    s = {w.lower() for w in re.findall(r"[^\W\d_]+", src)}
    h = [w.lower() for w in re.findall(r"[^\W\d_]+", hyp)]
    return not h or sum(w in s for w in h) / len(h) > 0.5


def bad(src: str, hyp: str) -> str:
    ratio = len(hyp.split()) / max(1, len(src.split()))
    if still_oshindonga(src, hyp):
        return "copy"
    if derailed(src, hyp):
        return "derailed"
    if not 0.6 <= ratio <= 3.0:  # English needs more words than Oshindonga, not fewer
        return "length"
    return ""


def must_yield() -> bool:
    tool = os.environ.get("OMALAKA_GPU_RUN")
    if not tool:
        return False
    owner = os.environ.get("OMALAKA_GPU_OWNER", "omalaka")
    return subprocess.run([sys.executable, tool, "check", "--owner", owner, "--minutes", "15"]).returncode == 3


def fetch(out: Path, data: Path = None, limit: int = 1000, batch: int = 32, exclude: list = (), seed: int = 13):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from ..common import DEVICE, Translator

    data = data or out.parent
    todo = list(sentences(load_jsonl(data / "afrivoice" / "rows.jsonl")))
    random.Random(seed).shuffle(todo)  # a trial or a partial run covers both topics
    todo = todo[:limit] if limit else todo
    skip = excluded(exclude)
    pairs, rejected = out / "pairs.jsonl", out / "rejected.jsonl"
    have = {r["ref"] for p in (pairs, rejected) if p.exists() for r in load_jsonl(p)}
    todo = [r for r in todo if r["ref"] not in have and r["ref"] not in skip]
    print(f"{len(todo)} to translate ({len(have)} done, {len(skip)} keys excluded)", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    if todo:
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
        translator = Translator(model, tok, MODEL, target="en")
        kept = dropped = 0
        for i in range(0, len(todo), 512):
            chunk = todo[i:i + 512]
            hyps = translator.translate([r["text"] for r in chunk], batch=batch)
            with open(pairs, "a", encoding="utf-8") as ok, open(rejected, "a", encoding="utf-8") as no:
                for r, h in zip(chunk, hyps):
                    why = bad(r["text"], h)
                    row = {"src": r["text"], "tgt": h, "lang": "ng", "origin": INFO["name"], "ref": r["ref"],
                           "category": r["category"], **({"rejected": why} if why else {})}
                    (no if why else ok).write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept, dropped = kept + (not why), dropped + bool(why)
            print(f"{i + len(chunk)}/{len(todo)}: {kept} kept, {dropped} rejected", flush=True)
            if i + 512 < len(todo) and must_yield():
                print("yielding the GPU to a reservation; run again to resume", flush=True)
                break
    write_source(out, {**INFO, "model": MODEL, "excluded_keys": len(skip)}, [])
    return sum(1 for _ in open(pairs, encoding="utf-8")) if pairs.exists() else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="data/afrivoice-bt")
    p.add_argument("--limit", type=int, default=0, help="sentences to translate, 0 for all (resumes)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--exclude", nargs="*", default=[], help="omalaka.net exports whose clip keys to leave out")
    args = p.parse_args(argv)
    print(fetch(Path(args.out), Path(args.data), args.limit, args.batch, args.exclude))


if __name__ == "__main__":
    main()
