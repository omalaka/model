"""Oshiwambo translations of open English sentences, made on omalaka.net.

Speakers on omalaka.net translate short everyday English sentences from
Tatoeba (CC BY 2.0 FR) into Oshikwanyama or Oshindonga, and a second speaker
checks each one. The Oshiwambo side is the contributor's own work (CC BY 4.0);
the English side is the Tatoeba sentence, credited in each row's `note`.

Translated from English, so it reads like a translation: training data only,
never dev or test (kind "translated"). Files: community/website/*-from-english.tsv.
"""
import csv
from pathlib import Path

from . import write_source
from ..common import write_jsonl

ROOT = Path(__file__).resolve().parents[3] / "community" / "website"

INFO = {
    "name": "community_from_english",
    "title": "omalaka.net translations from English (Tatoeba prompts)",
    "kind": "translated",
    "langs": ["kj", "ng"],
    "homepage": "https://github.com/omalaka/model/tree/main/community/website",
    "licence": "CC-BY-4.0 (Oshiwambo) + CC-BY-2.0-FR (English, Tatoeba)",
    "licence_note": "Oshiwambo by omalaka.net contributors under CC BY 4.0 (named in `translator`); "
                    "English from Tatoeba (tatoeba.org), CC BY 2.0 FR, sentence number and author in `note`.",
    "checked": "2026-11-08",
    "status": "verified",
    "size": "grows with contributions",
}


def read(root: Path = ROOT):
    rows = []
    for path in sorted(root.glob("*-from-english.tsv")):
        with open(path, encoding="utf-8", newline="") as fh:
            for rec in csv.DictReader(fh, delimiter="\t"):
                if rec.get("oshiwambo", "").strip() and rec.get("english", "").strip():
                    rows.append({"src": rec["oshiwambo"].strip(), "tgt": rec["english"].strip(),
                                 "lang": rec["lang"].strip(), "origin": INFO["name"],
                                 "split": "train",
                                 "reviewed": bool(rec.get("reviewed_by", "").strip())})
    return rows


def fetch(out: Path):
    rows = read()
    write_jsonl(out / "pairs.jsonl", rows)
    write_source(out, INFO, [])
    return len(rows)
