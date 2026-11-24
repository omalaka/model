"""Oshiwambo transcripts from Afrivoice V2 (Digital Umuganda), a speech dataset.

About 438 hours of speech in 76,905 clips. Speakers described an image
(agriculture and finance), and other people transcribed what they said. There
are no translations, so this is monolingual text: input for back-translation
(most useful for English -> Oshiwambo, where the Oshiwambo side stays human)
and for language ID. Only the manifest files are fetched (about 90 MB), not
the 26 GB of audio and images.

The card codes the language `ndo` (Oshindonga) but does not say which dialect
was recorded, so lines go to ng.txt until a speaker has checked a sample;
rows.jsonl keeps locale, location and speaker id for that check. Spoken image
descriptions repeat a lot, so exact duplicates are dropped.

The dataset is gated: accept its terms on the Hugging Face page while logged
in, then run `hf auth login` (or set HF_TOKEN) before fetching.
"""
import datetime
import hashlib
import json
from pathlib import Path

from . import write_source
from ..common import write_jsonl

REPO = "DigitalUmuganda/Afrivoice_V2"
FOLDER = "Oshiwambo"

INFO = {
    "name": "afrivoice",
    "title": "Afrivoice V2, Oshiwambo transcripts (Digital Umuganda)",
    "kind": "mono",
    "langs": ["ng"],
    "homepage": f"https://huggingface.co/datasets/{REPO}",
    "licence": "CC-BY-4.0",
    "licence_note": "Dataset card metadata 'cc-by-4.0'; card text: 'All datasets are licensed under the "
                    "Creative Commons license (CC-BY-4)'. Gated: users agree to share their contact "
                    "details with the publisher. Attribution: Digital Umuganda.",
    "checked": "2026-11-24",
    "status": "verified licence; dialect not yet checked",
    "size": "76,905 clips, 437.8 h (card); manifests about 90 MB",
}


def manifests():
    """Paths of the Oshiwambo manifest files. Listing works without access to the gated files."""
    from huggingface_hub import HfApi
    tree = HfApi().list_repo_tree(REPO, repo_type="dataset", path_in_repo=FOLDER, recursive=True)
    return sorted(f.path for f in tree if f.path.endswith(".jsonl") and "/manifest_" in f.path)


def fetch_manifest(path: str, raw: Path, log: list) -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError
    try:
        local = Path(hf_hub_download(REPO, path, repo_type="dataset", local_dir=raw))
    except GatedRepoError as e:
        raise PermissionError(f"afrivoice: no access to the gated dataset. Accept its terms at {INFO['homepage']} "
                         "while logged in, then run `hf auth login` or set HF_TOKEN.") from e
    log.append({"url": f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}", "file": path,
                "sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
                "retrieved": datetime.date.fromtimestamp(local.stat().st_mtime).isoformat()})
    return local


def read_manifest(local: Path, split: str):
    """One row per clip that has a transcript of at least three words."""
    with open(local, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            text = " ".join(str(r.get("transcription") or "").split())
            if len(text.split()) < 3:
                continue
            yield {"text": text, "split": split, "category": r.get("category"), "locale": r.get("locale"),
                   "location": r.get("location"), "speaker": r.get("voice_creator_id"), "uid": r.get("uid")}


def fetch(out: Path):
    files, rows, seen = [], [], set()
    for path in manifests():
        split = path.split("/")[1]  # e.g. agriculture_oshiwambo_dev_test
        try:
            local = fetch_manifest(path, out / "raw", files)
        except PermissionError as e:  # skip, so a full omalaka-download still fetches the other sets
            print(e, flush=True)
            return 0
        for row in read_manifest(local, split):
            if row["text"] not in seen:
                seen.add(row["text"])
                rows.append(row)
    write_jsonl(out / "rows.jsonl", rows)
    (out / "ng.txt").write_text("".join(r["text"] + "\n" for r in rows), encoding="utf-8")
    write_source(out, INFO, files)
    return len(rows)
