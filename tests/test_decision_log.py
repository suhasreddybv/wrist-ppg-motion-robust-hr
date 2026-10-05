"""Mechanical guard for docs/decisions.md.

Five entries in this log were left marked `open` after their successors landed, and the
inconsistency survived a close-out review. Prose review does not catch this; parsing does.

The log's own rules, enforced here:
  - an entry that points at a successor must point at one that exists;
  - an entry that supersedes or closes another must leave that predecessor pointing back,
    not still marked open or plainly adopted;
  - an entry cannot be both `adopted` and superseded;
  - every entry has an id, a status, and a unique id.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

LOG = Path(__file__).resolve().parents[1] / "docs" / "decisions.md"
ENTRY = re.compile(r"^### (D-\d+)\s*·\s*(.+)$", re.M)
STATUS = re.compile(r"^- \*\*Status:\*\*\s*(.+)$", re.M)
# a reference to another entry from inside a status line
REFERS = re.compile(r"\b(?:superseded by|closed .*?by|implemented .*?as|tested .*?in)\s+(D-\d+)")
# a claim, anywhere in an entry's body, that it supersedes or closes another
SUPERSEDES = re.compile(r"\((?:supersedes|closes|implements)\s+([^)]*?D-\d+[^)]*)\)")


def _entries() -> dict[str, dict]:
    text = LOG.read_text()
    marks = list(ENTRY.finditer(text))
    out = {}
    for i, m in enumerate(marks):
        body = text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        status = STATUS.search(body)
        out[m.group(1)] = dict(title=m.group(2).strip(), body=body,
                               status=status.group(1).strip() if status else None)
    return out


ENTRIES = _entries()


def test_the_log_exists_and_parses():
    assert len(ENTRIES) >= 10, "decision log did not parse into entries"


def test_every_entry_has_a_status():
    missing = [k for k, v in ENTRIES.items() if not v["status"]]
    assert not missing, f"entries without a Status line: {missing}"


def test_entry_ids_are_unique_and_sequential():
    text = LOG.read_text()
    ids = ENTRY.findall(text)
    seen = [i for i, _ in ids]
    assert len(seen) == len(set(seen)), f"duplicate entry ids: {[i for i in seen if seen.count(i) > 1]}"
    numbers = sorted(int(i.split("-")[1]) for i in seen)
    assert numbers == list(range(numbers[0], numbers[0] + len(numbers))), f"gap in entry ids: {numbers}"


def test_status_references_point_at_entries_that_exist():
    bad = []
    for key, v in ENTRIES.items():
        for ref in REFERS.findall(v["status"] or ""):
            if ref not in ENTRIES:
                bad.append((key, ref))
    assert not bad, f"status lines referring to entries that do not exist: {bad}"


def test_a_superseding_entry_leaves_its_predecessor_updated():
    """The failure this guard was written for: successors landing without updating predecessors."""
    stale = []
    for key, v in ENTRIES.items():
        for claim in SUPERSEDES.findall(v["body"]):
            for pred in re.findall(r"D-\d+", claim):
                if pred not in ENTRIES:
                    stale.append((key, pred, "predecessor does not exist"))
                    continue
                pred_status = (ENTRIES[pred]["status"] or "").lower()
                points_back = key.lower() in pred_status
                resolved = any(w in pred_status for w in
                               ("superseded", "closed", "implemented", "tested", "withdrawn"))
                if not (points_back or resolved):
                    stale.append((key, pred, ENTRIES[pred]["status"]))
    assert not stale, (
        "entries claim to supersede/close/implement a predecessor whose Status was never "
        f"updated: {stale}")


def test_no_entry_is_both_adopted_and_superseded():
    bad = [k for k, v in ENTRIES.items()
           if (v["status"] or "").lower().startswith("adopted") and "superseded by" in (v["status"] or "").lower()]
    assert not bad, f"entries marked both adopted and superseded: {bad}"


def test_open_entries_are_genuinely_open():
    """An entry may be open, but then nothing may claim to have resolved it."""
    text = LOG.read_text()
    wrongly_open = []
    for key, v in ENTRIES.items():
        if not (v["status"] or "").lower().startswith("open"):
            continue
        for other, ov in ENTRIES.items():
            if other == key:
                continue
            if re.search(rf"\((?:supersedes|closes|implements)[^)]*{key}\b", ov["body"]):
                wrongly_open.append((key, other))
    assert not wrongly_open, f"entries marked open that another entry claims to resolve: {wrongly_open}"


# An entry states its substance under one of these headings. Findings, corrections,
# registered predictions and deferrals do not all have a "Decision"; they all have something.
SUBSTANCE = ("Decision:", "Result", "Evidence:", "Note", "Correction", "Prediction", "What happened:")


def test_every_entry_says_something_substantive():
    """An entry must carry more than a title and a status."""
    empty = [k for k, v in ENTRIES.items()
             if not any(f"**{h}" in v["body"] for h in SUBSTANCE)]
    assert not empty, f"entries with a status but no substance: {empty}"


def test_adopted_entries_point_at_something_checkable():
    """An adopted decision must cite a committed artifact or a measured quantity.

    Heading names vary across entries - Evidence, Result, Criterion, Verification, How it
    happened - so this checks the substance rather than the label: either a path into
    results/, tests/, docs/ or figures/, or a number that came from somewhere.
    """
    artifact = re.compile(r"`[^`]*(?:results|tests|figures|docs)/[^`]+`|\b\d+\.\d+\b|\b\d{2,}\b")
    bare = [k for k, v in ENTRIES.items()
            if (v["status"] or "").lower().startswith("adopted") and not artifact.search(v["body"])]
    assert not bare, (
        "adopted entries citing neither a committed artifact nor a measured number: "
        f"{bare}")
