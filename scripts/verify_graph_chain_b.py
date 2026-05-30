#!/usr/bin/env python3
"""Chain B: compare on-disk PNG SHA256 to results/graph_manifest.json (local or CI)."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=pathlib.Path,
        default=None,
        help="Project root (default: parent of scripts/)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any check fails (default: warn only)",
    )
    args = p.parse_args()
    root = args.root
    if root is None:
        root = pathlib.Path(__file__).resolve().parents[1]
    root = root.resolve()
    manifest_path = root / "results" / "graph_manifest.json"
    if not manifest_path.is_file():
        msg = f"No manifest at {manifest_path} — run training or sync first."
        print(msg)
        return 1 if args.strict else 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arts = manifest.get("artifacts") or {}
    targets = (
        "graphs/pinn_loss_history.png",
        "graphs/pinn_ac225_pred_vs_true.png",
    )
    ok = True
    for rel in targets:
        entry = arts.get(rel)
        disk = root / rel
        if not disk.is_file():
            print(f"[missing file] {rel}")
            ok = False
            continue
        if not isinstance(entry, dict) or not entry.get("sha256"):
            print(f"[missing manifest entry] {rel}")
            ok = False
            continue
        got = _sha256(disk)
        exp = entry["sha256"]
        prod = entry.get("producer", "?")
        if got == exp:
            print(f"[OK] {rel} sha256 matches manifest (producer={prod})")
        else:
            print(f"[MISMATCH] {rel}\n  disk:    {got}\n  manifest:{exp}\n  producer:{prod}")
            ok = False

    run_path = root / "results" / "last_training_run.json"
    if run_path.is_file():
        run_data = json.loads(run_path.read_text(encoding="utf-8"))
        print(
            f"last_training_run: status={run_data.get('status')} run_id={run_data.get('run_id')}"
        )
    else:
        print("last_training_run.json not found")

    sync_log = root / "results" / "sync_log.txt"
    if sync_log.is_file():
        tail = sync_log.read_text(encoding="utf-8").strip().splitlines()[-3:]
        if tail:
            print("sync_log (last lines):")
            for line in tail:
                print(f"  {line}")

    if not ok:
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
