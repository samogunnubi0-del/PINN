"""
Empirical data sync for PINN training — cross-section bands, flux jitter, impurity anchors.

Usage:
  python scripts/empirical_sync.py init          # create data/empirical/ templates
  python scripts/empirical_sync.py validate      # check manifest + CSV schemas
  python scripts/empirical_sync.py manifest      # write data/empirical_manifest.json

Env (train.py hooks):
  PINN_EMPIRICAL_CSV=path       optional flux log CSV (time_h, phi, optional energy_ev)
  PINN_SIGMA_UNCERTAINTY=0.10   fractional ± band for n2n/ngamma (default 0.10)
  PINN_FLUX_JITTER_SIGMA=0.15   log-normal sigma for beam current jitter
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMPIRICAL = ROOT / "data" / "empirical"
MANIFEST = ROOT / "data" / "empirical_manifest.json"

CROSS_SECTION_TEMPLATE = """# sigma_name,value_cm2,uncertainty_frac,source
sigma_n2n_fast,2.7e-26,0.10,JENDL-5 spectrum avg
sigma_ngamma_thermal,1.28e-23,0.05,ENDF/B-VIII thermal
"""

FLUX_LOG_TEMPLATE = """time_h,phi_n_cm2_s,energy_ev,notes
0.0,1.0e14,1.4e7,fast reactor nominal
100.0,9.5e13,1.4e7,beam dip -5%
250.0,1.05e14,1.4e7,beam boost +5%
"""

IMPURITY_ANCHOR_TEMPLATE = """scenario,ac227_ac225_activity_ratio_max,source
fast14_virgin,0.0015,FDA clinical limit 0.15%
thermal_virgin,0.0001,negligible n2n at thermal
"""


def init_templates() -> None:
    EMPIRICAL.mkdir(parents=True, exist_ok=True)
    files = {
        "cross_section_bands.csv": CROSS_SECTION_TEMPLATE,
        "flux_log.csv": FLUX_LOG_TEMPLATE,
        "impurity_anchors.csv": IMPURITY_ANCHOR_TEMPLATE,
    }
    for name, content in files.items():
        path = EMPIRICAL / name
        if not path.exists():
            path.write_text(content.strip() + "\n", encoding="utf-8")
            print(f"[empirical_sync] created {path.relative_to(ROOT)}")
        else:
            print(f"[empirical_sync] exists {path.relative_to(ROOT)}")


def _read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate() -> bool:
    ok = True
    required = {
        "cross_section_bands.csv": {"sigma_name", "value_cm2", "uncertainty_frac"},
        "flux_log.csv": {"time_h", "phi_n_cm2_s"},
        "impurity_anchors.csv": {"scenario", "ac227_ac225_activity_ratio_max"},
    }
    for fname, cols in required.items():
        path = EMPIRICAL / fname
        if not path.is_file():
            print(f"[empirical_sync] MISSING {path.relative_to(ROOT)}")
            ok = False
            continue
        rows = _read_csv(path)
        if not rows:
            print(f"[empirical_sync] EMPTY {fname}")
            ok = False
            continue
        missing = cols - set(rows[0].keys())
        if missing:
            print(f"[empirical_sync] BAD columns in {fname}: missing {missing}")
            ok = False
        else:
            print(f"[empirical_sync] OK {fname} ({len(rows)} rows)")
    return ok


def write_manifest() -> None:
    EMPIRICAL.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in sorted(EMPIRICAL.glob("*.csv")):
        entries.append({
            "file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "rows": len(_read_csv(path)) if path.stat().st_size else 0,
        })
    manifest = {
        "version": 1,
        "description": "Empirical anchors for PINN physics/data loop (EXFOR-style bands, flux logs, impurity limits)",
        "files": entries,
        "train_env": {
            "PINN_EMPIRICAL_CSV": "data/empirical/flux_log.csv",
            "PINN_SIGMA_UNCERTAINTY": "0.10",
            "PINN_FLUX_JITTER_SIGMA": "0.15",
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[empirical_sync] wrote {MANIFEST.relative_to(ROOT)}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "init"
    if cmd == "init":
        init_templates()
        write_manifest()
    elif cmd == "validate":
        if not validate():
            sys.exit(1)
        write_manifest()
    elif cmd == "manifest":
        write_manifest()
    else:
        print("Usage: python scripts/empirical_sync.py [init|validate|manifest]")
        sys.exit(1)


if __name__ == "__main__":
    main()
