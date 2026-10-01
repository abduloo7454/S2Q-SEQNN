#!/usr/bin/env python3
"""
Collect the TQE-revision runs into results/tqe/ WITHOUT touching
results/all_runs.csv. Reuses the collector of the main campaign.

    python tools/collect_tqe.py manifest_tqe.csv                       # -> results/tqe/
    python tools/collect_tqe.py manifest_teacher_pilot.csv teacher_pilot  # -> results/teacher_pilot/

Teacher runs also carry their teacher_* fields (label counts, linear read-out references).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_results as C  # noqa: E402

man = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "manifest_tqe.csv")
found = C.gather_main_model(man)
if not found.empty:                                   # teacher runs: add their teacher_* fields
    extra = []
    for f in found.source_file:
        r = pd.read_csv(C.BUNDLE / f).iloc[0]
        extra.append({k: r[k] for k in r.index if k.startswith("teacher")})
    found = pd.concat([found, pd.DataFrame(extra, index=found.index)], axis=1)
out = C.RES / (sys.argv[2] if len(sys.argv) > 2 else "tqe")
out.mkdir(parents=True, exist_ok=True)
if not found.empty:
    for c in ("test_acc", "bal_test_acc", "val_acc", "train_acc"):
        found[c] = C._pct(found[c])
    found = found.sort_values(["campaign", "config", "seed"]).reset_index(drop=True)
    found.to_csv(out / "all_runs_tqe.csv", index=False)
exp = man.groupby("campaign").size().rename("expected")
got = found.groupby("campaign").size().rename("found") if not found.empty else pd.Series(dtype=int, name="found")
comp = pd.concat([exp, got], axis=1).fillna(0).astype(int)
comp["missing"] = comp["expected"] - comp["found"]
comp.to_csv(out / "completeness_tqe.csv")
done = set(found.task_id.astype(int)) if not found.empty else set()
missing = sorted(set(man.task_id.astype(int)) - done)
(out / "missing_task_ids_tqe.txt").write_text("\n".join(map(str, missing)) + ("\n" if missing else ""))
print(comp.to_string())
print(f"{len(found)} / {len(man)} finished; missing ids -> {out}/missing_task_ids_tqe.txt")
