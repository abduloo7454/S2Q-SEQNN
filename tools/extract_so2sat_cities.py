#!/usr/bin/env python3
"""
Trim So2Sat LCZ42 to the three cities the S2Q-SEQNN loader actually reads.

Optional: run once to shrink the data before copying it to a cluster.

Why this is safe
----------------
sq_seqnn_fast.load_so2sat() does exactly this:
    training.h5   -> rows where city in {berlin, cologne}
    validation.h5 -> rows where city == munich
    testing.h5    -> rows where city == munich
and from each file it reads only two datasets: "sen2" and "label".
The city mask comes from the matching *_geo.h5 "city" dataset.

Boolean-mask reads return rows in ascending index order. This script writes
those same rows, in that same order, into new files with the same dataset
names. The unchanged loader then applies a city mask that is all-True and
reads back the identical rows in the identical order, so every downstream
random draw (seeded per-class sampling, stratified splits) is bit-identical.

Nothing else in the original files is used by the loader, so nothing else is
kept. Original files are never modified.

Output
------
    <out>/training.h5       sen2 + label, berlin+cologne only
    <out>/training_geo.h5   city
    <out>/validation.h5     sen2 + label, munich only
    <out>/validation_geo.h5 city
    <out>/testing.h5        sen2 + label, munich only
    <out>/testing_geo.h5    city
    <out>/EXTRACT_MANIFEST.txt   row counts + sha256 of every output

Usage
-----
    python extract_so2sat_cities.py \
        --src data/So2Sat_LCZ42_full \
        --out data/So2Sat_LCZ42
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np

SPLITS = {
    "training":   ["berlin", "cologne"],
    "validation": ["munich"],
    "testing":    ["munich"],
}
CHUNK = 2048  # rows per read; keeps peak memory low on the 52 GB file


def _cities(geo_path: Path) -> np.ndarray:
    with h5py.File(geo_path, "r") as gf:
        raw = gf["city"][:]
    return np.array([
        v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
        for v in raw
    ])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def extract_split(src: Path, out: Path, split: str, want: list[str], log: list[str]) -> None:
    h5_in = src / f"{split}.h5"
    geo_in = src / f"{split}_geo.h5"
    h5_out = out / f"{split}.h5"
    geo_out = out / f"{split}_geo.h5"

    cities = _cities(geo_in)
    mask = np.isin(cities, want)
    idx = np.flatnonzero(mask)              # ascending, exactly like f[ds][mask]
    n = len(idx)
    print(f"[{split}] {len(cities)} rows total -> {n} rows for {want}")

    with h5py.File(h5_in, "r") as fin:
        sen2_shape = fin["sen2"].shape
        label_shape = fin["label"].shape
        sen2_dtype = fin["sen2"].dtype
        label_dtype = fin["label"].dtype
        print(f"    sen2  {sen2_shape} {sen2_dtype}")
        print(f"    label {label_shape} {label_dtype}")

        with h5py.File(h5_out, "w") as fout:
            d_sen2 = fout.create_dataset(
                "sen2", shape=(n,) + sen2_shape[1:], dtype=sen2_dtype,
                chunks=(min(256, n),) + sen2_shape[1:], compression="gzip", compression_opts=4,
            )
            d_lab = fout.create_dataset(
                "label", shape=(n,) + label_shape[1:], dtype=label_dtype,
            )
            # Write in contiguous chunks of the *selected* rows. h5py needs
            # increasing indices for fancy reads, which np.flatnonzero gives.
            for start in range(0, n, CHUNK):
                sel = idx[start:start + CHUNK]
                d_sen2[start:start + len(sel)] = fin["sen2"][sel]
                d_lab[start:start + len(sel)] = fin["label"][sel]
                if (start // CHUNK) % 10 == 0:
                    print(f"    {start + len(sel):7d}/{n}", flush=True)

    with h5py.File(geo_out, "w") as gout:
        gout.create_dataset("city", data=np.array(cities[idx], dtype="S"))

    log.append(f"{split}: {n} rows | {h5_out.name} sha256={_sha256(h5_out)} | "
               f"{geo_out.name} sha256={_sha256(geo_out)}")


def verify(src: Path, out: Path) -> None:
    """Re-derive the loader's row selection from both trees and compare."""
    print("\nVerifying that the trimmed files reproduce the loader's rows...")
    for split, want in SPLITS.items():
        c_src = _cities(src / f"{split}_geo.h5")
        c_out = _cities(out / f"{split}_geo.h5")
        m_src = np.isin(c_src, want)
        m_out = np.isin(c_out, want)
        assert m_out.all(), f"{split}: trimmed geo contains non-target cities"
        assert m_src.sum() == len(c_out), f"{split}: row count mismatch"
        with h5py.File(src / f"{split}.h5", "r") as a, h5py.File(out / f"{split}.h5", "r") as b:
            # spot-check first, middle, last selected row for exact equality
            sel = np.flatnonzero(m_src)
            for i_src, i_out in [(sel[0], 0), (sel[len(sel) // 2], len(sel) // 2), (sel[-1], len(sel) - 1)]:
                assert np.array_equal(a["sen2"][i_src], b["sen2"][i_out]), f"{split}: sen2 mismatch at {i_out}"
                assert np.array_equal(a["label"][i_src], b["label"][i_out]), f"{split}: label mismatch at {i_out}"
        print(f"  {split}: OK ({len(c_out)} rows, spot-checks identical)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Folder holding the original So2Sat_LCZ42 files")
    ap.add_argument("--out", required=True, help="Folder to write the trimmed files into")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    src, out = Path(args.src).expanduser(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    for split, want in SPLITS.items():
        extract_split(src, out, split, want, log)
    if not args.no_verify:
        verify(src, out)

    total = sum((out / f).stat().st_size for f in out.iterdir() if f.suffix == ".h5")
    log.append(f"TOTAL trimmed size: {total / 1e9:.2f} GB")
    (out / "EXTRACT_MANIFEST.txt").write_text("\n".join(log) + "\n")
    print("\n" + "\n".join(log))
    print(f"\nDone. Copy {out} to Panther as <S2Q_DATA_ROOT>/So2Sat_LCZ42/")


if __name__ == "__main__":
    main()
