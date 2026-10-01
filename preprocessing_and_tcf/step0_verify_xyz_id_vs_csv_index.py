#!/usr/bin/env python3
"""
Step 0 - Verify the Xe identity scheme, over the FULL dataset.

Two independent checks, run across every chunk, not a hand-picked sample:

  CHECK A (direct):  for every frame, match each CSV row to the XYZ Xe row
                      occupying the same position in that frame (they should
                      coincide almost exactly, the ML model predicts on that
                      exact snapshot's geometry), then compare CSV atom_index
                      to that XYZ row's id column. Reports the fraction of
                      rows where they agree.

  CHECK B (continuity): independent of the CSV entirely, track Xe atoms in
                      the XYZ files purely by their id column and ask two
                      things: (1) does the same set of Xe ids recur in every
                      single frame, across every chunk boundary, and (2) does
                      each id move by only a small, physically reasonable
                      amount between consecutive 100 fs frames. This tests
                      whether the XYZ id is genuinely a persistent per-atom
                      tag, independent of whatever atom_index turns out to be.

Output: a printed verdict plus the numbers behind it, and (if requested)
a .npz dump of the raw per-frame XYZ Xe id/position arrays so Step 1 can be
rewritten around whichever identity scheme this confirms, without re-parsing
the XYZ files a second time.

Usage
-----
    python step0_verify_xyz_id_vs_csv_index.py \\
        --data-dir /path/to/cnt_10_0_u23_Xe4_10ns \\
        --dt-fs 100.0 \\
        --save-xyz-xe-arrays ./xyz_xe_raw.npz
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_lattice(xyz_path: Path) -> np.ndarray:
    with open(xyz_path, "r") as f:
        f.readline()
        header = f.readline()
    m = re.search(r'Lattice="([^"]+)"', header)
    if not m:
        raise ValueError(f"No Lattice= entry found in {xyz_path}")
    vals = [float(x) for x in m.group(1).split()]
    L = np.array(vals).reshape(3, 3)
    off_diag = L - np.diag(np.diag(L))
    if not np.allclose(off_diag, 0.0, atol=1e-6):
        raise NotImplementedError(f"{xyz_path} has a non-diagonal lattice, extend this script for that case")
    return np.diag(L).copy()


def iter_xyz_frames_species(xyz_path: Path, species: str = "Xe"):
    """Stream an extended-XYZ file frame by frame, yielding only the rows of
    the requested species as an (n, 4) array of [id, x, y, z]. Never holds
    more than one frame in memory, and never touches the other species'
    columns beyond a cheap string comparison."""
    with open(xyz_path, "r") as f:
        while True:
            count_line = f.readline()
            if not count_line:
                return
            n = int(count_line.strip())
            f.readline()  # header line, not needed here
            rows = []
            for _ in range(n):
                parts = f.readline().split()
                if parts[0] == species:
                    rows.append((float(parts[4]), float(parts[1]), float(parts[2]), float(parts[3])))
            yield np.array(rows, dtype=np.float64) if rows else np.empty((0, 4))


def load_csv_chunk(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.sort_values(["structure_id", "atom_index"]).reset_index(drop=True)


def pbc_dist_matrix(a: np.ndarray, b: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Pairwise PBC-aware distances between rows of a (n,3) and b (m,3) -> (n,m)."""
    d = a[:, None, :] - b[None, :, :]
    d -= box * np.round(d / box)
    return np.linalg.norm(d, axis=-1)


def process_chunk(xyz_path: Path, csv_path: Path, box: np.ndarray):
    """Returns:
      xyz_xe   : (n_frames, n_xe, 4) array of [id, x, y, z] from the XYZ file
      agree    : (n_matched,) bool array, CHECK A per-row agreement
      match_d  : (n_matched,) float array, CHECK A matching distances (QC)
    """
    df = load_csv_chunk(csv_path)
    xyz_frames = list(iter_xyz_frames_species(xyz_path, "Xe"))

    n_atoms_ref = xyz_frames[0].shape[0]
    xyz_xe = np.empty((len(xyz_frames), n_atoms_ref, 4))

    agree_flags = []
    match_dists = []

    for sid, g in df.groupby("structure_id", sort=True):
        frame = xyz_frames[sid]
        if frame.shape[0] != n_atoms_ref:
            raise ValueError(f"{xyz_path} frame {sid} has {frame.shape[0]} Xe atoms, expected {n_atoms_ref}")
        xyz_xe[sid] = frame

        csv_pos = g[["x", "y", "z"]].to_numpy(dtype=np.float64)
        csv_idx = g["atom_index"].to_numpy()
        d = pbc_dist_matrix(csv_pos, frame[:, 1:4], box)
        nn = d.argmin(axis=1)
        matched_id = frame[nn, 0]
        agree_flags.append(csv_idx == matched_id)
        match_dists.append(d[np.arange(len(nn)), nn])

    return xyz_xe, np.concatenate(agree_flags), np.concatenate(match_dists)


def continuity_check(xyz_xe: np.ndarray, box: np.ndarray, dt_fs: float):
    """CHECK B: does the same set of ids recur every frame, and does each id
    move by only a small amount frame to frame."""
    n_frames, n_atoms, _ = xyz_xe.shape
    id_sets = [frozenset(xyz_xe[t, :, 0].tolist()) for t in range(n_frames)]
    first_set = id_sets[0]
    same_set_everywhere = all(s == first_set for s in id_sets)

    max_step = np.zeros(n_frames - 1)
    if same_set_everywhere:
        ids_sorted = sorted(first_set)
        order = np.empty((n_frames, n_atoms), dtype=int)
        for t in range(n_frames):
            id_to_row = {v: i for i, v in enumerate(xyz_xe[t, :, 0])}
            order[t] = [id_to_row[i] for i in ids_sorted]
        pos = np.stack([xyz_xe[t, order[t], 1:4] for t in range(n_frames)])
        for t in tqdm(range(1, n_frames), desc="Continuity check (by XYZ id)"):
            d = pos[t] - pos[t - 1]
            d -= box * np.round(d / box)
            max_step[t - 1] = np.linalg.norm(d, axis=-1).max()

    return same_set_everywhere, max_step


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--dt-fs", type=float, default=100.0)
    ap.add_argument("--chunk-glob", default="chunk_*")
    ap.add_argument("--xyz-name-fmt", default="cnt_traj_2ns-10ns_out_chunk_{i}.xyz")
    ap.add_argument("--csv-name-fmt", default="output_prediction_chunk_{i}.csv")
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--save-xyz-xe-arrays", type=Path, default=None,
                     help="optional path to dump the raw XYZ Xe id/position arrays for reuse")
    args = ap.parse_args()

def discover_chunks(data_dir: Path, chunk_glob: str):
    """Same fix as step1_ingest_and_track.py: find chunk directories and
    return them in true chronological (integer) order, with the exact
    numeric suffix string each one uses for its own files. Plain string
    sorting breaks under mixed padding widths (chunk_011 sorts before
    chunk_02), and a reconstructed, uniformly-padded index breaks the same
    way, silently or with a FileNotFoundError."""
    chunk_dirs = list(data_dir.glob(chunk_glob))
    if not chunk_dirs:
        raise FileNotFoundError(f"No chunk folders matching {chunk_glob!r} under {data_dir}")
    chunks = []
    for cdir in chunk_dirs:
        m = re.search(r"(\d+)$", cdir.name)
        if not m:
            raise ValueError(f"Could not find a trailing chunk number in directory name: {cdir.name!r}")
        chunks.append((int(m.group(1)), m.group(1), cdir))
    chunks.sort(key=lambda t: t[0])
    return chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--dt-fs", type=float, default=100.0)
    ap.add_argument("--chunk-glob", default="chunk_*")
    ap.add_argument("--xyz-name-fmt", default="cnt_traj_2ns-10ns_out_chunk_{i}.xyz")
    ap.add_argument("--csv-name-fmt", default="output_prediction_chunk_{i}.csv")
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--save-xyz-xe-arrays", type=Path, default=None,
                     help="optional path to dump the raw XYZ Xe id/position arrays for reuse")
    args = ap.parse_args()

    chunks = discover_chunks(args.data_dir, args.chunk_glob)
    print(f"Found {len(chunks)} chunk folders, numbered {chunks[0][0]}..{chunks[-1][0]}")

    first_num, first_suffix, first_cdir = chunks[0]
    box = read_lattice(first_cdir / args.xyz_name_fmt.format(i=first_suffix))
    print(f"Box: Lx={box[0]:.4f} Ly={box[1]:.4f} Lz={box[2]:.4f}")

    jobs = []
    for num, suffix, cdir in chunks:
        xyz_p = cdir / args.xyz_name_fmt.format(i=suffix)
        csv_p = cdir / args.csv_name_fmt.format(i=suffix)
        jobs.append((xyz_p, csv_p))

    results = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {ex.submit(process_chunk, xp, cp, box): idx for idx, (xp, cp) in enumerate(jobs)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks (CHECK A)"):
            results[futures[fut]] = fut.result()

    xyz_xe_all = np.concatenate([r[0] for r in results], axis=0)
    agree_all = np.concatenate([r[1] for r in results])
    dist_all = np.concatenate([r[2] for r in results])

    print()
    print("=== CHECK A: does CSV atom_index equal the XYZ id at the same position? ===")
    print(f"  rows checked        : {agree_all.size}")
    print(f"  agree               : {agree_all.sum()}  ({100*agree_all.mean():.2f}%)")
    print(f"  position-match dist : mean={dist_all.mean():.6f} Å, max={dist_all.max():.6f} Å "
          f"(should be ~0, this is just confirming CSV rows correspond to real XYZ Xe atoms)")

    same_set, max_step = continuity_check(xyz_xe_all, box, args.dt_fs)
    print()
    print("=== CHECK B: is the XYZ id itself a persistent per-atom tag? ===")
    print(f"  same 4 (or N) ids present in every frame across every chunk : {same_set}")
    if same_set:
        print(f"  per-step displacement of each id : mean={max_step.mean():.4f} Å, max={max_step.max():.4f} Å")

    print()
    print("=== Verdict ===")
    if agree_all.mean() > 0.999:
        print("CSV atom_index matches the XYZ id directly. Step 1 can track atoms by atom_index/id "
              "and the Hungarian-matching step is unnecessary.")
    elif same_set and max_step.max() < 2.0:
        print("CSV atom_index does NOT reliably match the XYZ id, but the XYZ id IS a genuine, "
              "persistent per-atom tag (same ids every frame, small physically-reasonable steps). "
              "Step 1 should track atoms by the XYZ id column, matched to CSV rows by position "
              "per frame, not by atom_index directly.")
    else:
        print("Neither atom_index nor the XYZ id behaves as a clean persistent tag in this data. "
              "Falling back to pure position-based (Hungarian) tracking, as in the previous script, "
              "is the safe default here.")

    if args.save_xyz_xe_arrays:
        np.savez_compressed(args.save_xyz_xe_arrays, xyz_xe=xyz_xe_all, box=box, dt_fs=args.dt_fs)
        print(f"\nSaved raw XYZ Xe id/position arrays to {args.save_xyz_xe_arrays}")


if __name__ == "__main__":
    main()
