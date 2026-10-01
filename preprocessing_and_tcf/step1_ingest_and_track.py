#!/usr/bin/env python3
"""
Step 1 - Ingest and assemble Xe@CNT MD + NMR-ML data.

Identity scheme, confirmed against the real dataset before this was written:
  - The XYZ `id` column is the persistent, per-atom LAMMPS id. Verified: the
    same set of ids recurs in every one of the 400000 checked rows across
    all 10 chunks, with small, physically sane per-step displacements
    (mean 0.297 A, max 0.669 A over 100 fs).
  - The CSV `atom_index` column does NOT match that id (1.77% coincidental
    agreement) and is not used for identity anywhere below.
  - Frame order is the same in both files (structure_id in the CSV
    corresponds 1:1 to frame order in the XYZ). Every frame is matched
    between the two files by coordinates: the CSV row for a given Xe atom
    and its XYZ row occupy the same position (to floating point) within
    that frame, since both come from the same MD snapshot. This coordinate
    match is checked, and its distance recorded, for every atom in every
    frame, not spot-checked once.

Pipeline
--------
1. Get per-frame Xe (id, x, y, z) from the XYZ files, either by parsing them
   directly, or by loading a cached array from step0_verify's
   --save-xyz-xe-arrays output, if you point --xyz-xe-cache at it (skips
   re-parsing the XYZ files entirely).
2. Parse the CSV chunks (id-independent columns: x, y, z, sigma_iso, tensor_*).
3. Per frame: match each CSV row to the XYZ row at the same coordinates
   (nearest neighbor, PBC-aware). Record the match distance as QC.
4. Sort every frame into a persistent track order using the XYZ id
   (ascending id order, works because the same id set recurs every frame).
5. Flag and clean NMR-ML outliers: the shielding model occasionally produces
   nonsense values for some local environments. Detected via Tukey's IQR
   rule on sigma_iso, pooled over every frame and track in the system, same
   convention as code_plot_delta_iso_loading_outliers.py. Flagged points are
   never dropped (that would break Step 2's uniform-sampling assumption),
   they're linearly interpolated per track from its own nearest valid
   neighbors, so the time series stays continuous. Only the ML-predicted
   tensor/sigma_iso are touched, positions are untouched, they come from the
   (separately validated) MLIP trajectory, not the NMR-ML model, so they
   aren't a plausible source of this kind of glitch. On by default,
   --disable-outlier-filter turns it off.
6. Save one consolidated .npz for Step 2.

Usage
-----
    python step1_ingest_and_track.py \\
        --data-dir /path/to/cnt_10_0_u23_Xe4_10ns \\
        --system-name cnt_10_0_u23_Xe4_10ns \\
        --dt-fs 100.0 \\
        --xyz-xe-cache ./xyz_xe_raw.npz \\
        --out ./cnt_10_0_u23_Xe4_10ns_assembled.npz
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

TENSOR_COLS = [
    "tensor_xx", "tensor_xy", "tensor_xz",
    "tensor_yx", "tensor_yy", "tensor_yz",
    "tensor_zx", "tensor_zy", "tensor_zz",
]


def read_lattice(xyz_path: Path) -> np.ndarray:
    with open(xyz_path, "r") as f:
        f.readline()
        header = f.readline()
    m = re.search(r'Lattice="([^"]+)"', header)
    if not m:
        raise ValueError(f"No Lattice= entry found in {xyz_path}")
    L = np.array([float(x) for x in m.group(1).split()]).reshape(3, 3)
    if not np.allclose(L - np.diag(np.diag(L)), 0.0, atol=1e-6):
        raise NotImplementedError(f"{xyz_path} has a non-diagonal lattice, extend read_lattice for that case")
    return np.diag(L).copy()


def iter_xyz_frames_species(xyz_path: Path, species: str = "Xe"):
    """Stream an extended-XYZ file frame by frame, yielding only the requested
    species as an (n, 4) array of [id, x, y, z]."""
    with open(xyz_path, "r") as f:
        while True:
            count_line = f.readline()
            if not count_line:
                return
            n = int(count_line.strip())
            f.readline()  # header, not needed here
            rows = []
            for _ in range(n):
                parts = f.readline().split()
                if parts[0] == species:
                    rows.append((float(parts[4]), float(parts[1]), float(parts[2]), float(parts[3])))
            yield np.array(rows, dtype=np.float64)


def pbc_dist_matrix(a: np.ndarray, b: np.ndarray, box: np.ndarray) -> np.ndarray:
    d = a[:, None, :] - b[None, :, :]
    d -= box * np.round(d / box)
    return np.linalg.norm(d, axis=-1)


def iqr_outlier_bounds(values: np.ndarray, k: float = 2.5):
    """Tukey's IQR rule: values outside [Q1 - k*IQR, Q3 + k*IQR] are
    outliers. Same convention as code_plot_delta_iso_loading_outliers.py
    (there applied to delta_iso = sigma_ref - sigma_iso, an affine, sign-
    flipped rescaling of sigma_iso; IQR outlier detection is invariant to
    that, the flagged point set is identical either way)."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def filter_outliers(sigma_iso: np.ndarray, tensor: np.ndarray, k: float = 2.5):
    """
    Detect NMR-ML model glitches via Tukey's IQR rule on sigma_iso, pooled
    across every frame and every track in this system. Flagged entries are
    linearly interpolated (per track, from that track's own nearest valid
    frames), not dropped, so the cleaned time series stays continuous and
    uniformly sampled for Step 2's FFT-based correlation function. Tensor
    components are interpolated the same way as sigma_iso; since linear
    interpolation commutes with the trace, the cleaned sigma_iso and the
    cleaned tensor's own trace/3 agree exactly.

    Returns sigma_iso_clean, tensor_clean, outlier_mask (n_frames, n_atoms),
    bounds (lower, upper). The mask is saved alongside the cleaned data, not
    discarded, so exactly which points were touched stays auditable.
    """
    n_frames, n_atoms = sigma_iso.shape
    lower, upper = iqr_outlier_bounds(sigma_iso.ravel(), k)
    mask = (sigma_iso < lower) | (sigma_iso > upper)
    n_out = int(mask.sum())
    frac = 100.0 * n_out / mask.size
    print(f"Outlier filter (IQR k={k}): sigma_iso bounds [{lower:.2f}, {upper:.2f}] ppm, "
          f"{n_out} / {mask.size} points flagged ({frac:.3f}%)")
    if frac > 2.0:
        print(f"WARNING: {frac:.2f}% flagged is high for k={k} (a clean IQR-normal distribution flags "
              f"~0.7%), this may be catching real physical variation rather than model glitches, "
              f"worth a manual look before trusting the cleaned output.")

    sigma_iso_clean = sigma_iso.copy()
    tensor_clean = tensor.copy()
    frame_idx = np.arange(n_frames)

    for a in range(n_atoms):
        bad = mask[:, a]
        if not bad.any():
            continue
        good = ~bad
        if good.sum() < 2:
            print(f"  WARNING: track {a} has fewer than 2 valid (non-outlier) frames, cannot "
                  f"interpolate, leaving its {int(bad.sum())} flagged points unchanged.")
            continue
        print(f"  track {a}: {int(bad.sum())} flagged points ({100*bad.sum()/n_frames:.3f}%), "
              f"interpolating from its own neighboring frames")
        sigma_iso_clean[bad, a] = np.interp(frame_idx[bad], frame_idx[good], sigma_iso[good, a])
        for i in range(3):
            for j in range(3):
                tensor_clean[bad, a, i, j] = np.interp(frame_idx[bad], frame_idx[good], tensor[good, a, i, j])

    return sigma_iso_clean, tensor_clean, mask, (lower, upper)


def process_chunk(xyz_path, csv_path: Path, box: np.ndarray, xyz_xe_chunk):
    """
    xyz_xe_chunk, if given, is this chunk's slice of a pre-parsed
    (n_frames, n_atoms, 4) [id,x,y,z] array, skipping the XYZ file entirely.
    Otherwise xyz_path is parsed directly.

    Returns pos, sigma_iso, tensor, ids, match_dist, every array indexed
    [frame, track] with track 0..n_atoms-1 in ascending-id order.
    """
    df = pd.read_csv(csv_path).sort_values(["structure_id", "atom_index"]).reset_index(drop=True)

    if xyz_xe_chunk is not None:
        xyz_frames = xyz_xe_chunk
        n_frames, n_atoms, _ = xyz_frames.shape
    else:
        xyz_frames = list(iter_xyz_frames_species(xyz_path, "Xe"))
        n_frames = len(xyz_frames)
        n_atoms = xyz_frames[0].shape[0]

    pos = np.empty((n_frames, n_atoms, 3))
    siso = np.empty((n_frames, n_atoms))
    tensor = np.empty((n_frames, n_atoms, 3, 3))
    ids = np.empty((n_frames, n_atoms))
    match_dist = np.empty((n_frames, n_atoms))

    for sid, g in df.groupby("structure_id", sort=True):
        frame = xyz_frames[sid]
        if frame.shape[0] != n_atoms:
            raise ValueError(f"{csv_path} frame {sid}: {frame.shape[0]} Xe atoms in XYZ, expected {n_atoms}")

        csv_pos = g[["x", "y", "z"]].to_numpy(dtype=np.float64)
        csv_siso = g["sigma_iso"].to_numpy(dtype=np.float64)
        csv_tensor = g[TENSOR_COLS].to_numpy(dtype=np.float64).reshape(-1, 3, 3)

        d = pbc_dist_matrix(frame[:, 1:4], csv_pos, box)  # (xyz_row, csv_row)
        nn = d.argmin(axis=1)
        dmin = d[np.arange(n_atoms), nn]

        order = np.argsort(frame[:, 0])  # canonical track order = ascending XYZ id

        pos[sid] = frame[order, 1:4]
        ids[sid] = frame[order, 0]
        siso[sid] = csv_siso[nn][order]
        tensor[sid] = csv_tensor[nn][order]
        match_dist[sid] = dmin[order]

    return pos, siso, tensor, ids, match_dist


def discover_chunks(data_dir: Path, chunk_glob: str):
    """
    Find chunk directories and return them in true chronological order,
    with the exact numeric suffix string each one uses for its files.

    Deliberately does NOT rely on plain string-sorting the directory names,
    or on a reconstructed, uniformly-padded index. Some datasets mix
    padding widths across the same run (chunk_01..chunk_10, then
    chunk_011..chunk_100), and under naive string sort "chunk_011" sorts
    BEFORE "chunk_02", silently reordering frames with no error at all.
    Instead, the numeric part is extracted from each directory's own name,
    sorted as an integer, and that exact string (whatever padding it uses)
    is reused verbatim for that chunk's XYZ/CSV filenames, so there's no
    assumption that every chunk shares one padding convention.

    Returns a list of (chunk_number: int, suffix: str, path: Path), in
    correct chronological order.
    """
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

    numbers = [c[0] for c in chunks]
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"Duplicate chunk numbers found under {data_dir}: {numbers}")
    gaps = [b - a for a, b in zip(numbers, numbers[1:]) if b - a != 1]
    if gaps:
        print(f"WARNING: chunk numbering under {data_dir} is not contiguous (found {numbers[0]}..{numbers[-1]}, "
              f"{len(numbers)} chunks), double check nothing is silently missing.")
    return chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--system-name", required=True)
    ap.add_argument("--dt-fs", type=float, default=100.0)
    ap.add_argument("--chunk-glob", default="chunk_*")
    ap.add_argument("--xyz-name-fmt", default="cnt_traj_2ns-10ns_out_chunk_{i}.xyz")
    ap.add_argument("--csv-name-fmt", default="output_prediction_chunk_{i}.csv")
    ap.add_argument("--xyz-xe-cache", type=Path, default=None,
                     help="optional .npz from step0_verify's --save-xyz-xe-arrays, skips re-parsing the XYZ files")
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--outlier-iqr-k", type=float, default=2.5,
                     help="Tukey IQR multiplier for flagging NMR-ML sigma_iso outliers "
                          "(matches code_plot_delta_iso_loading_outliers.py's convention)")
    ap.add_argument("--disable-outlier-filter", action="store_true",
                     help="skip outlier detection/interpolation, keep raw NMR-ML output as-is")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_path = args.out or Path(f"{args.system_name}_assembled.npz")
    chunks = discover_chunks(args.data_dir, args.chunk_glob)
    print(f"Found {len(chunks)} chunk folders, numbered {chunks[0][0]}..{chunks[-1][0]}")

    if args.xyz_xe_cache:
        cached = np.load(args.xyz_xe_cache)
        box = cached["box"]
        xyz_xe_full = cached["xyz_xe"]
        print(f"Loaded cached XYZ Xe array {xyz_xe_full.shape} from {args.xyz_xe_cache}, not re-parsing XYZ files")
    else:
        first_num, first_suffix, first_cdir = chunks[0]
        box = read_lattice(first_cdir / args.xyz_name_fmt.format(i=first_suffix))
        xyz_xe_full = None
    print(f"Box: Lx={box[0]:.4f} Ly={box[1]:.4f} Lz={box[2]:.4f}")

    jobs = []
    frame_cursor = 0
    for num, suffix, cdir in chunks:
        csv_p = cdir / args.csv_name_fmt.format(i=suffix)
        xyz_p = cdir / args.xyz_name_fmt.format(i=suffix)
        n_frames_chunk = pd.read_csv(csv_p, usecols=["structure_id"])["structure_id"].nunique()
        chunk_xyz_slice = None
        if xyz_xe_full is not None:
            chunk_xyz_slice = xyz_xe_full[frame_cursor:frame_cursor + n_frames_chunk]
        jobs.append((xyz_p, csv_p, chunk_xyz_slice))
        frame_cursor += n_frames_chunk

    results = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {
            ex.submit(process_chunk, xp, cp, box, xslice): idx
            for idx, (xp, cp, xslice) in enumerate(jobs)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Assembling chunks"):
            results[futures[fut]] = fut.result()

    pos = np.concatenate([r[0] for r in results], axis=0)
    siso = np.concatenate([r[1] for r in results], axis=0)
    tensor = np.concatenate([r[2] for r in results], axis=0)
    ids = np.concatenate([r[3] for r in results], axis=0)
    match_dist = np.concatenate([r[4] for r in results], axis=0)

    n_frames, n_atoms = siso.shape
    print(f"Assembled {n_frames} frames, {n_atoms} Xe atoms per frame")
    print(f"Coordinate-match QC: mean={match_dist.mean():.6f} A, max={match_dist.max():.6f} A "
          f"(should be ~0, every atom in every frame)")

    id_sets_equal = bool(np.all(ids == ids[0], axis=0).all())
    print(f"Same track identity (by XYZ id) preserved in every frame: {id_sets_equal}")
    if match_dist.max() > 1e-3 or not id_sets_equal:
        print("WARNING: coordinate match or id consistency failed somewhere, inspect before trusting this output.")

    if args.disable_outlier_filter:
        outlier_mask = np.zeros_like(siso, dtype=bool)
        outlier_bounds = (np.nan, np.nan)
    else:
        siso, tensor, outlier_mask, outlier_bounds = filter_outliers(siso, tensor, k=args.outlier_iqr_k)

    time_fs = np.arange(n_frames) * args.dt_fs

    np.savez_compressed(
        out_path,
        system_name=args.system_name,
        box=box,
        dt_fs=args.dt_fs,
        time_fs=time_fs,
        track_ids=ids[0],  # the persistent XYZ ids, in track order
        pos=pos,
        sigma_iso=siso,
        tensor=tensor,
        coord_match_dist=match_dist,
        outlier_mask=outlier_mask,
        outlier_bounds=np.array(outlier_bounds),
        outlier_iqr_k=(np.nan if args.disable_outlier_filter else args.outlier_iqr_k),
    )
    print(f"Wrote {out_path}  (pos {pos.shape}, sigma_iso {siso.shape}, tensor {tensor.shape})")


if __name__ == "__main__":
    main()
