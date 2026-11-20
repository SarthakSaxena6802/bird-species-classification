"""
OPTIONAL heavy cleaning of the spectrograms - used as a separate ablation row.

The DEFAULT model input is the lightly cleaned spectrogram from
src/data/spectrogram.py (data/processed/). This script writes a heavily cleaned
copy to data/processed_clean/ with the same file names and shape, so training
only needs a different folder to compare "light vs heavy cleaning".

Why it is not the default: step 1 below subtracts each mel band's median over
time, which also removes sounds that last the whole clip - exactly what the
frog and the 9 insect classes (cicadas, crickets) sound like. The hard
threshold in step 2 also drops low-energy detail a CNN can use.

Cleaning steps (only on the real, unpadded part of each clip):
  1. Background removal: subtract each band's median over time and each frame's median over frequency.
  2. Thresholding: keep only energy above clean.threshold x robust noise spread (MAD).
  3. Speckle removal: small median filter.
  4. Blob removal: delete connected regions smaller than clean.min_blob_pixels.
  5. Rescale to 0..1.
Settings are in configs/default.yaml (clean section).

Run from the project root, after python -m src.data.spectrogram:
    python -m src.data.clean_spectrograms                       # clean everything
    python -m src.data.clean_spectrograms --png-only --examples auto
    python -m src.data.clean_spectrograms --png-only --examples bird=trainfile001 empty=trainfile010

--examples auto picks one bird-only, one frog, one insect-only and one empty
clip from the label files and saves before/after PNGs to data/processed_clean/png/.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from src.config import ROOT, load_config


def clean_spectrogram(spec, c):
    """spec: (n_mels, frames) log-mel, any scale; c: cfg.clean -> cleaned (n_mels, frames) in 0..1."""
    s = spec.astype(np.float32)

    # 1. Background removal along time (per band) and along frequency (per frame)
    s = s - np.median(s, axis=1, keepdims=True)
    s = s - np.median(s, axis=0, keepdims=True)

    # 2. Keep only energy clearly above the noise floor (MAD is not fooled by loud calls)
    mad = np.median(np.abs(s - np.median(s))) * 1.4826 + 1e-8
    s = np.where(s > c.threshold * mad, s, 0.0)

    # 3. Remove salt-and-pepper speckle
    s = ndimage.median_filter(s, size=tuple(c.median_size))

    # 4. Remove tiny isolated blobs (keep only call-sized regions)
    mask = s > 0
    labels, n = ndimage.label(mask)
    if n > 0:
        sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = sizes >= c.min_blob_pixels
        s = np.where(keep[labels], s, 0.0)

    # 5. Scale to 0..1
    peak = s.max()
    return (s / peak).astype(np.float32) if peak > 0 else s


def read_index(path):
    """{id: real number of frames} from *_index.csv (empty dict if missing)."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["id"], df["frames"]))


def clean_file(arr, real_frames, c):
    """arr: (1, n_mels, T). Cleans only the real part; the padded tail stays 0."""
    spec = arr[0]
    T = int(real_frames) if real_frames else spec.shape[1]
    out = np.zeros_like(spec, dtype=np.float32)
    out[:, :T] = clean_spectrogram(spec[:, :T], c)
    return out[np.newaxis]


def pick_examples(cfg):
    """One bird-only, one frog, one insect-only and one empty training clip, from the label files."""
    labels = pd.read_csv(ROOT / cfg.paths.labels_csv, header=None)
    species = pd.read_csv(ROOT / cfg.paths.species_csv)
    species.columns = [str(col).strip().lower().replace(" ", "_") for col in species.columns]
    species["class_number"] = pd.to_numeric(species["class_number"], errors="coerce")
    species = species.dropna(subset=["class_number"])
    kind = {int(k): str(t).strip().lower() for k, t in zip(species["class_number"], species["type"])}

    picks = {}
    for _, row in labels.iterrows():
        fid = f"trainfile{int(row[0]):03d}"
        types = {kind[k] for k in range(1, 88) if row[k] == 1}
        if not types and "empty" not in picks:
            picks["empty"] = fid
        elif types == {"bird"} and "bird" not in picks:
            picks["bird"] = fid
        elif "amphibian" in types and "frog" not in picks:
            picks["frog"] = fid
        elif types == {"insect"} and "insect" not in picks:
            picks["insect"] = fid
    return [f"{k}={v}" for k, v in picks.items()]


def save_png(before, after, path, title, sr, hop):
    """before/after: (n_mels, frames) in 0..1."""
    t = before.shape[1] * hop / sr
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, img, name in ((axes[0], before, "light cleaning (default input)"),
                          (axes[1], after, "heavy cleaning (ablation)")):
        ax.imshow(img, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, t, 0, img.shape[0]], vmin=0, vmax=1)
        ax.set_title(f"{title}  -  {name}")
        ax.set_ylabel("Mel band")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def process_split(in_root, out_root, split, c):
    files = sorted((in_root / split).glob("*.npy"))
    if not files:
        print(f"No .npy files in {in_root / split} - run python -m src.data.spectrogram first")
        return
    index = read_index(in_root / f"{split}_index.csv")
    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    empty = 0
    for i, f in enumerate(files):
        cleaned = clean_file(np.load(f), index.get(f.stem), c)
        np.save(out_dir / f.name, cleaned)
        empty += cleaned.max() == 0
        if (i + 1) % 100 == 0 or i + 1 == len(files):
            print(f"  {split}: {i + 1}/{len(files)}")
    print(f"{split}: {len(files)} files -> {out_dir}  ({empty} came out completely empty)")


def save_examples(in_root, out_root, examples, cfg):
    png_dir = out_root / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    for item in examples:
        label, _, fid = item.partition("=")
        split = "test" if fid.startswith("test") else "train"
        src = in_root / split / f"{fid}.npy"
        if not src.exists():
            print(f"  example {item}: {src} not found, skipped")
            continue
        before = np.load(src)
        after = clean_file(before, read_index(in_root / f"{split}_index.csv").get(fid), cfg.clean)
        out = png_dir / f"{label}_{fid}.png"
        save_png(before[0], after[0], out, f"{label}: {fid}", cfg.audio.sample_rate, cfg.audio.hop_length)
        print(f"  saved {out}")


def main():
    p = argparse.ArgumentParser(description="Optional heavy cleaning of NIPS4B spectrograms")
    p.add_argument("--examples", nargs="*", default=[], metavar="LABEL=ID",
                   help="'auto', or e.g. bird=trainfile001 frog=... insect=... empty=...")
    p.add_argument("--png-only", action="store_true",
                   help="only make the example PNGs, skip cleaning the whole dataset")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="config overrides, e.g. clean.threshold=2.0")
    args = p.parse_args()

    cfg = load_config(overrides=args.override)
    in_root = ROOT / cfg.paths.processed_dir
    out_root = ROOT / cfg.clean.out_dir

    if not args.png_only:
        for split in ("train", "test"):
            process_split(in_root, out_root, split, cfg.clean)
    examples = pick_examples(cfg) if args.examples == ["auto"] else args.examples
    if examples:
        print("Examples:", " ".join(examples))
        save_examples(in_root, out_root, examples, cfg)
    print(f"Done. Output is in {out_root}")


if __name__ == "__main__":
    main()
