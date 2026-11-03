"""Split the 687 training clips into 5 folds, so every species appears in every fold.

    python -m src.data.folds

Writes data/folds.csv (columns: file_id, fold). This file is committed to git, so every
model (baseline, CNN, PANNs...) is tested on exactly the same clips.
"""

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from src.config import load_config
from src.data.labels import labels_from_config


def make_folds(labels, n_folds=5, seed=42):
    """Give each clip a fold number 0..n_folds-1, spreading every species evenly across folds."""
    splitter = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(labels), -1, dtype=int)
    for k, (_, val_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels.to_numpy())):
        fold[val_idx] = k
    return pd.DataFrame({"file_id": labels.index, "fold": fold})


def main():
    cfg = load_config()
    labels = labels_from_config(cfg)
    folds = make_folds(labels, cfg.cv.n_folds, cfg.seed)
    folds.to_csv(cfg.paths.folds_csv, index=False)

    print(f"{len(labels)} clips, {labels.shape[1]} species -> {cfg.paths.folds_csv}")
    for k in range(cfg.cv.n_folds):
        in_fold = labels[(folds["fold"] == k).to_numpy()]
        print(f"fold {k}: {len(in_fold)} clips, {(in_fold.sum() > 0).sum()} of {labels.shape[1]} species present")


if __name__ == "__main__":
    main()
