"""Read the NIPS4B training labels as a table: one row per clip, one 0/1 column per species.

Source files (in data/NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS/):
  numero_file_train.csv                 687 rows, no header: clip number, 87 x (0/1), duration
  nips4b_birdchallenge_espece_list.csv  species table: class number 1..87, class name, type, ...
"""

import pandas as pd


def load_species(species_csv, num_classes=87):
    """Species table sorted by class number 1..87 (the 'Empty' row is not a class, so it is dropped)."""
    df = pd.read_csv(species_csv)
    df = df[pd.to_numeric(df["class number"], errors="coerce").notna()].copy()
    df["class number"] = df["class number"].astype(int)
    df = df.sort_values("class number").reset_index(drop=True)
    assert list(df["class number"]) == list(range(1, num_classes + 1)), (
        f"Expected class numbers 1..{num_classes} in {species_csv}"
    )
    return df


def load_labels(labels_csv, species_csv, num_classes=87):
    """Return a DataFrame: index = clip id ("trainfile001"), columns = species names, values = 0/1."""
    raw = pd.read_csv(labels_csv, header=None)
    assert raw.shape[1] == num_classes + 2, (
        f"Expected {num_classes + 2} columns (id + {num_classes} labels + duration), got {raw.shape[1]}"
    )

    clip_ids = [f"trainfile{int(n):03d}" for n in raw[0]]
    labels = raw.iloc[:, 1:num_classes + 1]
    assert labels.isin([0, 1]).all().all(), "Labels must be 0 or 1"

    names = load_species(species_csv, num_classes)["class name"].tolist()
    return pd.DataFrame(labels.to_numpy(dtype="int8"), index=pd.Index(clip_ids, name="file_id"), columns=names)


def labels_from_config(cfg):
    return load_labels(cfg.paths.labels_csv, cfg.paths.species_csv, cfg.num_classes)
