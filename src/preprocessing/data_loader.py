"""Load, clean, normalize, and split the Elliptic Bitcoin Dataset.

Stage 2 deliverables
--------------------
* load_elliptic()   -- merges CSVs, cleans, normalises, encodes labels,
                      returns temporal (train / val / test) splits.
* save_splits()     -- persists the three DataFrames to data/processed/ as
                      Parquet files for fast re-loading in later stages.
* load_splits()     -- reloads the saved Parquet files (avoids re-processing).

Run this file directly to execute the full Stage 2 pipeline and verify
acceptance criteria (non-overlapping splits, zero nulls, correct shapes).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# ---------------------------------------------------------------------------
# Column layout of the Elliptic features CSV (no header row)
# ---------------------------------------------------------------------------
FEATURE_COLUMN_NAMES = [config.TX_ID_COL, config.TIME_STEP_COL] + [
    f"feature_{i}" for i in range(1, 166)
]

FEATURE_COLS = [f"feature_{i}" for i in range(1, 166)]   # 165 raw features

# Binary label mapping  (Elliptic: "1" = illicit, "2" = licit  ← note order)
LABEL_MAP = {
    config.ILLICIT_LABEL: 1,   # "2" in raw CSV -> 1 (fraud)
    config.LICIT_LABEL:   0,   # "1" in raw CSV -> 0 (legit)
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {path}\n"
            f"Download from Kaggle (ellipticco/elliptic-data-set) and place "
            f"CSVs in {config.RAW_DATA_DIR}:\n"
            f"  - elliptic_txs_features.csv\n"
            f"  - elliptic_txs_classes.csv\n"
            f"  - elliptic_txs_edgelist.csv"
        )


def _load_features(path: Path) -> pd.DataFrame:
    """Load features CSV (no header row, Kaggle standard)."""
    _require_file(path)
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(FEATURE_COLUMN_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_COLUMN_NAMES)} feature columns, "
            f"got {df.shape[1]} in {path}"
        )
    df.columns = FEATURE_COLUMN_NAMES
    return df


def _load_classes(path: Path) -> pd.DataFrame:
    """Load classes CSV (header: txId, class)."""
    _require_file(path)
    df = pd.read_csv(path)
    # Normalise column names defensively
    df.columns = [c.strip() for c in df.columns]
    if config.TX_ID_COL not in df.columns:
        df = df.rename(columns={df.columns[0]: config.TX_ID_COL})
    if config.CLASS_COL not in df.columns:
        df = df.rename(columns={df.columns[1]: config.CLASS_COL})
    # Coerce class column to string so label lookups are consistent
    df[config.CLASS_COL] = df[config.CLASS_COL].astype(str).str.strip()
    return df


def _load_edgelist(path: Path) -> pd.DataFrame:
    """Load edgelist CSV (txId1 -> txId2)."""
    _require_file(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "txId1" not in df.columns or "txId2" not in df.columns:
        df = df.rename(columns={df.columns[0]: "txId1", df.columns[1]: "txId2"})
    return df


def _split_by_timestep(df: pd.DataFrame, time_steps: range) -> pd.DataFrame:
    return df[df[config.TIME_STEP_COL].isin(time_steps)].copy()


def _encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw string class labels to binary integers (0 = licit, 1 = illicit)."""
    df = df.copy()
    df["label"] = df[config.CLASS_COL].map(LABEL_MAP)
    # Rows whose class was not in LABEL_MAP (e.g. 'unknown') become NaN -> drop
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df


def _normalize_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    StandardScaler fit on train only, applied to all splits.

    Only the 164 raw feature columns are scaled; txId, time step, class,
    and label are left untouched.
    """
    scaler = StandardScaler()

    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
    val_df[FEATURE_COLS]   = scaler.transform(val_df[FEATURE_COLS])
    test_df[FEATURE_COLS]  = scaler.transform(test_df[FEATURE_COLS])

    return train_df, val_df, test_df


def _assert_splits_valid(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Verify non-overlapping splits and zero nulls (acceptance criteria)."""
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        null_count = df[FEATURE_COLS + ["label"]].isnull().sum().sum()
        if null_count > 0:
            bad_cols = df.columns[df.isnull().any()].tolist()
            raise ValueError(f"{name} split has {null_count} nulls in: {bad_cols}")

    train_ids = set(train_df[config.TX_ID_COL])
    val_ids   = set(val_df[config.TX_ID_COL])
    test_ids  = set(test_df[config.TX_ID_COL])

    if train_ids & val_ids:
        raise ValueError("Train and validation splits overlap on transaction IDs.")
    if train_ids & test_ids:
        raise ValueError("Train and test splits overlap on transaction IDs.")
    if val_ids & test_ids:
        raise ValueError("Validation and test splits overlap on transaction IDs.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_elliptic(
    features_path: Path | None = None,
    classes_path:  Path | None = None,
    edgelist_path: Path | None = None,
    include_unknown: bool = False,
    normalize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the Elliptic Bitcoin Dataset, clean it, and return temporal splits.

    Steps
    -----
    1. Load three CSVs (features, classes, edgelist).
    2. Merge features + classes on txId.
    3. Drop 'unknown'-labelled rows (default) or keep them if include_unknown=True.
    4. Drop any remaining NaN rows.
    5. Encode class labels to binary integers (0 = licit, 1 = illicit).
    6. Split by time-step into train / val / test (no data leakage).
    7. StandardScaler normalisation fit on train, applied to all splits.
    8. Attach the edgelist as df.attrs["edgelist"] on each split.

    Returns
    -------
    train_df, val_df, test_df, edgelist
        - Three non-overlapping DataFrames with a 'label' column.
        - edgelist DataFrame (txId1, txId2).
    """
    features_path = features_path or config.ELLIPTIC_FEATURES_FILE
    classes_path  = classes_path  or config.ELLIPTIC_CLASSES_FILE
    edgelist_path = edgelist_path or config.ELLIPTIC_EDGELIST_FILE

    print("[Stage 2] Loading raw CSVs ...")
    features = _load_features(features_path)
    classes  = _load_classes(classes_path)
    edgelist = _load_edgelist(edgelist_path)
    print(f"  features : {features.shape}")
    print(f"  classes  : {classes.shape}")
    print(f"  edgelist : {edgelist.shape}")

    # --- Merge ---
    merged = features.merge(classes, on=config.TX_ID_COL, how="inner")
    print(f"[Stage 2] After merge: {merged.shape}")

    # --- Remove unknowns ---
    if not include_unknown:
        before = len(merged)
        merged = merged[merged[config.CLASS_COL] != config.UNKNOWN_LABEL].copy()
        print(f"[Stage 2] Dropped {before - len(merged)} 'unknown' rows -> {len(merged)} remain")

    # --- Drop remaining NaNs ---
    before = len(merged)
    merged = merged.dropna()
    if before - len(merged) > 0:
        print(f"[Stage 2] Dropped {before - len(merged)} NaN rows")

    # --- Label encoding ---
    merged = _encode_labels(merged)
    print(f"[Stage 2] Label distribution:\n{merged['label'].value_counts().to_string()}")

    # --- Temporal split ---
    train_df = _split_by_timestep(merged, config.TRAIN_TIME_STEPS)
    val_df   = _split_by_timestep(merged, config.VAL_TIME_STEPS)
    test_df  = _split_by_timestep(merged, config.TEST_TIME_STEPS)

    # --- Normalisation (fit on train only) ---
    if normalize:
        print("[Stage 2] Normalising features (StandardScaler fit on train) ...")
        train_df, val_df, test_df = _normalize_features(train_df, val_df, test_df)

    # --- Attach edgelist as metadata ---
    for split in (train_df, val_df, test_df):
        split.attrs["edgelist"] = edgelist

    # --- Validate ---
    _assert_splits_valid(train_df, val_df, test_df)
    print("[Stage 2] OK Acceptance criteria passed -- no nulls, no overlapping IDs.")

    return train_df, val_df, test_df, edgelist


def save_splits(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    edgelist: pd.DataFrame,
    out_dir:  Path | None = None,
) -> None:
    """Persist cleaned splits to data/processed/ as Parquet files."""
    out_dir = out_dir or config.PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parquet cannot serialise DataFrame attrs — strip before saving
    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        clean = df.copy()
        clean.attrs = {}
        clean.to_parquet(out_dir / f"{name}.parquet", index=False)

    edgelist.to_parquet(out_dir / "edgelist.parquet", index=False)
    print(f"[Stage 2] Splits saved to {out_dir}")


def load_splits(
    in_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reload previously saved Parquet splits (fast path, skips CSV re-parsing)."""
    in_dir = in_dir or config.PROCESSED_DATA_DIR

    train_df = pd.read_parquet(in_dir / "train.parquet")
    val_df   = pd.read_parquet(in_dir / "val.parquet")
    test_df  = pd.read_parquet(in_dir / "test.parquet")
    edgelist = pd.read_parquet(in_dir / "edgelist.parquet")

    # NOTE: edgelist is returned as a separate value — do NOT store it in
    # df.attrs to avoid ambiguous-truth-value errors when pandas compares
    # attrs across DataFrames during pd.concat and similar operations.

    return train_df, val_df, test_df, edgelist


# ---------------------------------------------------------------------------
# Stage 2 runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train, val, test, edges = load_elliptic()

    print("\n=== Stage 2 Results ===")
    print(f"  Train  : {train.shape}  (time steps {config.TRAIN_TIME_STEPS.start}–{config.TRAIN_TIME_STEPS.stop - 1})")
    print(f"  Val    : {val.shape}    (time steps {config.VAL_TIME_STEPS.start}–{config.VAL_TIME_STEPS.stop - 1})")
    print(f"  Test   : {test.shape}   (time steps {config.TEST_TIME_STEPS.start}–{config.TEST_TIME_STEPS.stop - 1})")
    print(f"  Edges  : {edges.shape}")
    print(f"  Train nulls : {train[FEATURE_COLS + ['label']].isnull().sum().sum()}")
    print(f"  Val nulls   : {val[FEATURE_COLS   + ['label']].isnull().sum().sum()}")
    print(f"  Test nulls  : {test[FEATURE_COLS  + ['label']].isnull().sum().sum()}")

    save_splits(train, val, test, edges)
    print("\n[Stage 2] [DONE] Complete -- splits saved to data/processed/")
