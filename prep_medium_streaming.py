import os, gc, json, math, random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# --------- Paths / constants ---------
PROC  = Path("data/processed")
READY = Path("data/ready")
READY.mkdir(parents=True, exist_ok=True)

LABEL_COL = "Is Laundering"
DROP_COLS = {"Timestamp", "From Account", "To Account", LABEL_COL}

# tune for your RAM; 200k–400k is usually safe on 16 GB
BATCH_ROWS = 250_000

# SMOTE will be run on a stratified reservoir sample (not full dataset)
SMOTE_CAP  = 2_000_000      # total rows used for SMOTE (1–2M is OK on 16 GB)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def get_num_cols(sample_df: pd.DataFrame) -> list[str]:
    return [c for c in sample_df.columns if c not in DROP_COLS
            and pd.api.types.is_numeric_dtype(sample_df[c])]


def stream_batches(parquet_path: Path, columns: list[str] | None = None, batch_rows=BATCH_ROWS):
    """
    Stream a parquet file into pandas DataFrames safely.
    - Works with PyArrow ≥ 14
    - Skips non-numeric columns automatically
    """
    dataset = ds.dataset(str(parquet_path), format="parquet")

    for fragment in dataset.get_fragments():
        scanner = fragment.scanner(columns=columns, batch_size=batch_rows)
        reader = scanner.to_reader()

        for record_batch in reader:
            try:
                # Only keep numeric columns (skip complex dtypes)
                df = record_batch.to_pandas(ignore_metadata=True)
                df = df.select_dtypes(include=["number"]).astype("float32", errors="ignore")
                yield df
            except Exception as e:
                print(f"⚠️ Skipping problematic batch: {e}")
                continue





def partial_fit_scaler(parquet_path: Path, num_cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    seen = 0
    for df in stream_batches(parquet_path, columns=num_cols, batch_rows=BATCH_ROWS):
        X = df[num_cols].astype("float32").fillna(0.0).to_numpy(copy=False)
        scaler.partial_fit(X)
        seen += len(df)
        del df, X
        gc.collect()
    print(f"• Scaler fitted on ~{seen:,} rows")
    return scaler


def write_scaled(parquet_in: Path, parquet_out: Path, num_cols: list[str], scaler: StandardScaler):
    if parquet_out.exists():
        parquet_out.unlink()

    writer = None
    rows_written = 0

    cols_needed = list(num_cols) + [LABEL_COL] + list(DROP_COLS)
    cols_needed = sorted(set([c for c in cols_needed if c]))  # clean

    for df in stream_batches(parquet_in, columns=cols_needed, batch_rows=BATCH_ROWS):
        # assure all columns exist even if missing in this batch conversion
        for c in num_cols:
            if c not in df.columns:
                df[c] = 0.0

        X = df[num_cols].astype("float32").fillna(0.0).to_numpy(copy=False)
        Xs = scaler.transform(X).astype("float32", copy=False)
        df.loc[:, num_cols] = Xs

        # write append
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(parquet_out), table.schema, compression="snappy")
        writer.write_table(table)

        rows_written += len(df)
        del df, X, Xs, table
        gc.collect()

    if writer is not None:
        writer.close()

    print(f"✅ Scaled written → {parquet_out}  ({rows_written:,} rows)")


def stratified_reservoir_for_smote(parquet_scaled: Path, num_cols: list[str],
                                   cap_total=SMOTE_CAP, pos_ratio=None):
    """
    Build a stratified reservoir sample WITHOUT loading the full file:
    - keep up to cap_total rows, split pos/neg by observed ratio (or target pos_ratio).
    """
    # First pass: estimate class ratios cheaply (sample some batches)
    pos_seen = neg_seen = 0
    sample_batches = 10  # peek at ~10 batches to estimate class rate
    for i, df in enumerate(stream_batches(parquet_scaled, columns=[LABEL_COL], batch_rows=BATCH_ROWS)):
        vc = df[LABEL_COL].value_counts()
        pos_seen += int(vc.get(1, 0))
        neg_seen += int(vc.get(0, 0))
        if i + 1 >= sample_batches:
            break
        del df
    base_pos_rate = (pos_seen / (pos_seen + neg_seen)) if (pos_seen + neg_seen) else 0.001

    if pos_ratio is None:
        # keep natural scarcity, but ensure we have enough positives for SMOTE
        pos_ratio = max(base_pos_rate, 0.002)  # at least 0.2%

    cap_pos = max(2_000, int(cap_total * pos_ratio))
    cap_neg = max(10_000, cap_total - cap_pos)

    print(f"• Reservoir targets → pos: {cap_pos:,}, neg: {cap_neg:,} (base_pos_rate≈{base_pos_rate:.4%})")

    # Reservoir containers
    pos_keep = []
    neg_keep = []

    cols = num_cols + [LABEL_COL]
    for df in stream_batches(parquet_scaled, columns=cols, batch_rows=BATCH_ROWS):
        # split current batch
        pos_df = df[df[LABEL_COL] == 1]
        neg_df = df[df[LABEL_COL] == 0]

        # append until cap reached (cheap)
        if len(pos_keep) < cap_pos:
            need = cap_pos - len(pos_keep)
            if len(pos_df) > need:
                pos_keep.append(pos_df.sample(n=need, random_state=RANDOM_SEED))
            else:
                pos_keep.append(pos_df)

        if len(neg_keep) < cap_neg:
            need = cap_neg - len(neg_keep)
            if len(neg_df) > need:
                neg_keep.append(neg_df.sample(n=need, random_state=RANDOM_SEED))
            else:
                neg_keep.append(neg_df)

        del df, pos_df, neg_df
        gc.collect()

        if len(pos_keep) and sum(map(len, pos_keep)) >= cap_pos and \
           len(neg_keep) and sum(map(len, neg_keep)) >= cap_neg:
            break

    pos_cat = pd.concat(pos_keep, axis=0, ignore_index=True) if pos_keep else pd.DataFrame(columns=cols)
    neg_cat = pd.concat(neg_keep, axis=0, ignore_index=True) if neg_keep else pd.DataFrame(columns=cols)

    # If we overshot (due to last sample), trim
    if len(pos_cat) > cap_pos:
        pos_cat = pos_cat.sample(n=cap_pos, random_state=RANDOM_SEED)
    if len(neg_cat) > cap_neg:
        neg_cat = neg_cat.sample(n=cap_neg, random_state=RANDOM_SEED)

    sample_df = pd.concat([pos_cat, neg_cat], axis=0, ignore_index=True)
    # shuffle
    sample_df = sample_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"• Reservoir built → {len(sample_df):,} rows | pos={int((sample_df[LABEL_COL]==1).sum()):,}, "
          f"neg={int((sample_df[LABEL_COL]==0).sum()):,}")
    del pos_keep, neg_keep, pos_cat, neg_cat
    gc.collect()
    return sample_df


def run_smote(sample_df: pd.DataFrame, num_cols: list[str]):
    y = sample_df[LABEL_COL].astype(int).to_numpy()
    X = sample_df[num_cols].astype("float32").fillna(0.0).to_numpy(copy=False)

    pos_ct = int((y == 1).sum())
    if pos_ct < 2:
        raise RuntimeError("Not enough positive samples for SMOTE.")

    k = max(1, min(5, pos_ct - 1))
    sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
    Xb, yb = sm.fit_resample(X, y)
    print(f"• SMOTE done → {len(yb):,} rows | class dist: "
          f"{int((yb==0).sum())}/{int((yb==1).sum())} (neg/pos)")
    return Xb, yb


def process_one(fe_file: Path):
    name = fe_file.stem.replace("_fe", "")
    print(f"\n==== {name} ====")

    # 1) Read a tiny sample to detect numeric columns
    #    (Using Arrow dataset with small batch)
    sample_iter = stream_batches(fe_file, columns=None, batch_rows=50_000)
    sample_df = next(sample_iter)
    num_cols = get_num_cols(sample_df)
    print(f"• Numeric features: {len(num_cols)}")
    del sample_df; gc.collect()

    # 2) Incremental fit scaler over all rows (no full load)
    scaler = partial_fit_scaler(fe_file, num_cols)

    # 3) Transform and write scaled parquet in chunks
    scaled_path = READY / f"{name}_scaled.parquet"
    write_scaled(fe_file, scaled_path, num_cols, scaler)

    # 4) Build reservoir sample directly from scaled parquet (streaming)
    sample_df = stratified_reservoir_for_smote(scaled_path, num_cols, cap_total=SMOTE_CAP)

    # 5) Run SMOTE on the reservoir
    Xb, yb = run_smote(sample_df, num_cols)

    # 6) Save final ready dataset for ML
    out_df = pd.DataFrame(Xb, columns=num_cols)
    out_df[LABEL_COL] = yb
    out_path = READY / f"{name}_ready.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"✅ Saved → {out_path}")

    # 7) Report
    report = {
        "dataset": name,
        "num_features": len(num_cols),
        "rows_ready": int(len(out_df)),
        "pos_ready": int((out_df[LABEL_COL]==1).sum()),
        "neg_ready": int((out_df[LABEL_COL]==0).sum()),
        "smote_cap": int(SMOTE_CAP),
        "batch_rows": int(BATCH_ROWS),
    }
    with open(READY / f"{name}_prep_report.json", "w") as f:
        json.dump(report, f, indent=2)

    del out_df, Xb, yb
    gc.collect()


if __name__ == "__main__":
    for fname in ["HI_Small_fe.parquet", "LI_Small_fe.parquet"]:
        process_one(PROC / fname)
    print("\n✅ Small datasets preprocessed (streaming).")

