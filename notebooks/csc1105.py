"""Shared loading and cleaning for the CSC 1105 take-home assignment.

Every notebook imports from here so that Questions 1 to 4 all run on the
same cleaned table. If Question 1 removed rows and Question 3 were then
run on the original table, the counts in the report would contradict one
another. Keeping the cleaning in one place is what prevents that.

All four cleaning decisions have been settled on the real file. Each one
is documented below with the counts it rests on, so that the code and
Table 2.1 of the report state the same thing.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------- constants
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "synthetic_mobile_money_transaction_dataset.csv"
FIGURES = ROOT / "figures"

#: the five numeric coordinates, in the order fixed by the assignment.
#: Do not reorder: the matrix B in Question 3 is written for this order.
COLS = [
    "amount",
    "oldBalInitiator",
    "newBalInitiator",
    "oldBalRecipient",
    "newBalRecipient",
]

BALANCE_COLS = COLS[1:]

#: seed from the starter code, so every member draws the same sample
SEED = 1105

#: declared tolerance for the transfer balance check, Question 3(b)
TOL = 0.01

#: balance-change operator from the assignment, B @ x_i = (delta_I, delta_R)
B = np.array([
    [0.0, -1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, -1.0, 1.0],
])

DTYPES = {
    "step": "int32",
    "transactionType": "category",
    "amount": "float64",
    "initiator": "string",      # identifiers are names, never numbers
    "recipient": "string",
    "oldBalInitiator": "float64",
    "newBalInitiator": "float64",
    "oldBalRecipient": "float64",
    "newBalRecipient": "float64",
}


# ------------------------------------------------------------------ loading
def load_raw(path=DATA):
    """Load the CSV exactly as published, with no cleaning applied."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it from "
            "https://data.mendeley.com/datasets/zhj366m53p/2."
        )
    return pd.read_csv(path, dtype=DTYPES)


def audit(df):
    """Return the figures Question 1(b) asks for, without changing anything."""
    neg = df[BALANCE_COLS] < 0
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_by_column": df.isna().sum(),
        "rows_with_any_missing": int(df.isna().any(axis=1).sum()),
        "exact_duplicates": int(df.duplicated().sum()),
        "fraud_label_values": df["isFraud"].unique().tolist(),
        "transaction_types": df["transactionType"].value_counts(dropna=False),
        "negative_by_column": neg.sum(),
        "rows_with_any_negative": int(neg.any(axis=1).sum()),
    }


def clean(df):
    """Apply the group's cleaning decisions and return (clean_df, record).

    `record` is the row-count reconciliation the cleaning table of the
    report needs: start from the original count, subtract what is listed,
    and the result must equal the final count.
    """
    record = {"rows before cleaning": len(df)}
    out = df

    # Decision 1: exact duplicate rows. REMOVE ANY, BUT THERE ARE NONE.
    # df.duplicated().sum() is 0 on all 1,720,181 rows, so this call drops
    # nothing. It is kept as a safeguard: if the file is ever replaced by a
    # larger release, duplicates would be removed here rather than silently
    # inflating the step counts in Question 3.
    before = len(out)
    out = out.drop_duplicates(keep="first")
    record["exact duplicates removed"] = before - len(out)

    # Decision 2: missing values. NOTHING TO DO.
    # isna().sum() is 0 for every one of the ten columns, so no row needs
    # dropping and no column needs filling. Had there been missing
    # balances, filling them would have corrupted the Question 3 identity.
    record["rows dropped for missing values"] = 0

    # Decision 3: invalid fraud labels. NOTHING TO DO.
    # isFraud.unique() is {0, 1}, with no missing, negative or out-of-range
    # value, so every label is usable as an outcome.
    record["rows dropped for invalid labels"] = 0

    # Decision 4: negative balances. KEEP THEM.
    # Investigated on the full file (see negative_balance_report):
    #   48,213 rows, 2.80% of the data
    #   fraud rate among them 12.80% against 10.20% overall
    #   present in all five transaction types (0.23% to 3.76% of each)
    #   almost entirely on the initiator side: 44,402 oldBalInitiator and
    #   44,621 newBalInitiator, against 117 and 15 on the recipient side
    # Reading: an overdraft convention on the sending account, not corrupt
    # data. Removing them would also remove 6,170 fraudulent
    # transactions, 3.52% of all the fraud in the dataset, and would
    # bias Questions 2 to 4. Change this only with a reason in the report.
    record["rows dropped for negative balances"] = 0

    record["rows after cleaning"] = len(out)
    return out.reset_index(drop=True), record


def negative_balance_report(df):
    """The investigation Question 1(b) requires before any removal."""
    mask = (df[BALANCE_COLS] < 0).any(axis=1)
    rows = df[mask]
    return {
        "rows_with_negative_balance": int(mask.sum()),
        "share_of_all_rows": float(mask.mean()),
        "by_transaction_type": rows["transactionType"].value_counts(),
        "fraud_rate_among_them": float(rows["isFraud"].mean()) if len(rows) else float("nan"),
        "fraud_rate_overall": float(df["isFraud"].mean()),
    }


def load_clean(path=DATA, verbose=True):
    """Load, clean, and report the reconciliation in one call."""
    raw = load_raw(path)
    out, record = clean(raw)
    if verbose:
        for k, v in record.items():
            print(f"  {k:35s} {v:>12,}")
    return out


def sample_matrix(analysis_df, n=2000, seed=SEED):
    """Question 2: draw the sample and return (sample_df, X)."""
    sample = analysis_df.sample(n=min(n, len(analysis_df)), random_state=seed).copy()
    X = sample[COLS].to_numpy(dtype=float)
    return sample, X


def standardise(X):
    """Return (Z, mu, s) using the n - 1 divisor, as the starter code does.

    scikit-learn's StandardScaler divides by n instead, which would give
    different numbers, so ddof=1 is used throughout.
    """
    mu = X.mean(axis=0)
    s = X.std(axis=0, ddof=1)
    if np.any(s == 0):
        raise ValueError("a constant column cannot be standardised")
    return (X - mu) / s, mu, s


def two_nonzero_rows(Z):
    """Question 2(c) asks for two NONZERO standardised vectors."""
    norms = np.linalg.norm(Z, axis=1)
    idx = np.flatnonzero(norms > 0)[:2]
    if len(idx) < 2:
        raise ValueError("fewer than two nonzero standardised rows")
    return Z[idx[0]], Z[idx[1]], idx


def angle_between(u, v):
    """Return (norm_u, norm_v, distance, dot, cosine, angle_in_degrees)."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        raise ValueError("cannot take the angle with a zero vector")
    dot = float(u @ v)
    cos = dot / (nu * nv)
    return nu, nv, float(np.linalg.norm(u - v)), dot, cos, float(
        np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    )


def savefig(fig, name):
    """Save a figure into figures/ at a resolution that survives printing."""
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print("saved", path)
    return path
