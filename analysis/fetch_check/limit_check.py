"""Step 4 offline: kern lemma frequencies from data/raw/dlexdb/kern.sqlite vs. limit=5000.

Corpus frequency counts tokens, not sentences or documents. A lemma with f > 5000 has more
than 5000 hits in kern, so a single-page fetch with limit=5000 is a sample, not everything.
"""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
LIMIT = 5000

con = sqlite3.connect(ROOT / "data/raw/dlexdb/kern.sqlite")
freq = pd.read_sql("SELECT l AS lemma, SUM(f) AS kern_freq FROM lex GROUP BY l", con)
freq = freq.groupby("lemma", as_index=False).kern_freq.sum()

all_lemmas = pd.read_csv(ROOT / "data/processed/analysis/per_lemma_summary.csv")[["lemma"]]
df = all_lemmas.merge(freq, on="lemma", how="left")
df["limit_binds"] = df.kern_freq > LIMIT

sample = pd.read_csv(HERE / "sample_lemmas.csv").merge(df, on="lemma", how="left")
sample.to_csv(HERE / "limit_check_sample.csv", index=False)
df.to_csv(HERE / "limit_check_all.csv", index=False)

print(sample[["lemma", "pos", "kern_freq", "n_ours_stats", "limit_binds"]].to_string(index=False))
print(f"\nAll 502: missing freq {df.kern_freq.isna().sum()}, "
      f"f > {LIMIT}: {df.limit_binds.sum()} ({df.limit_binds.mean():.0%}), "
      f"median f {df.kern_freq.median():.0f}")
