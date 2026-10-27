"""Stratified sample of 20 lemmas (by POS) from the 502 diachronic lemmas. Offline."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "sample_lemmas.csv"
SEED = 20261027
N = 20

lemmas = set(pd.read_csv(ROOT / "data/processed/analysis/per_lemma_summary.csv").lemma)
pos = {}
for line in open(ROOT / "data/processed/germanet/senses_combined.jsonl", encoding="utf-8"):
    d = json.loads(line)
    if d["lemma"] in lemmas:
        pos[d["lemma"]] = d["pos"]
df = pd.DataFrame(sorted(pos.items()), columns=["lemma", "pos"])

# Proportional allocation (largest remainder)
counts = df.pos.value_counts()
quota = counts / counts.sum() * N
alloc = quota.astype(int)
for p in (quota - alloc).sort_values(ascending=False).index[: N - alloc.sum()]:
    alloc[p] += 1

sample = pd.concat(
    df[df.pos == p].sample(n=int(k), random_state=SEED) for p, k in alloc.items()
)

stats = pd.read_csv(ROOT / "data/processed/diachronic/aggregated/pilot_v3_coarse_stats.csv")
n_ours = stats.groupby("lemma").n_contexts.sum().rename("n_ours_stats")
sample = sample.merge(n_ours, left_on="lemma", right_index=True, how="left")

n_ctx = {}
for line in open(ROOT / "data/processed/dwds/combined_contexts_v3.jsonl", encoding="utf-8"):
    l = json.loads(line)["lemma"]
    n_ctx[l] = n_ctx.get(l, 0) + 1
sample["n_ours_contexts"] = sample.lemma.map(n_ctx)
sample.to_csv(OUT, index=False)
print(counts.to_dict(), alloc.to_dict())
print(sample.to_string(index=False))
