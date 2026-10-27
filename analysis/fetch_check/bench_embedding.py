"""Time multilingual-e5-large on 1,000 real DWDS sentences, same settings as predict_dwds_senses.py."""
import json, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.wsd import Embedder  # noqa: E402

lines = open(ROOT / "data/processed/dwds/combined_contexts_v3.jsonl", encoding="utf-8").readlines()
random.Random(0).shuffle(lines)
texts = ["query: " + json.loads(l)["sentence"] for l in lines[:1000]]

t0 = time.perf_counter()
emb = Embedder("intfloat/multilingual-e5-large")
print(f"model load: {time.perf_counter()-t0:.1f}s, device={emb.model.device}")
emb.encode(texts[:64], batch_size=64)  # warm-up
for run in range(2):
    t0 = time.perf_counter()
    emb.encode(texts, batch_size=64)
    print(f"run {run+1}: 1000 sentences in {time.perf_counter()-t0:.1f}s")
