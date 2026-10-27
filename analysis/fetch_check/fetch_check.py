"""Steps 3-5: count DWDS hits with and without the base-form filter for the 20 sampled lemmas.

NOT RUN YET: DWDS blocks non-browser user agents on /r/ (robots.txt: Disallow /r;
TDM policy requires consent). Run only once access is cleared with DWDS.

Counts per lemma:
  a  rows returned by the API (limit=5000, sort=random, corpus=kern)
  b  after length 6-60, max_per_doc=1, near-dup 0.7 -- WITHOUT base-form filter
  c  same, WITH base-form filter (= apply_filters, our current pipeline)
Plus: share of raw rows that contain no exact base form (= API returns inflected forms),
and whether a == limit (limit binds).
"""
import re, sys, time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.dwds.client import DWDSClient  # noqa: E402
from src.data.dwds.fetcher import FetchOptions, apply_filters  # noqa: E402
from src.data.dwds.util.context_utils import clean_context, count_words, target_matches_token  # noqa: E402

HERE = Path(__file__).parent
OPTS = FetchOptions(corpus="kern", limit=5000, sort="random", min_tokens=6, max_tokens=60,
                    max_per_doc=1, near_dup_threshold=0.7, target_n=0, auto_select=False)
PAUSE = 5.0  # seconds between lemmas, on top of client delay


def n_without_baseform_filter(raw: pd.DataFrame, lemma: str) -> int:
    # Same as apply_filters, but the target filter always passes.
    import src.data.dwds.fetcher as f
    orig = f.target_matches_token
    f.target_matches_token = lambda text, word: True
    try:
        return len(apply_filters(raw, lemma, OPTS))
    finally:
        f.target_matches_token = orig


def main():
    sample = pd.read_csv(HERE / "sample_lemmas.csv")
    client = DWDSClient(delay=1.0)
    rows = []
    for lemma, pos in zip(sample.lemma, sample.pos):
        raw = client.corpus_csv(lemma, corpus=OPTS.corpus, limit=OPTS.limit, sort=OPTS.sort)
        raw = pd.DataFrame() if raw is None else raw
        a = len(raw)
        if a:
            hits = raw["Hit"].apply(clean_context)
            pat = rf"(?<!\w){re.escape(lemma)}(?!\w)"
            share_no_base = (hits.apply(lambda t: len(re.findall(pat, t, flags=re.I)) == 0)).mean()
            b = n_without_baseform_filter(raw, lemma)
            c = len(apply_filters(raw, lemma, OPTS))
        else:
            share_no_base, b, c = float("nan"), 0, 0
        rows.append(dict(lemma=lemma, pos=pos, a_api=a, b_no_baseform=b, c_baseform=c,
                         share_raw_without_baseform=share_no_base, limit_binds=a >= OPTS.limit,
                         columns="|".join(raw.columns)))
        print(rows[-1], flush=True)
        raw.to_csv(HERE / "raw" / f"{lemma}.csv", index=False) if a else None
        time.sleep(PAUSE)
    out = pd.DataFrame(rows).merge(sample[["lemma", "n_ours_stats"]], on="lemma")
    out.to_csv(HERE / "fetch_counts.csv", index=False)
    print(out.groupby("pos")[["b_no_baseform", "c_baseform"]].sum().assign(
        factor=lambda d: d.b_no_baseform / d.c_baseform))
    print("total factor", out.b_no_baseform.sum() / out.c_baseform.sum())


if __name__ == "__main__":
    (HERE / "raw").mkdir(exist_ok=True)
    main()
