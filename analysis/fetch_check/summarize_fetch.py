"""Summarise fetch_counts.csv: factors b/c, d/c, e/c (pooled, per POS) and compute projection."""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SEC_PER_1K = 44.0          # embedding benchmark, bench_embedding.py
N_TODAY = 318_611          # sentences in combined_contexts_v3
PAUSE = 3.5                # s per request in a re-run
SEC_PER_REQ = 8.0          # pause + download of a 5000-row page (measured, see RESULTS.md)

d = pd.read_csv(HERE / "fetch_counts.csv")
s = d[d.group == "sample"].copy()
s["e_est_sent"] = np.where(s.a_api > 0, s.e_hits * s.b_no_baseform / s.a_api, 0)


def pooled(g):
    return pd.Series({
        "n": len(g),
        "a": g.a_api.sum(), "b": g.b_no_baseform.sum(), "c": g.c_baseform.sum(),
        "d": g.d_no_baseform_old.sum(), "e_hits": g.e_hits.sum(),
        "e_est_sent": g.e_est_sent.sum(),
        "b/c": g.b_no_baseform.sum() / g.c_baseform.sum(),
        "d/c": g.d_no_baseform_old.sum() / g.c_baseform.sum(),
        "e/c": g.e_hits.sum() / g.c_baseform.sum(),
        "e_est/c": g.e_est_sent.sum() / g.c_baseform.sum(),
        "median b/c": (g.b_no_baseform / g.c_baseform).median(),
        "share no base form (mean)": g.share_raw_without_baseform.mean(),
        "limit binds": int(g.limit_binds.sum()),
    })


tab = pd.concat([s.groupby("pos").apply(pooled), pooled(s).to_frame("ALL").T])
pd.set_option("display.width", 250)
print(tab.round(2))

print("\nper lemma")
cols = ["lemma", "pos", "a_api", "b_no_baseform", "c_baseform", "n_ours_stats", "e_hits",
        "e_tokens_total", "share_raw_without_baseform"]
pl = s[cols].assign(bc=s.b_no_baseform / s.c_baseform, ec=s.e_hits / s.c_baseform)
print(pl.round(2).to_string(index=False))

o = pd.read_csv(HERE / "old_orth_counts.csv")  # from old_orth_check.py (clean raw files)
print("\nold spellings")
print(o.to_string(index=False))

# tokens (dlexDB) vs. "insgesamt" on the result page
lc = pd.read_csv(HERE / "limit_check_all.csv")
m = d.merge(lc, on="lemma", how="left")
print("\n'insgesamt' == kern.sqlite freq:", (m.e_tokens_total == m.kern_freq).sum(), "of",
      m.kern_freq.notna().sum())
print("hits / tokens (median):", (m.e_hits / m.e_tokens_total).median().round(3))

print("\nprojection (embedding only, %.0f s / 1k sentences)" % SEC_PER_1K)
for name, f in [("today (c)", 1.0), ("b/c", tab.loc["ALL", "b/c"]), ("d/c", tab.loc["ALL", "d/c"]),
                ("e_est/c", tab.loc["ALL", "e_est/c"]), ("e/c", tab.loc["ALL", "e/c"])]:
    n = N_TODAY * f
    print(f"{name:10s} factor {f:6.2f}  sentences {n:10,.0f}  {n * SEC_PER_1K / 1000 / 3600:6.1f} h")

# fetch requests for a full re-run of 502 lemmas: hits estimated from dlexDB tokens
ratio = (m.e_hits / m.e_tokens_total).median()
hits_all = lc.kern_freq.fillna(0) * ratio
pages = np.ceil(hits_all.clip(lower=1) / 5000)
print(f"\nfetch: 502 lemmas, one page each: {502 * SEC_PER_REQ / 3600:.1f} h; "
      f"all pages ({int(pages.sum())} requests): {pages.sum() * SEC_PER_REQ / 3600:.1f} h")
