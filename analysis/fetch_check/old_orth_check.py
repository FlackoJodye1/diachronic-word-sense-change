"""Step 4d offline: do old spellings add sentences? Reads raw/ from fetch_check.py.

Only fetches a query whose raw CSV is missing (after the read timeout on 'Schloß' on
2026-10-28). Writes old_orth_counts.csv.
"""
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import fetch_check as fc  # noqa: E402

f = None
rows = []
for lemma, pos in fc.EXTRA:
    raws, totals = {}, {}
    for q in [lemma] + fc.OLD_ORTH[lemma]:
        p = fc.RAW / f"{fc.fname(q)}.csv"
        if not p.exists():
            f = f or fc.Fetcher(fc.load_key())
            raws[q] = f.csv(q)
            totals[q] = f.totals(q)
        else:
            raws[q] = pd.read_csv(p)
            t = fc.html.unescape(re.sub(r"<[^>]+>", " ", (fc.RAW / f"{fc.fname(q)}.html").read_text()))
            m = re.search(r"von\s+([\d\s \xa0.]+?)\s*Treffern", t)
            totals[q] = (float(re.sub(r"[^\d]", "", m.group(1))) if m else float("nan"), None)
    base = raws[lemma]
    olds = fc.OLD_ORTH[lemma]
    seen = set(zip(base.Bibl, base.Hit))
    old_raw = pd.concat([raws[o] for o in olds], ignore_index=True)
    both = pd.concat([base, old_raw], ignore_index=True)
    old_pat = "|".join(rf"(?<!\w){re.escape(o)}(?!\w)" for o in olds)
    rows.append(dict(
        lemma=lemma, old_forms="|".join(olds),
        a_base=len(base), a_old=len(old_raw),
        e_hits_base=totals[lemma][0], e_hits_old=sum(totals[o][0] for o in olds),
        rows_old_not_in_base=len(set(zip(old_raw.Bibl, old_raw.Hit)) - seen),
        share_base_rows_with_old_form=base.Hit.str.contains(old_pat, case=False, regex=True).mean(),
        b=fc.n_filtered(base, lemma, None),
        c=len(fc.apply_filters(base, lemma, fc.OPTS)),
        c_old_base_query=fc.n_filtered(base, lemma, [lemma] + olds),
        d=fc.n_filtered(both, lemma, None),
    ))
    print(rows[-1], flush=True)
out = pd.DataFrame(rows)
stats = pd.read_csv(fc.ROOT / "data/processed/diachronic/aggregated/pilot_v3_coarse_stats.csv")
out = out.merge(stats.groupby("lemma").n_contexts.sum().rename("n_ours_stats"),
                left_on="lemma", right_index=True, how="left")
out.to_csv(HERE / "old_orth_counts.csv", index=False)
print(out.to_string(index=False))
