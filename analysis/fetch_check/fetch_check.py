"""Steps 3-5: how many DWDS sentences do the base-form filter, old spellings and limit=5000 cost?

Run 2026-10-28 with the personal DWDS API key (see documentation/DATA_SOURCES.md).
The key is read from .env (DWDS_API_KEY), sent only as a request header and never printed,
logged or written to disk. Honest research User-Agent, >= 3 s between requests.

Counts per lemma (same settings as the v3 fetch: kern, limit=5000, sort=random):
  a  rows returned by the CSV export
  b  after length 6-60, max_per_doc=1, near-dup 0.7 -- WITHOUT base-form filter
  c  same, WITH base-form filter (= apply_filters, our current pipeline)
  d  like b, but the raw rows of the old spelling(s) from GermaNet (oldOrthForm) are added
     before filtering. Only 4 of the 502 lemmas have one, none of them in the 20-lemma sample,
     so they are fetched as an extra set (EXTRA below). For the sample d == b.
  e  total hits in kern without limit (from the normal result page, "von N Treffern")
Raw responses go to raw/ (gitignored).

This script does not touch the pipeline: DWDSClient is not used, because it sends a browser
User-Agent and no key. The filters are the pipeline's own apply_filters.
"""
import html
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode

import pandas as pd
import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.data.dwds.fetcher as fetcher  # noqa: E402
from src.data.dwds.fetcher import FetchOptions, apply_filters  # noqa: E402
from src.data.dwds.util.context_utils import clean_context  # noqa: E402

HERE = Path(__file__).parent
RAW = HERE / "raw"
OPTS = FetchOptions(corpus="kern", limit=5000, sort="random", min_tokens=6, max_tokens=60,
                    max_per_doc=1, near_dup_threshold=0.7, target_n=0, auto_select=False)
BASE = "https://www.dwds.de/r/"
UA = "WSD-Research-TUWien/1.0 (Kontakt: Paul Schmitt; Zweck: Diachrone WSD, Kernkorpus)"
PAUSE = 3.5  # seconds before every request
# GermaNet 19.0 oldOrthForm for the 502 lemmas (searched over all nomen/verben/adj XML files)
OLD_ORTH = {"Bass": ["Baß"], "Schluss": ["Schluß"], "Schloss": ["Schloß"], "Foto": ["Photo"]}
EXTRA = [("Bass", "NOUN"), ("Schluss", "NOUN"), ("Schloss", "NOUN"), ("Foto", "NOUN")]


def fname(q: str) -> str:
    # macOS folds ß to ss in file names (Baß.csv == Bass.csv), so percent-encode.
    return quote(q, safe="")


class Blocked(RuntimeError):
    pass


def load_key() -> str:
    env = {k.strip(): (v or "").strip() for k, v in dotenv_values(ROOT / ".env").items()}
    key = env.get("DWDS_API_KEY")
    if not key:
        raise SystemExit("DWDS_API_KEY missing in .env")
    return key


class Fetcher:
    def __init__(self, key: str):
        self.s = requests.Session()
        # Key in headers only, never in the URL (URLs end up in our log).
        self.s.headers.update({"User-Agent": UA, "Authorization": f"Bearer {key}", "X-API-Key": key})
        self.key = key
        self.log = open(RAW / "requests.log", "a", encoding="utf-8")
        self.n = 0

    def get(self, params: dict, expect: str) -> requests.Response:
        url = BASE + "?" + urlencode(params)
        for attempt in range(5):
            time.sleep(PAUSE)
            try:
                r = self.s.get(url, timeout=180)
            except (requests.ConnectionError, requests.Timeout) as e:
                wait = 30 * 2 ** attempt
                print(f"  {type(e).__name__}, waiting {wait} s", flush=True)
                self.log.write(f"{datetime.now(timezone.utc).isoformat()}\tERR\t{type(e).__name__}\t0\t{url}\n")
                time.sleep(wait)
                continue
            self.n += 1
            ct = r.headers.get("Content-Type", "")
            self.log.write(f"{datetime.now(timezone.utc).isoformat()}\t{r.status_code}\t{ct}\t"
                           f"{len(r.content)}\t{url}\n")
            self.log.flush()
            if r.status_code in (429, 503):
                wait = float(r.headers.get("Retry-After", 30 * 2 ** attempt))
                print(f"  {r.status_code}, waiting {wait:.0f} s", flush=True)
                time.sleep(wait)
                continue
            if r.status_code in (401, 403) or "als Bot identifiziert" in r.text:
                raise Blocked(f"HTTP {r.status_code} ({ct}) for {url}")
            r.raise_for_status()
            if expect not in ct:
                raise Blocked(f"expected {expect}, got {ct} for {url}")
            assert self.key not in r.text
            return r
        raise Blocked(f"gave up after retries: {url}")

    def csv(self, q: str) -> pd.DataFrame:
        r = self.get(dict(q=q, view="csv", corpus=OPTS.corpus, limit=OPTS.limit, sort=OPTS.sort),
                     expect="text/")
        (RAW / f"{fname(q)}.csv").write_bytes(r.content)
        if not r.text.strip():
            return pd.DataFrame()
        return pd.read_csv(io.StringIO(r.text))

    def totals(self, q: str) -> tuple[float, float]:
        """'1-1 von 32 698 Treffern ( 40503 insgesamt )' -> (32698, 40503)."""
        r = self.get(dict(q=q, corpus=OPTS.corpus, limit=1), expect="text/html")
        (RAW / f"{fname(q)}.html").write_bytes(r.content)
        t = html.unescape(re.sub(r"<[^>]+>", " ", r.text))
        num = lambda s: float(re.sub(r"[^\d]", "", s))
        m = re.search(r"von\s+([\d\s \xa0.]+?)\s*Treffern\s*\(\s*([\d\s \xa0.]+?)\s*insgesamt", t)
        if m:
            return num(m.group(1)), num(m.group(2))
        m = re.search(r"von\s+([\d\s \xa0.]+?)\s*Treffern", t)
        if m:
            return num(m.group(1)), float("nan")
        return (0.0, 0.0) if re.search(r"[Kk]eine Treffer", t) else (float("nan"), float("nan"))


def n_filtered(raw: pd.DataFrame, lemma: str, forms: list[str] | None) -> int:
    """apply_filters with the base-form check replaced.

    forms=None: no base-form check (b, d). Otherwise: exactly one occurrence of any of the
    forms (c with old spellings)."""
    if raw.empty:
        return 0
    orig = fetcher.target_matches_token
    if forms is None:
        fetcher.target_matches_token = lambda text, word: True
    else:
        pat = "|".join(re.escape(f) for f in forms)
        fetcher.target_matches_token = lambda text, word: len(
            re.findall(rf"(?<!\w)(?:{pat})(?!\w)", text, flags=re.I)) == 1
    try:
        return len(apply_filters(raw, lemma, OPTS))
    finally:
        fetcher.target_matches_token = orig


def share_without_baseform(raw: pd.DataFrame, lemma: str) -> float:
    if raw.empty:
        return float("nan")
    pat = rf"(?<!\w){re.escape(lemma)}(?!\w)"
    return raw["Hit"].apply(clean_context).apply(
        lambda t: len(re.findall(pat, t, flags=re.I)) == 0).mean()


def measure(f: Fetcher, lemma: str, pos: str, group: str) -> dict:
    raw = f.csv(lemma)
    hits, total = f.totals(lemma)
    a = len(raw)
    row = dict(group=group, lemma=lemma, pos=pos, a_api=a,
               b_no_baseform=n_filtered(raw, lemma, None),
               c_baseform=len(apply_filters(raw, lemma, OPTS)) if a else 0,
               share_raw_without_baseform=share_without_baseform(raw, lemma),
               limit_binds=a >= OPTS.limit, e_hits=hits, e_tokens_total=total,
               columns="|".join(raw.columns))
    olds = OLD_ORTH.get(lemma, [])
    raws = [raw]
    a_old = hits_old = 0.0
    for old in olds:
        r_old = f.csv(old)
        h_old, _ = f.totals(old)
        a_old += len(r_old)
        hits_old += h_old
        raws.append(r_old)
    both = pd.concat(raws, ignore_index=True) if olds else raw
    # rows of the old-spelling query that the base query did not return (same Bibl + Hit)
    seen = set(zip(raw.Bibl, raw.Hit)) if a else set()
    new_old = sum(len(set(zip(r.Bibl, r.Hit)) - seen) for r in raws[1:])
    base_has_old = (raw.Hit.str.contains("|".join(rf"(?<!\w){re.escape(o)}(?!\w)" for o in olds),
                                         case=False, regex=True).mean() if olds and a else 0.0)
    row.update(old_forms="|".join(olds), a_old=a_old, e_hits_old=hits_old,
               rows_new_from_old_query=new_old, share_base_rows_with_old_form=base_has_old,
               c_baseform_old_base_query=n_filtered(raw, lemma, [lemma] + olds) if olds else row["c_baseform"],
               d_no_baseform_old=n_filtered(both, lemma, None) if olds else row["b_no_baseform"],
               c_baseform_old=n_filtered(both, lemma, [lemma] + olds) if olds else row["c_baseform"])
    return row


def main():
    only_old = "--only-old-orth" in sys.argv  # re-run just the EXTRA set, keep sample rows
    RAW.mkdir(exist_ok=True)
    (RAW / ".gitignore").write_text("*\n")
    f = Fetcher(load_key())
    sample = pd.read_csv(HERE / "sample_lemmas.csv")
    todo = ([] if only_old else [(l, p, "sample") for l, p in zip(sample.lemma, sample.pos)]) + \
           [(l, p, "old_orth") for l, p in EXTRA]
    rows = []
    t0 = time.time()
    for lemma, pos, group in todo:
        try:
            rows.append(measure(f, lemma, pos, group))
        except Blocked as e:
            print("STOP:", e, flush=True)
            break
        r = rows[-1]
        print({k: r[k] for k in ("lemma", "a_api", "b_no_baseform", "c_baseform",
                                 "d_no_baseform_old", "e_hits", "e_tokens_total")}, flush=True)
    out = pd.DataFrame(rows)
    stats = pd.read_csv(ROOT / "data/processed/diachronic/aggregated/pilot_v3_coarse_stats.csv")
    ours = stats.groupby("lemma").n_contexts.sum().rename("n_ours_stats")
    out = out.merge(ours, left_on="lemma", right_index=True, how="left")
    if only_old:
        prev = pd.read_csv(HERE / "fetch_counts.csv")
        out = pd.concat([prev[prev.group != "old_orth"], out], ignore_index=True)
    out.to_csv(HERE / "fetch_counts.csv", index=False)
    meta = dict(date=datetime.now(timezone.utc).isoformat(), user_agent=UA, pause_s=PAUSE,
                n_requests=f.n, runtime_s=round(time.time() - t0, 1), options=vars(OPTS),
                auth="DWDS API key as headers Authorization: Bearer + X-API-Key")
    meta_path = HERE / ("fetch_meta_old_orth.json" if only_old else "fetch_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(meta)


if __name__ == "__main__":
    main()
