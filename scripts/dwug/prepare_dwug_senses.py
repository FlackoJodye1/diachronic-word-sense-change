#!/usr/bin/env python3
"""
Prepare GermaNet senses for the DWUG DE target words with exactly the paper's settings.

Selects the DWUG targets with >= 2 GermaNet senses that are not yet in the 502-lemma set
(sense_clusters_reviewed.json), resolving pre-1996 spellings (Engpaß -> Engpass) via
GermaNet's oldOrthForm. Then, per sense:
  1. baseline text  — same template as senses_base.csv (BaselineTextBuilder)
  2. llm_definition + 5 llm_examples — enrich_senses_with_llm.py prompts
  3. core_gloss — sibling-aware prompt as in generate_expanded_senses.py
gpt-4o-mini, temperature 0.7. Output has the same format as senses_expanded.jsonl.

Only the DWUG target *list* is read (directory names). No DWUG uses, clusters or gold
scores are touched, so the manual cluster review stays independent of the gold.

Writes only to data/processed/dwug/ — existing project data is never modified.

Usage:
    python scripts/dwug/prepare_dwug_senses.py --verify-baseline
    python scripts/dwug/prepare_dwug_senses.py --dry-run
    python scripts/dwug/prepare_dwug_senses.py --yes [--resume]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts" / "data"))

from src.data.germanet.parsing import XMLParser
from src.data.germanet.transforms import BaselineTextBuilder
from src.data.llm import LLMClient, SenseTextGenerator
from generate_expanded_senses import build_sibling_map, clean_gloss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GN_XML_DIR = BASE_DIR / "data" / "raw" / "GN_V190" / "GN_V190_XML"
GN_BASE_DIR = BASE_DIR / "data" / "processed" / "germanet_base"
SENSES_BASE_FILE = BASE_DIR / "data" / "processed" / "germanet" / "senses_base.csv"
REVIEWED_FILE = BASE_DIR / "data" / "processed" / "germanet" / "sense_clusters_reviewed.json"
DWUG_DATA_DIR = BASE_DIR / "data" / "external" / "dwug_de" / "dwug_de" / "data"

OUT_DIR = BASE_DIR / "data" / "processed" / "dwug"
LEMMA_MAP_FILE = OUT_DIR / "lemma_map.csv"
BASE_OUT_FILE = OUT_DIR / "senses_base_dwug.csv"
SENSES_OUT_FILE = OUT_DIR / "senses_dwug.jsonl"
REVIEW_LEMMAS_FILE = OUT_DIR / "review_lemmas.txt"

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.7


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def load_builder() -> BaselineTextBuilder:
    return BaselineTextBuilder(
        read_csv(GN_BASE_DIR / "lexical_layer" / "lemma_senses.csv"),
        read_csv(GN_BASE_DIR / "sysnets" / "sense_texts.csv"),
        read_csv(GN_BASE_DIR / "semantic_layer" / "relations.csv"),
    )


def verify_baseline(builder: BaselineTextBuilder) -> bool:
    """Rebuild every row of senses_base.csv and compare column by column."""
    expected = read_csv(SENSES_BASE_FILE)
    rebuilt = builder.build(expected["lemma"].unique().tolist())
    ok = len(rebuilt) == len(expected)
    logger.info(f"Rows: expected {len(expected)}, rebuilt {len(rebuilt)}")
    exp = expected.set_index(["lemma", "synset_id"])
    reb = rebuilt.set_index(["lemma", "synset_id"]).reindex(exp.index)
    for col in ["pos", "class", "paraphrase", "text_for_embedding", "text_source",
                "sense_frequency", "embedding_word_count", "has_examples"]:
        e, r = exp[col].astype(str), reb[col].astype(str)
        n_bad = int((e != r).sum())
        logger.info(f"  {col:22s} mismatches: {n_bad}")
        ok &= n_bad == 0
    logger.info("Baseline reconstruction " + ("VERIFIED" if ok else "FAILED"))
    return ok


def resolve_targets(builder: BaselineTextBuilder) -> pd.DataFrame:
    """Map DWUG target lemmas to GermaNet orthForms and decide which need preparation."""
    targets = sorted(p.name for p in DWUG_DATA_DIR.iterdir() if p.is_dir())
    _, lexical_units = XMLParser().parse_xml_files(GN_XML_DIR)
    old_to_new: dict[str, set[str]] = {}
    for lu in lexical_units:
        for old in lu["old_orth_forms"]:
            old_to_new.setdefault(old, set()).add(lu["lemma"])

    sense_counts = builder.lemma_senses.groupby("lemma").size()
    with open(REVIEWED_FILE, encoding="utf-8") as f:
        existing = set(json.load(f))

    rows = []
    for t in targets:
        if t in sense_counts.index:
            gn_lemma, via = t, "orthForm"
        elif len(old_to_new.get(t, ())) == 1:
            gn_lemma, via = next(iter(old_to_new[t])), "oldOrthForm"
        else:
            gn_lemma, via = None, "not found"
        n = int(sense_counts.get(gn_lemma, 0)) if gn_lemma else 0
        in_502 = gn_lemma in existing
        if n < 2:
            status = "single_sense"
        elif in_502:
            status = "existing_502"
        else:
            status = "prepare"
        rows.append({"dwug_lemma": t, "germanet_lemma": gn_lemma, "resolved_via": via,
                     "n_germanet_senses": n, "in_existing_502": in_502, "status": status})
    return pd.DataFrame(rows)


def load_done(path: Path) -> dict[str, dict]:
    done = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("core_gloss") and rec.get("llm_definition") and len(rec.get("llm_examples", [])) == 5:
                    done[rec["synset_id"]] = rec
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify-baseline", action="store_true",
                        help="Only check that the baseline template reproduces senses_base.csv")
    parser.add_argument("--dry-run", action="store_true", help="Resolve targets and build baselines, no API calls")
    parser.add_argument("--resume", action="store_true", help="Keep complete senses already in the output file")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    builder = load_builder()
    if not verify_baseline(builder):
        sys.exit(1)
    if args.verify_baseline:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lemma_map = resolve_targets(builder)
    lemma_map.to_csv(LEMMA_MAP_FILE, index=False)
    logger.info(f"Targets: {len(lemma_map)} — " + ", ".join(
        f"{k}: {v}" for k, v in lemma_map["status"].value_counts().items()))
    for r in lemma_map[lemma_map["resolved_via"] != "orthForm"].itertuples():
        logger.info(f"  {r.dwug_lemma} -> {r.germanet_lemma} ({r.resolved_via})")

    lemmas = sorted(lemma_map.loc[lemma_map["status"] == "prepare", "germanet_lemma"])
    base_df = builder.build(lemmas)
    base_df.to_csv(BASE_OUT_FILE, index=False)
    REVIEW_LEMMAS_FILE.write_text("\n".join(lemmas) + "\n", encoding="utf-8")
    logger.info(f"Prepare {len(lemmas)} lemmas, {len(base_df)} senses -> {BASE_OUT_FILE.relative_to(BASE_DIR)}")

    senses = [{"lemma": r["lemma"], "pos": r["pos"], "synset_id": r["synset_id"], "class": r["class"],
               "baseline": r["text_for_embedding"]} for r in base_df.to_dict("records")]
    if args.dry_run:
        logger.info("Dry run — no API calls, no sense file written.")
        return

    done = load_done(SENSES_OUT_FILE) if args.resume else {}
    if SENSES_OUT_FILE.exists() and not args.resume:
        logger.error(f"{SENSES_OUT_FILE} exists; pass --resume or remove it first.")
        sys.exit(1)
    todo = [s for s in senses if s["synset_id"] not in done]
    logger.info(f"Senses to generate: {len(todo)} (already done: {len(done)}); "
                f"~{3 * len(todo)} calls to {MODEL}")
    if todo and not args.yes:
        if input("Proceed? [y/N]: ").lower() != "y":
            sys.exit(0)

    generator = SenseTextGenerator(LLMClient(provider="openai", model=MODEL, temperature=TEMPERATURE))
    sibling_map = build_sibling_map(senses)
    for i, s in enumerate(todo, 1):
        logger.info(f"[{i}/{len(todo)}] {s['lemma']} ({s['synset_id']})")
        definition = generator.generate_definition(s["lemma"], s["pos"], s["baseline"])
        examples = generator.generate_examples(s["lemma"], s["pos"], s["baseline"], n=5)
        examples += [""] * (5 - len(examples))
        gloss = generator.generate_core_gloss(
            lemma=s["lemma"], pos=s["pos"], sense_description=s["baseline"],
            sibling_descriptions=sibling_map.get(s["synset_id"]) or None,
        )
        done[s["synset_id"]] = {**s, "core_gloss": clean_gloss(gloss),
                                "llm_definition": definition, "llm_examples": examples}
        # Rewrite after every sense so an interrupted run can --resume
        with open(SENSES_OUT_FILE, "w", encoding="utf-8") as f:
            for sense in senses:
                if sense["synset_id"] in done:
                    f.write(json.dumps(done[sense["synset_id"]], ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(done)} senses -> {SENSES_OUT_FILE.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
