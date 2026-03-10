#!/usr/bin/env python3
"""
Enrich GermaNet senses with LLM-generated text representations.

This is part of the core data preparation pipeline (Phase 1).
Creates enhanced sense files in JSONL format with both baseline and LLM-generated text.

Pipeline position:
    1. process_germanet.py → senses_base.csv (baseline)
    2. enrich_senses_with_llm.py → senses_full.jsonl / senses_pilot.jsonl (THIS SCRIPT)
    3. fetch_dwds_contexts.py → dwds contexts

The enhanced JSONL file is the canonical source for all experiments.

Usage:
    # Generate for all words in the sense file
    python scripts/data/enrich_senses_with_llm.py --provider openai

    # Generate only for specific words (e.g., pilot set)
    python scripts/data/enrich_senses_with_llm.py \
        --word-list data/raw/target_words/pilot_words_diachronic.txt \
        --provider openai

    # Dry run to test
    python scripts/data/enrich_senses_with_llm.py --dry-run

    # Resume from checkpoint
    python scripts/data/enrich_senses_with_llm.py --resume checkpoint_100.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.llm import LLMClient, SenseTextGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
GERMANET_DIR = DATA_DIR / "processed" / "germanet"

# Input/output files
INPUT_FILE = GERMANET_DIR / "senses_base.csv"
OUTPUT_FILE = GERMANET_DIR / "senses_full.jsonl"
CHECKPOINT_DIR = GERMANET_DIR / "llm_checkpoints"


def load_senses(
    input_file: Path,
    word_list_file: Optional[Path] = None
) -> pd.DataFrame:
    """Load senses to enrich.

    Args:
        input_file: Path to baseline sense CSV
        word_list_file: Optional file with list of words to filter to

    Returns:
        DataFrame with senses to process
    """
    logger.info(f"Loading senses from {input_file}...")
    senses_df = pd.read_csv(input_file)
    logger.info(f"  Loaded {len(senses_df)} total senses")

    if word_list_file:
        logger.info(f"Filtering to words from {word_list_file}...")
        with open(word_list_file, 'r') as f:
            target_words = {line.strip() for line in f if line.strip()}

        senses_df = senses_df[senses_df['lemma'].isin(target_words)].copy()
        logger.info(f"  Filtered to {len(senses_df)} senses for {len(target_words)} words")

    # Show stats
    word_counts = senses_df.groupby('lemma').size()
    logger.info(f"  Words: {len(word_counts)}")
    logger.info(f"  Senses per word: min={word_counts.min()}, "
                f"max={word_counts.max()}, mean={word_counts.mean():.1f}")

    return senses_df


def load_checkpoint(checkpoint_file: Path) -> pd.DataFrame:
    """Load progress from a checkpoint file."""
    logger.info(f"Resuming from checkpoint: {checkpoint_file}")
    checkpoint_df = pd.read_csv(checkpoint_file)
    logger.info(f"  Loaded {len(checkpoint_df)} already-processed senses")
    return checkpoint_df


def save_as_jsonl(df: pd.DataFrame, output_path: Path):
    """Save DataFrame in JSONL format with cleaner structure.

    Args:
        df: DataFrame with enhanced senses
        output_path: Path to output JSONL file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # Collect examples into an array
            examples = [
                row['llm_example_1'],
                row['llm_example_2'],
                row['llm_example_3'],
                row['llm_example_4'],
                row['llm_example_5']
            ]

            # Create JSON object
            record = {
                'lemma': row['lemma'],
                'pos': row['pos'],
                'synset_id': row['synset_id'],
                'class': row['class'],
                'baseline': row['text_for_embedding'],
                'llm_definition': row['llm_definition'],
                'llm_examples': examples
            }

            # Write as single line JSON
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def enrich_senses(
    senses_df: pd.DataFrame,
    generator: Optional[SenseTextGenerator],
    dry_run: bool = False,
    checkpoint_freq: int = 50,
    checkpoint_dir: Path = CHECKPOINT_DIR
) -> pd.DataFrame:
    """Add LLM-generated text columns to sense DataFrame.

    Args:
        senses_df: Input DataFrame with baseline senses
        generator: LLM text generator (None for dry run)
        dry_run: If True, skip API calls
        checkpoint_freq: Save checkpoint every N senses
        checkpoint_dir: Where to save checkpoints

    Returns:
        Enhanced DataFrame with new columns:
            - llm_definition: LLM-generated definition
            - llm_example_1 through llm_example_5: Example sentences
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Add new columns (only if they don't exist)
    result_df = senses_df.copy()
    if 'llm_definition' not in result_df.columns:
        result_df['llm_definition'] = None
    for i in range(1, 6):
        if f'llm_example_{i}' not in result_df.columns:
            result_df[f'llm_example_{i}'] = None

    total = len(result_df)

    # Use enumerate to get sequential counter (not DataFrame index)
    for counter, (idx, row) in enumerate(result_df.iterrows(), start=1):
        lemma = row['lemma']
        pos = row['pos']
        synset_id = row['synset_id']
        sense_description = row['text_for_embedding']

        # Skip if already has LLM content (for resuming partial runs)
        if pd.notna(result_df.at[idx, 'llm_definition']) and result_df.at[idx, 'llm_definition']:
            logger.info(f"[{counter}/{total}] Skipping {lemma} ({synset_id}) - already processed")
            continue

        logger.info(f"[{counter}/{total}] Processing: {lemma} ({synset_id})")

        if dry_run:
            result_df.at[idx, 'llm_definition'] = f"[DRY RUN] Definition for {lemma}"
            for i in range(1, 6):
                result_df.at[idx, f'llm_example_{i}'] = f"[DRY RUN] Example {i} for {lemma}"
        else:
            try:
                variants = generator.generate_all(
                    lemma=lemma,
                    pos=pos,
                    sense_description=sense_description
                )

                result_df.at[idx, 'llm_definition'] = variants['definition']

                examples = variants['examples_5']
                for i in range(1, 6):
                    if i-1 < len(examples):
                        result_df.at[idx, f'llm_example_{i}'] = examples[i-1]
                    else:
                        result_df.at[idx, f'llm_example_{i}'] = ""

                logger.info(f"  Generated definition ({len(variants['definition'])} chars) "
                           f"and {len(examples)} examples")

            except Exception as e:
                logger.error(f"  Failed: {e}")
                result_df.at[idx, 'llm_definition'] = f"[ERROR] {e}"
                for i in range(1, 6):
                    result_df.at[idx, f'llm_example_{i}'] = ""

        # Save checkpoint (use counter for consistent checkpoint naming)
        if counter % checkpoint_freq == 0:
            checkpoint_file = checkpoint_dir / f"checkpoint_{counter}.csv"
            result_df.to_csv(checkpoint_file, index=False)
            logger.info(f"  Checkpoint saved to {checkpoint_file}")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Enrich GermaNet senses with LLM-generated text"
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help=f'Input sense file (default: {INPUT_FILE})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help=f'Output JSONL file (default: {OUTPUT_FILE})'
    )
    parser.add_argument(
        '--word-list',
        type=str,
        default=None,
        help='Optional: file with list of words to process (one per line)'
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='openai',
        choices=['openai', 'claude'],
        help='LLM provider to use'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name (default: gpt-4o-mini for openai, claude-3-5-sonnet for claude)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Generation temperature'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test run without API calls'
    )
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=50,
        help='Save checkpoint every N senses'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint file'
    )
    parser.add_argument(
        '--yes',
        action='store_true',
        help='Skip confirmation prompt'
    )

    args = parser.parse_args()

    input_file = Path(args.input) if args.input else INPUT_FILE
    output_file = Path(args.output) if args.output else OUTPUT_FILE
    word_list_file = Path(args.word_list) if args.word_list else None

    logger.info("=" * 70)
    logger.info("ENRICH GERMANET SENSES WITH LLM TEXT")
    logger.info("=" * 70)
    logger.info(f"Input:  {input_file}")
    logger.info(f"Output: {output_file}")
    if word_list_file:
        logger.info(f"Filter: {word_list_file}")
    logger.info("=" * 70)

    # Load or resume
    if args.resume:
        senses_df = load_checkpoint(Path(args.resume))
        logger.info("Resuming from checkpoint")

        # Count what's already done
        already_done = senses_df['llm_definition'].notna().sum()
        remaining = len(senses_df) - already_done
        logger.info(f"  Already processed: {already_done}/{len(senses_df)}")
        logger.info(f"  Remaining: {remaining}")
    else:
        # Load senses
        senses_df = load_senses(input_file, word_list_file)

    # Initialize LLM client
    if not args.dry_run:
        logger.info(f"\nInitializing {args.provider} client...")
        try:
            client = LLMClient(
                provider=args.provider,
                model=args.model,
                temperature=args.temperature
            )
        except ValueError as e:
            env_var = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
            logger.error(str(e))
            logger.error(f"Set the environment variable: export {env_var}='your-api-key-here'")
            logger.error(f"Or add {env_var}=your-api-key-here to a .env file in the project root")
            sys.exit(1)

        generator = SenseTextGenerator(client)

        # Estimate cost (only for remaining senses if resuming)
        if args.resume:
            n_senses = remaining
        else:
            n_senses = len(senses_df)

        logger.info(f"\n{'=' * 70}")
        logger.info("COST ESTIMATE")
        logger.info(f"{'=' * 70}")
        logger.info(f"Senses to process: {n_senses}")
        logger.info(f"API calls per sense: 2 (definition + 5 examples)")
        logger.info(f"Total API calls: ~{n_senses * 2}")

        if args.provider == 'openai':
            cost_per_call = 0.001  # gpt-4o-mini rough estimate
            total_cost = n_senses * 2 * cost_per_call
            logger.info(f"Estimated cost: ~${total_cost:.2f} (gpt-4o-mini)")

        logger.info(f"{'=' * 70}")

        if not args.yes:
            confirm = input("\nProceed with LLM generation? [y/N]: ")
            if confirm.lower() != 'y':
                logger.info("Cancelled by user")
                sys.exit(0)
        else:
            logger.info("\nProceeding with LLM generation (--yes flag set)...")
    else:
        generator = None
        logger.info("\nDRY RUN - no API calls will be made")

    # Enrich senses
    logger.info("\nEnriching senses...")
    result_df = enrich_senses(
        senses_df,
        generator,
        dry_run=args.dry_run,
        checkpoint_freq=args.checkpoint_freq
    )

    # Save final result as JSONL
    logger.info(f"\nSaving enhanced senses...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"  → JSONL: {output_file}")
    save_as_jsonl(result_df, output_file)

    logger.info(f"\n{'=' * 70}")
    logger.info("ENRICHMENT COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"Input senses:  {len(result_df)}")

    # Count successes
    if not args.dry_run:
        successful = result_df['llm_definition'].notna().sum()
        errors = result_df['llm_definition'].str.startswith('[ERROR]', na=False).sum()
        logger.info(f"Successful:    {successful} ({successful/len(result_df)*100:.1f}%)")
        logger.info(f"Errors:        {errors}")

    logger.info(f"Output file:")
    logger.info(f"  JSONL: {output_file}")
    logger.info(f"{'=' * 70}")

    # Show sample
    logger.info("\nSample of enhanced data:")
    sample = result_df[['lemma', 'synset_id', 'llm_definition']].head(3)
    for _, row in sample.iterrows():
        logger.info(f"  {row['lemma']} ({row['synset_id']})")
        logger.info(f"    {row['llm_definition'][:100]}...")


if __name__ == "__main__":
    main()
