#!/usr/bin/env python3
"""
Generate LLM-based text variants for sense representations.

Creates definitions and example sentences for eval word senses
to compare against the baseline GermaNet template approach.

Usage:
    python scripts/data_preparation/generate_llm_sense_texts.py --provider openai
    python scripts/data_preparation/generate_llm_sense_texts.py --provider claude
    python scripts/data_preparation/generate_llm_sense_texts.py --provider openai --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.llm import LLMClient, SenseTextGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
EVAL_DIR = DATA_DIR / "evaluation" / "quick_eval"
GERMANET_DIR = DATA_DIR / "processed" / "germanet"


def load_eval_senses() -> pd.DataFrame:
    """Load senses for the 10 eval words."""
    words_df = pd.read_csv(EVAL_DIR / "words.csv")
    eval_lemmas = set(words_df['lemma'].tolist())
    logger.info(f"Eval words: {eval_lemmas}")

    senses_df = pd.read_csv(GERMANET_DIR / "wsd_senses.csv")
    eval_senses = senses_df[senses_df['lemma'].isin(eval_lemmas)].copy()
    logger.info(f"Found {len(eval_senses)} senses for {len(eval_lemmas)} eval words")

    return eval_senses


def generate_variants(
    senses_df: pd.DataFrame,
    generator: SenseTextGenerator,
    dry_run: bool = False
) -> pd.DataFrame:
    """Generate LLM text variants for all senses."""
    results = []

    for _, row in senses_df.iterrows():
        lemma = row['lemma']
        pos = row['pos']
        synset_id = row['synset_id']
        semantic_class = row['class']
        sense_description = row['text_for_embedding']

        logger.info(f"Processing: {lemma} ({synset_id})")

        result = {
            'lemma': lemma,
            'pos': pos,
            'synset_id': synset_id,
            'class': semantic_class,
            'text_baseline': sense_description,
        }

        if dry_run:
            result['text_llm_definition'] = f"[DRY RUN] Definition for {lemma}"
            for i in range(1, 6):
                result[f'text_llm_example_{i}'] = f"[DRY RUN] Example {i} for {lemma}"
        else:
            try:
                variants = generator.generate_all_variants(
                    lemma=lemma,
                    pos=pos,
                    sense_description=sense_description
                )
                result['text_llm_definition'] = variants['definition']
                # Store each example in its own column
                examples = variants['examples_5']
                for i in range(5):
                    result[f'text_llm_example_{i+1}'] = examples[i] if i < len(examples) else ""
            except Exception as e:
                logger.error(f"Failed to generate for {lemma} ({synset_id}): {e}")
                result['text_llm_definition'] = f"[ERROR] {e}"
                for i in range(1, 6):
                    result[f'text_llm_example_{i}'] = f"[ERROR] {e}"

        results.append(result)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Generate LLM-based sense text variants")
    parser.add_argument('--provider', type=str, default='openai', choices=['openai', 'claude'])
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    logger.info("Loading eval senses...")
    senses_df = load_eval_senses()
    logger.info(f"Loaded {len(senses_df)} senses")

    if not args.dry_run:
        logger.info(f"Initializing {args.provider} client...")
        try:
            client = LLMClient(
                provider=args.provider,
                model=args.model,
                temperature=args.temperature
            )
        except ValueError as e:
            # Missing API key -> exit with clear message
            logger.error(str(e))
            env_var = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
            print(f"ERROR: {e}")
            print(f"\nTo fix, set the environment variable:")
            print(f"  export {env_var}='your-api-key-here'")
            print(f"\nOr create a .env file in the project root with:")
            print(f"  {env_var}=your-api-key-here")
            sys.exit(1)
        generator = SenseTextGenerator(client)
    else:
        generator = None
        logger.info("DRY RUN - no API calls will be made")

    logger.info("Generating text variants...")
    results_df = generate_variants(senses_df, generator, dry_run=args.dry_run)

    output_path = Path(args.output) if args.output else EVAL_DIR / "sense_variants.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(results_df)} sense variants to {output_path}")

    print(f"Generated text variants for {len(results_df)} senses")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()