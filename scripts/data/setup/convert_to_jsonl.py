#!/usr/bin/env python3
"""
Convert target_word_senses_enhanced CSV to JSONL format.
"""

import pandas as pd
import json
from pathlib import Path


def convert_csv_to_jsonl(csv_path: str, output_path: str):
    """Convert enhanced CSV to JSONL format with cleaner structure."""

    # Read the CSV
    df = pd.read_csv(csv_path)

    # Open output file
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
                'baseline': row['text_for_embedding'],  # Using text_for_embedding as baseline
                'llm_definition': row['llm_definition'],
                'llm_examples': examples
            }

            # Write as single line JSON
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"✓ Converted {len(df)} records")
    print(f"✓ Written to: {output_path}")


def main():
    base_dir = Path(__file__).parent.parent.parent

    # Convert full version
    csv_path = base_dir / "data/processed/germanet/target_word_senses_enhanced_full.csv"
    output_path = base_dir / "data/processed/germanet/target_word_senses_enhanced_full.jsonl"

    print(f"Converting: {csv_path.name}")
    convert_csv_to_jsonl(str(csv_path), str(output_path))

    # Also convert pilot version
    csv_path_pilot = base_dir / "data/processed/germanet/target_word_senses_enhanced_pilot.csv"
    output_path_pilot = base_dir / "data/processed/germanet/target_word_senses_enhanced_pilot.jsonl"

    if csv_path_pilot.exists():
        print(f"\nConverting: {csv_path_pilot.name}")
        convert_csv_to_jsonl(str(csv_path_pilot), str(output_path_pilot))


if __name__ == "__main__":
    main()