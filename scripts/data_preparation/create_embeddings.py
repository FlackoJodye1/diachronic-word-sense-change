#!/usr/bin/env python3
"""
Script to create embeddings from existing GermaNet sense texts.

Usage:
    python scripts/create_embeddings.py path/to/sense_texts.csv
    python scripts/create_embeddings.py path/to/sense_texts.csv --model sentence-transformers/distiluse-base-multilingual-cased
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to the path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.data.germanet.transforms import EmbeddingBuilder


def main():
    """Main embedding creation function."""
    parser = argparse.ArgumentParser(description="Create embeddings from sense texts")
    
    parser.add_argument('sense_texts_csv', type=Path, help='Path to sense_texts.csv file')
    parser.add_argument('--model', type=str, default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    parser.add_argument('--batch-size', type=int, default=256)
    
    args = parser.parse_args()
    
    # Load sense texts
    print(f"Loading {args.sense_texts_csv}...")
    sense_texts_df = pd.read_csv(args.sense_texts_csv)
    print(f"Loaded {len(sense_texts_df)} sense texts")
    
    # Create embeddings
    print(f"Creating embeddings with {args.model}...")
    config = {
        'model_name': args.model,
        'batch_size': args.batch_size,
        'normalize_embeddings': True,
        'show_progress': True
    }
    
    builder = EmbeddingBuilder(config)
    embeddings = builder.create_embeddings(sense_texts_df)
    
    # Save in same directory as input file
    output_dir = args.sense_texts_csv.parent.parent
    embeddings_path, metadata_path = builder.save_embeddings(output_dir)
    
    print(f"✅ Saved embeddings: {embeddings_path}")
    print(f"✅ Saved metadata: {metadata_path}")
    print(f"✅ Shape: {embeddings.shape}")


if __name__ == "__main__":
    main()