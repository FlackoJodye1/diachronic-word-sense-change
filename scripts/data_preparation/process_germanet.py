#!/usr/bin/env python3
"""
Script to process GermaNet XML data into structured tables and embeddings for WSD tasks.

This script handles the complete GermaNet processing pipeline:
1. Parse XML files (synsets, lexical units, relations)
2. Create derived tables (lemma_senses, sense_texts)
3. Enrich sense texts with semantic relations (hypernyms, meronyms, etc.)
4. Generate candidate mappings for fast WSD lookup
5. Optionally create embeddings using sentence transformers

Usage:
    python scripts/process_germanet.py data/raw/GN_V190 data/processed/germanet_v190
    python scripts/process_germanet.py data/raw/GN_V190 data/processed/germanet_v190 --embeddings
    python scripts/process_germanet.py data/raw/GN_V190 data/processed/germanet_v190 --no-relations  # Skip relation enrichment
    python scripts/process_germanet.py data/raw/GN_V190 data/processed/germanet_v190 --embeddings --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

# Add src to the path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.data.germanet.make_dataset import GermaNetProcessor


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_input_path(input_path: Path) -> Path:
    """Validate and resolve the GermaNet XML directory path."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    # Look for XML directory structure
    possible_xml_dirs = [
        input_path / "GN_V190_XML",
        input_path / "xml",
        input_path
    ]
    
    for xml_dir in possible_xml_dirs:
        if xml_dir.exists() and any(xml_dir.glob("*.xml")):
            logging.info(f"Found XML files in: {xml_dir}")
            return xml_dir
    
    raise FileNotFoundError(
        f"Could not find GermaNet XML files in {input_path}. "
        f"Expected structure: {input_path}/GN_V190_XML/*.xml"
    )


def create_config(args) -> Dict:
    """Create configuration dictionary from command line arguments."""
    config = {}
    
    if args.embeddings:
        config.update({
            'model_name': args.model,
            'batch_size': args.batch_size,
            'normalize_embeddings': True,
            'show_progress': True
        })
    
    if args.text_separator:
        config['text_separator'] = args.text_separator
    
    return config


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description="Process GermaNet XML data for Word Sense Disambiguation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Input directory containing GermaNet XML files (e.g., data/raw/GN_V190)'
    )
    parser.add_argument(
        'output_dir',
        type=Path,
        help='Output directory for processed data (e.g., data/processed/germanet_v190)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--embeddings',
        action='store_true',
        help='Create embeddings using sentence transformers (requires sentence-transformers package)'
    )
    parser.add_argument(
        '--no-relations',
        action='store_true',
        help='Skip relation enrichment of sense texts (faster but less semantic info)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        help='Sentence transformer model name for embeddings'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for embedding creation'
    )
    parser.add_argument(
        '--text-separator',
        type=str,
        default=' ; ',
        help='Separator for combining text elements'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print detailed processing statistics'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to JSON config file (overrides command line options)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input directory
        xml_dir = validate_input_path(args.input_dir)
        
        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")
        
        # Load config from file if provided
        config = {}
        if args.config and args.config.exists():
            with open(args.config) as f:
                config = json.load(f)
            logger.info(f"Loaded config from: {args.config}")
        
        # Override with command line arguments
        config.update(create_config(args))
        
        # Create processor
        logger.info("Initializing GermaNet processor...")
        processor = GermaNetProcessor(
            xml_dir=xml_dir,
            output_dir=args.output_dir,
            config=config
        )
        
        # Run processing pipeline
        logger.info("Starting GermaNet processing pipeline...")
        enrich_with_relations = not args.no_relations
        if enrich_with_relations:
            logger.info("Relation enrichment enabled (use --no-relations to disable)")
        else:
            logger.info("Relation enrichment disabled")
        
        processor.process_all(
            create_embeddings=args.embeddings,
            enrich_with_relations=enrich_with_relations
        )
        
        # Print statistics if requested
        if args.stats:
            logger.info("Generating processing statistics...")
            stats = processor.get_statistics()
            print("\n" + "="*50)
            print("PROCESSING STATISTICS")
            print("="*50)
            print(json.dumps(stats, indent=2, default=str))
        
        # Print output summary
        output_files = list(args.output_dir.glob("*"))
        logger.info(f"Processing completed successfully!")
        logger.info(f"Created {len(output_files)} output files in {args.output_dir}")
        
        # List key output files (check both root and germanet_v190 subdirectory)
        csv_files = [
            "synsets.csv", "lexical_units.csv", "relations.csv",
            "lemma_senses.csv", "sense_texts.csv"
        ]
        root_files = ["candidate_map.pkl"]
        if args.embeddings:
            root_files.extend(["sense_embeddings.npz", "sense_meta.csv"])
        
        print("\n" + "="*50)
        print("OUTPUT FILES")
        print("="*50)
        
        # Check CSV files in germanet_v190 subdirectory
        csv_dir = args.output_dir / "base-tables"
        for filename in csv_files:
            filepath = csv_dir / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"✅ {filename:<20} ({size_mb:.2f} MB)")
            else:
                print(f"❌ {filename:<20} (missing)")
        
        # Check root files
        for filename in root_files:
            filepath = args.output_dir / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"✅ {filename:<20} ({size_mb:.2f} MB)")
            else:
                print(f"❌ {filename:<20} (missing)")
        
        print(f"\nReady for WSD tasks! 🎯")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()