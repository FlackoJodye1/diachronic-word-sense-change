#!/usr/bin/env python3
"""
Process GermaNet XML data into structured tables for WSD tasks.

This script handles the complete GermaNet processing pipeline:
1. Parse XML files (synsets, lexical units, relations)
2. Create derived tables (lemma_senses, sense_texts)
3. Enrich sense texts with semantic relations (hypernyms, meronyms, etc.)
4. Generate candidate mappings for fast WSD lookup

Usage:
    python scripts/data/process_germanet.py data/raw/GN_V190 data/processed/germanet
    python scripts/data/process_germanet.py data/raw/GN_V190 data/processed/germanet --no-relations
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.germanet.make_dataset import GermaNetProcessor

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_input_path(input_path: Path) -> Path:
    """Validate and resolve the GermaNet XML directory path."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    for xml_dir in [input_path / "GN_V190_XML", input_path / "xml", input_path]:
        if xml_dir.exists() and any(xml_dir.glob("*.xml")):
            logger.info(f"Found XML files in: {xml_dir}")
            return xml_dir

    raise FileNotFoundError(
        f"Could not find GermaNet XML files in {input_path}. "
        f"Expected structure: {input_path}/GN_V190_XML/*.xml"
    )


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description="Process GermaNet XML data for Word Sense Disambiguation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing GermaNet XML files (e.g., data/raw/GN_V190)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for processed data (e.g., data/processed/germanet)",
    )
    parser.add_argument(
        "--no-relations",
        action="store_true",
        help="Skip relation enrichment of sense texts (faster but less semantic info)",
    )
    parser.add_argument(
        "--text-separator",
        type=str,
        default=" ; ",
        help="Separator for combining text elements",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print detailed processing statistics",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        xml_dir = validate_input_path(args.input_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")

        config = {}
        if args.text_separator:
            config["text_separator"] = args.text_separator

        logger.info("Initializing GermaNet processor...")
        processor = GermaNetProcessor(
            xml_dir=xml_dir,
            output_dir=args.output_dir,
            config=config,
        )

        logger.info("Starting GermaNet processing pipeline...")
        enrich_with_relations = not args.no_relations
        logger.info(
            "Relation enrichment " + ("enabled" if enrich_with_relations else "disabled (--no-relations)")
        )

        processor.process_all(enrich_with_relations=enrich_with_relations)

        if args.stats:
            stats = processor.get_statistics()
            logger.info("\n" + "=" * 50)
            logger.info("PROCESSING STATISTICS")
            logger.info("=" * 50)
            logger.info(json.dumps(stats, indent=2, default=str))

        output_files = list(args.output_dir.glob("*"))
        logger.info(f"Processing complete. Created {len(output_files)} output files in {args.output_dir}")

        csv_files = ["synsets.csv", "lexical_units.csv", "relations.csv", "lemma_senses.csv", "sense_texts.csv"]
        logger.info("=" * 50)
        csv_dir = args.output_dir / "base-tables"
        for filename in csv_files + ["candidate_map.pkl"]:
            filepath = (csv_dir if filename.endswith(".csv") else args.output_dir) / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                logger.info(f"  {filename:<20} ({size_mb:.2f} MB)")
            else:
                logger.warning(f"  {filename:<20} (missing)")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
