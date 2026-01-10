#!/usr/bin/env python3
"""
Create gold standard annotations for German WSD using human + AI annotation.

This script implements the hybrid annotation workflow:
1. Load GermaNet senses and DWDS corpus sentences
2. Prepare annotation tasks for target words
3. Run human annotation (interactive)
4. Run AI annotation (automatic)
5. Validate agreement between annotators
6. Create gold standard with conflict resolution
7. Export final annotated dataset

Usage:
    python scripts/create_gold_standard.py --words "Hund,Katze,laufen" --instances 10
    python scripts/create_gold_standard.py --word-file data/target_words_sample.txt --instances 20
    python scripts/create_gold_standard.py --all-words --instances 5  # Quick test
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.annotation import AnnotationWorkflow, HumanAnnotator, AIAnnotator
import pandas as pd


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('annotation.log')
        ]
    )


def load_target_words(words_arg: str, word_file: str, all_words: bool) -> List[str]:
    """Load target words from various sources."""
    if words_arg:
        return [w.strip() for w in words_arg.split(',')]
    elif word_file and Path(word_file).exists():
        with open(word_file) as f:
            return [line.strip() for line in f if line.strip()]
    elif all_words:
        # Load a few test words for demonstration
        return ["Hund", "Katze", "Auto", "Buch", "laufen"]
    else:
        raise ValueError("Must specify target words via --words, --word-file, or --all-words")


def load_corpus_data(corpus_file: str) -> pd.DataFrame:
    """Load DWDS corpus sentences."""
    if not Path(corpus_file).exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
        
    # Load DWDS sentences (adjust column names as needed)
    if corpus_file.endswith('.jsonl'):
        import json
        data = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(corpus_file)
    
    # Ensure required columns exist
    required_columns = ['lemma', 'sentence']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
        
    return df


def load_sense_data(sense_file: str) -> pd.DataFrame:
    """Load GermaNet sense data."""
    if not Path(sense_file).exists():
        raise FileNotFoundError(f"Sense file not found: {sense_file}")
        
    return pd.read_csv(sense_file)


def main():
    """Main annotation workflow."""
    parser = argparse.ArgumentParser(description="Create German WSD gold standard annotations")
    
    # Target word selection
    word_group = parser.add_mutually_exclusive_group(required=True)
    word_group.add_argument('--words', help='Comma-separated list of words (e.g., "Hund,Katze,Auto")')
    word_group.add_argument('--word-file', help='File containing target words (one per line)')
    word_group.add_argument('--all-words', action='store_true', help='Use predefined test words')
    
    # Data files
    parser.add_argument('--corpus-file', 
                       default='data/dwds_contexts/fetch_progress_25_words.jsonl',
                       help='DWDS corpus file with sentences')
    parser.add_argument('--sense-file',
                       default='data/processed/germanet/wsd_senses_enhanced.csv', 
                       help='GermaNet sense file')
    
    # Annotation parameters
    parser.add_argument('--instances', type=int, default=10,
                       help='Maximum instances per word to annotate')
    parser.add_argument('--output-dir', default='data/annotation_results',
                       help='Output directory for annotation results')
    
    # Workflow options
    parser.add_argument('--human-only', action='store_true',
                       help='Run only human annotation (skip AI)')
    parser.add_argument('--ai-only', action='store_true', 
                       help='Run only AI annotation (skip human)')
    parser.add_argument('--mock-ai', action='store_true', default=True,
                       help='Use mock AI annotation (default: True)')
    parser.add_argument('--resolution-strategy', default='human_priority',
                       choices=['human_priority', 'confidence', 'agreement', 'majority'],
                       help='Strategy for resolving annotation disagreements')
    
    # Other options
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load data
        logger.info("🔄 Loading data...")
        target_words = load_target_words(args.words, args.word_file, args.all_words)
        corpus_sentences = load_corpus_data(args.corpus_file)
        sense_data = load_sense_data(args.sense_file)
        
        logger.info(f"📝 Target words: {target_words}")
        logger.info(f"📊 Corpus: {len(corpus_sentences)} sentences")
        logger.info(f"🧠 Senses: {len(sense_data)} senses for {sense_data['lemma'].nunique()} words")
        
        # Initialize workflow
        workflow = AnnotationWorkflow(
            target_words=target_words,
            corpus_sentences=corpus_sentences,
            sense_data=sense_data,
            output_dir=Path(args.output_dir)
        )
        
        # Prepare tasks
        logger.info("🎯 Preparing annotation tasks...")
        workflow.prepare_tasks(instances_per_word=args.instances)
        
        # Show progress
        progress = workflow.get_progress_summary()
        logger.info(f"📋 Task summary:\n{progress}")
        
        # Process each word
        results_summary = []
        
        for word in target_words:
            if word not in workflow.tasks:
                logger.warning(f"⚠️  No task created for '{word}' (no sentences found)")
                continue
                
            logger.info(f"\n🚀 Processing word: '{word}'")
            
            try:
                if not args.ai_only:
                    # Human annotation
                    logger.info(f"👤 Human annotation for '{word}'...")
                    workflow.run_human_annotation(word, max_instances=args.instances)
                
                if not args.human_only:
                    # AI annotation
                    logger.info(f"🤖 AI annotation for '{word}'...")
                    workflow.run_ai_annotation(word, use_mock=args.mock_ai)
                
                # Validation (if both annotators ran)
                if not args.human_only and not args.ai_only:
                    logger.info(f"🔍 Validating annotations for '{word}'...")
                    validation_stats = workflow.validate_annotations(word)
                    logger.info(f"📊 Agreement: {validation_stats.get('agreement_rate', 0):.2%}")
                
                # Create gold standard
                logger.info(f"🏆 Creating gold standard for '{word}'...")
                gold_standard = workflow.create_gold_standard(
                    word, 
                    resolution_strategy=args.resolution_strategy
                )
                
                results_summary.append({
                    "word": word,
                    "instances_annotated": len(gold_standard),
                    "avg_confidence": gold_standard['confidence'].mean() if not gold_standard.empty else 0,
                    "agreement_rate": validation_stats.get('agreement_rate', 0) if not args.human_only and not args.ai_only else None
                })
                
                logger.info(f"✅ Completed '{word}': {len(gold_standard)} gold standard instances")
                
            except Exception as e:
                logger.error(f"❌ Error processing '{word}': {e}")
                results_summary.append({
                    "word": word,
                    "error": str(e)
                })
        
        # Final summary
        logger.info("\n🎉 Annotation workflow complete!")
        logger.info("📊 Results summary:")
        
        for result in results_summary:
            if "error" in result:
                logger.info(f"  ❌ {result['word']}: {result['error']}")
            else:
                logger.info(f"  ✅ {result['word']}: {result['instances_annotated']} instances, "
                          f"confidence: {result['avg_confidence']:.3f}")
                
        # Save combined results
        output_dir = Path(args.output_dir)
        combined_file = output_dir / "combined_gold_standard.csv"
        
        # Combine all gold standard files
        gold_standards = []
        for word in target_words:
            word_file = output_dir / f"gold_standard_{word}.csv"
            if word_file.exists():
                gold_df = pd.read_csv(word_file)
                gold_standards.append(gold_df)
        
        if gold_standards:
            from src.annotation.gold_standard import GoldStandardBuilder
            builder = GoldStandardBuilder()
            combined_gold = builder.merge_gold_standards(gold_standards)
            combined_gold.to_csv(combined_file, index=False)
            logger.info(f"💾 Saved combined gold standard: {combined_file} ({len(combined_gold)} instances)")
            
            # Validation report
            validation_report = builder.validate_gold_standard(combined_gold)
            logger.info(f"🔍 Gold standard quality: {validation_report}")
            
        logger.info(f"📁 All results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        return 1
        
    return 0


if __name__ == '__main__':
    exit(main())