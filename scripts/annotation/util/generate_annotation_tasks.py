#!/usr/bin/env python3
"""
Generate annotation tasks for our evaluation word set.
Creates annotation tasks from eval contexts and GermaNet senses.
"""

import sys
import pandas as pd
import logging
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from annotation.task import AnnotationTask, AnnotationInstance

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def find_target_position(sentence: str, target_word: str) -> int:
    """Find position of target word in sentence."""
    words = sentence.split()
    target_word_lower = target_word.lower()
    
    for i, word in enumerate(words):
        # Remove punctuation for comparison
        clean_word = word.strip('.,!?;:"()[]').lower()
        if clean_word == target_word_lower:
            return i
    
    # Fallback: look for partial matches
    for i, word in enumerate(words):
        if target_word_lower in word.lower():
            return i
    
    logger.warning(f"Could not find '{target_word}' in sentence: {sentence}")
    return 0

def create_annotation_tasks() -> Dict[str, AnnotationTask]:
    """Create annotation tasks for all evaluation words."""
    
    # Paths
    base_path = Path(__file__).parent.parent.parent
    contexts_path = base_path / 'data' / 'evaluation' / 'eval_contexts.csv'
    words_path = base_path / 'data' / 'processed' / 'target_words' / 'eval_set_selection.tsv'
    
    # Load data
    logger.info("Loading evaluation contexts and word selection...")
    contexts_df = pd.read_csv(contexts_path)
    words_df = pd.read_csv(words_path, sep='\t')
    
    logger.info(f"Loaded {len(contexts_df)} contexts for {len(words_df)} target words")
    
    # Load WSD senses with proper text_for_embedding
    logger.info("Loading WSD senses with text_for_embedding...")
    wsd_senses_path = base_path / 'data' / 'processed' / 'germanet' / 'wsd_senses.csv'
    
    if not wsd_senses_path.exists():
        logger.error(f"WSD senses not found: {wsd_senses_path}")
        logger.error("Please ensure germanet processing has been completed")
        return {}
    
    wsd_senses_df = pd.read_csv(wsd_senses_path)
    logger.info(f"Loaded {len(wsd_senses_df)} WSD senses with text_for_embedding")
    
    annotation_tasks = {}
    
    for _, word_row in words_df.iterrows():
        target_word = word_row['word']
        logger.info(f"\nProcessing '{target_word}'...")
        
        # Get contexts for this word
        word_contexts = contexts_df[contexts_df['target_word'] == target_word].copy()
        
        if len(word_contexts) == 0:
            logger.warning(f"No contexts found for '{target_word}'")
            continue
        
        # Get WSD senses for this word with proper text_for_embedding
        logger.info(f"  Fetching WSD senses...")
        word_wsd_senses = wsd_senses_df[wsd_senses_df['lemma'] == target_word]
        
        if len(word_wsd_senses) == 0:
            logger.warning(f"No WSD senses found for '{target_word}'")
            continue
        
        logger.info(f"  Found {len(word_wsd_senses)} senses")
        
        # Prepare candidate senses from wsd_senses.csv
        candidate_senses = []
        for _, sense_row in word_wsd_senses.iterrows():
            synset_id = sense_row['synset_id']
            text_for_embedding = sense_row['text_for_embedding']
            word_class = sense_row['class']
            sense_frequency = sense_row['sense_frequency']
            
            candidate_sense = {
                'synset_id': synset_id,
                'text_for_embedding': text_for_embedding,
                'class': word_class,
                'sense_frequency': sense_frequency
            }
            candidate_senses.append(candidate_sense)
        
        # Create annotation instances
        instances = []
        for idx, context_row in word_contexts.iterrows():
            sentence = context_row['Hit']
            
            # Find target word position
            target_position = find_target_position(sentence, target_word)
            
            instance = AnnotationInstance(
                instance_id=f"{target_word}_{idx}",
                target_word=target_word,
                pos="UNKNOWN",  # We don't have POS info in our contexts
                sentence=sentence,
                target_position=target_position,
                candidate_senses=candidate_senses
            )
            instances.append(instance)
        
        # Create annotation task
        task = AnnotationTask(target_word, instances)
        annotation_tasks[target_word] = task
        
        logger.info(f"  Created task with {len(instances)} instances and {len(candidate_senses)} senses")
    
    return annotation_tasks

def save_annotation_tasks(tasks: Dict[str, AnnotationTask]) -> None:
    """Save annotation tasks to files."""
    
    base_path = Path(__file__).parent.parent.parent
    output_dir = base_path / 'data' / 'evaluation'
    output_dir.mkdir(exist_ok=True)
    
    # Save as pickle (for Flask app)
    pickle_path = output_dir / 'annotation_tasks.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(tasks, f)
    logger.info(f"Saved annotation tasks to {pickle_path}")
    
    # Save as JSON (for inspection/backup)
    json_path = output_dir / 'annotation_tasks.json'
    json_data = {}
    
    for word, task in tasks.items():
        json_data[word] = {
            'target_word': task.target_word,
            'instances': []
        }
        
        for instance in task.instances:
            json_data[word]['instances'].append({
                'instance_id': instance.instance_id,
                'target_word': instance.target_word,
                'pos': instance.pos,
                'sentence': instance.sentence,
                'target_position': instance.target_position,
                'candidate_senses': instance.candidate_senses
            })
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON backup to {json_path}")

def print_summary(tasks: Dict[str, AnnotationTask]) -> None:
    """Print summary of created annotation tasks."""
    
    logger.info("\n" + "="*50)
    logger.info("ANNOTATION TASKS SUMMARY")
    logger.info("="*50)
    
    total_instances = 0
    total_senses = 0
    
    for word, task in tasks.items():
        num_instances = len(task.instances)
        num_senses = len(task.instances[0].candidate_senses) if task.instances else 0
        
        total_instances += num_instances
        total_senses += num_senses
        
        logger.info(f"{word:10s}: {num_instances:2d} instances, {num_senses:2d} senses")
    
    logger.info("-" * 50)
    logger.info(f"{'TOTAL':10s}: {total_instances:2d} instances, {len(tasks):2d} words")
    logger.info(f"Average: {total_instances/len(tasks):.1f} instances per word")
    logger.info(f"Average: {total_senses/len(tasks):.1f} senses per word")
    
    logger.info("\n🚀 Ready for annotation! Run the Flask app:")
    logger.info("   python scripts/annotation/web_annotation_app.py")
    logger.info("   Then open: http://localhost:5002")

def main():
    logger.info("Generating annotation tasks for evaluation word set")
    logger.info("=" * 55)
    
    try:
        # Create annotation tasks
        tasks = create_annotation_tasks()
        
        if not tasks:
            logger.error("No annotation tasks created!")
            return 1
        
        # Save tasks
        save_annotation_tasks(tasks)
        
        # Print summary
        print_summary(tasks)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error creating annotation tasks: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())