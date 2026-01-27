#!/usr/bin/env python3
"""
Run manual annotation for evaluation dataset.
"""

import sys
import pickle
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from annotation.human_annotator import HumanAnnotator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_annotation_tasks():
    """Load prepared annotation tasks."""
    data_dir = Path(__file__).parent.parent / 'data' / 'evaluation'
    tasks_path = data_dir / 'annotation_tasks.pkl'
    
    with open(tasks_path, 'rb') as f:
        tasks = pickle.load(f)
    
    logger.info(f"Loaded {len(tasks)} annotation tasks")
    return tasks

def save_annotation_results(tasks, output_dir):
    """Save annotation results to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for word, task in tasks.items():
        if task.results:
            # Save results for this word
            word_output = output_dir / f"{word}_annotations.csv"
            task.save_results(word_output)
            logger.info(f"Saved {len(task.results)} annotations for '{word}' to {word_output}")
    
    # Save combined results
    combined_results = []
    for word, task in tasks.items():
        if task.results:
            df = task.get_results_df()
            combined_results.append(df)
    
    if combined_results:
        import pandas as pd
        combined_df = pd.concat(combined_results, ignore_index=True)
        combined_output = output_dir / "all_annotations.csv"
        combined_df.to_csv(combined_output, index=False)
        logger.info(f"Saved {len(combined_df)} total annotations to {combined_output}")

def main():
    """Run annotation interface."""
    
    # Load tasks
    tasks = load_annotation_tasks()
    
    # Create annotator
    annotator_id = input("Enter your annotator ID (e.g., your name): ").strip()
    if not annotator_id:
        annotator_id = "annotator_1"
    
    annotator = HumanAnnotator(annotator_id)
    
    # Show available words
    print(f"\n📋 Available words to annotate ({len(tasks)} total):")
    words_by_difficulty = {}
    
    for word, task in tasks.items():
        # Get difficulty from first instance
        difficulty = "unknown"
        if task.instances:
            # We need to look up difficulty from our data
            pass
        
        num_senses = len(task.instances[0].candidate_senses) if task.instances else 0
        completed = len([r for r in task.results if r.annotator_id == annotator_id])
        
        print(f"  {word:12} - {num_senses:2} senses, {completed:2}/{len(task.instances)} annotated")
    
    print(f"\n🎯 Options:")
    print(f"  1. Annotate specific word")
    print(f"  2. Continue from where you left off")
    print(f"  3. Start fresh with first word")
    
    choice = input("\nYour choice (1-3): ").strip()
    
    if choice == "1":
        # Annotate specific word
        word = input("Enter word name: ").strip()
        if word not in tasks:
            print(f"Word '{word}' not found!")
            return
        
        task = tasks[word]
        task = annotator.resume_task(task)
        tasks[word] = task
        
    elif choice == "2":
        # Resume from where left off
        for word, task in tasks.items():
            print(f"\n🔄 Resuming '{word}'...")
            task = annotator.resume_task(task)
            tasks[word] = task
            
            # Ask if want to continue to next word
            if input(f"\nContinue to next word? (y/n): ").lower() != 'y':
                break
                
    elif choice == "3":
        # Start with first word
        word = list(tasks.keys())[0]
        print(f"\n🚀 Starting with '{word}'")
        task = tasks[word]
        task = annotator.annotate_task(task)
        tasks[word] = task
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'data' / 'evaluation' / 'annotations'
    save_annotation_results(tasks, output_dir)
    
    # Update tasks file
    tasks_path = Path(__file__).parent.parent / 'data' / 'evaluation' / 'annotation_tasks.pkl'
    with open(tasks_path, 'wb') as f:
        pickle.dump(tasks, f)
    
    print(f"\n✅ Annotation session complete!")
    print(f"📁 Results saved to: {output_dir}")

if __name__ == '__main__':
    main()