#!/usr/bin/env python3
"""
Generate annotation tasks for quick eval set.
Creates annotation_tasks.pkl from quick_eval contexts and GermaNet senses.
"""

import sys
import pandas as pd
import pickle
import json
from pathlib import Path

# Add src to path (same as simple_app.py does)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from annotation.task import AnnotationTask, AnnotationInstance

def main():
    base_path = Path(__file__).parent.parent.parent

    # Load quick eval data
    contexts_path = base_path / 'data' / 'evaluation' / 'quick_eval' / 'contexts.csv'
    words_path = base_path / 'data' / 'evaluation' / 'quick_eval' / 'words.csv'
    wsd_senses_path = base_path / 'data' / 'processed' / 'germanet' / 'wsd_senses.csv'

    print("Loading data...")
    contexts_df = pd.read_csv(contexts_path)
    words_df = pd.read_csv(words_path)
    wsd_senses_df = pd.read_csv(wsd_senses_path)

    print(f"  Contexts: {len(contexts_df)}")
    print(f"  Words: {len(words_df)}")
    print(f"  WSD senses: {len(wsd_senses_df)}")

    annotation_tasks = {}

    for _, word_row in words_df.iterrows():
        lemma = word_row['lemma']
        pos = word_row['pos']

        print(f"\nProcessing '{lemma}' ({pos})...")

        # Get contexts for this word
        word_contexts = contexts_df[contexts_df['lemma'] == lemma]
        print(f"  Found {len(word_contexts)} contexts")

        # Get senses for this word
        word_senses = wsd_senses_df[wsd_senses_df['lemma'] == lemma]
        print(f"  Found {len(word_senses)} senses")

        if len(word_contexts) == 0 or len(word_senses) == 0:
            print(f"  ⚠️ Skipping - missing data")
            continue

        # Prepare candidate senses
        candidate_senses = []
        for _, sense_row in word_senses.iterrows():
            candidate_senses.append({
                'synset_id': sense_row['synset_id'],
                'text_for_embedding': sense_row['text_for_embedding'],
                'class': sense_row['class'],
                'paraphrase': sense_row.get('paraphrase', ''),
            })

        # Create instances
        instances = []
        for idx, (_, ctx_row) in enumerate(word_contexts.iterrows()):
            instance = AnnotationInstance(
                instance_id=f"{lemma}_{idx}",
                target_word=lemma,
                pos=pos,
                sentence=ctx_row['Hit'],
                target_position=0,  # Not used in simple app
                candidate_senses=candidate_senses
            )
            instances.append(instance)

        # Create task
        task = AnnotationTask(lemma, instances)
        annotation_tasks[lemma] = task
        print(f"  ✓ Created task: {len(instances)} instances, {len(candidate_senses)} senses")

    # Save to data/evaluation (where the app expects it)
    output_dir = base_path / 'data' / 'evaluation'
    output_dir.mkdir(exist_ok=True)

    # Save pickle
    pkl_path = output_dir / 'annotation_tasks.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(annotation_tasks, f)
    print(f"\n✅ Saved: {pkl_path}")

    # Save JSON backup
    json_path = output_dir / 'annotation_tasks.json'
    json_data = {}
    for word, task in annotation_tasks.items():
        json_data[word] = {
            'target_word': task.target_word,
            'num_instances': len(task.instances),
            'num_senses': len(task.instances[0].candidate_senses) if task.instances else 0,
            'instances': [
                {
                    'instance_id': inst.instance_id,
                    'sentence': inst.sentence,
                    'candidate_senses': inst.candidate_senses
                }
                for inst in task.instances
            ]
        }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {json_path}")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    total = 0
    for word, task in annotation_tasks.items():
        n = len(task.instances)
        s = len(task.instances[0].candidate_senses) if task.instances else 0
        print(f"  {word:12} {n:2} instances, {s:2} senses")
        total += n
    print("-"*50)
    print(f"  {'TOTAL':12} {total:2} instances")

    print("\n🚀 To start annotation, run:")
    print("   python scripts/annotation/simple_app.py")
    print("   Then open: http://localhost:8000")

if __name__ == '__main__':
    main()