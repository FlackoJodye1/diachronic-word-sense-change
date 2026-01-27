#!/usr/bin/env python3
"""
Export annotation tasks to human-readable JSON format.
"""

import sys
import pickle
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def export_tasks_to_json():
    """Export pickle tasks to readable JSON."""
    
    # Load pickle file
    data_dir = Path(__file__).parent.parent / 'data' / 'evaluation'
    pickle_path = data_dir / 'annotation_tasks_final.pkl'
    
    with open(pickle_path, 'rb') as f:
        tasks = pickle.load(f)
    
    # Convert to JSON-serializable format
    json_data = {}
    
    for word, task in tasks.items():
        json_data[word] = {
            'target_word': task.target_word,
            'instances': [],
            'results': []
        }
        
        # Convert instances
        for instance in task.instances:
            json_data[word]['instances'].append({
                'instance_id': instance.instance_id,
                'target_word': instance.target_word,
                'pos': instance.pos,
                'sentence': instance.sentence,
                'target_position': instance.target_position,
                'candidate_senses': instance.candidate_senses,
                'context_window': instance.context_window
            })
        
        # Convert results
        for result in task.results:
            json_data[word]['results'].append({
                'instance_id': result.instance_id,
                'annotator_id': result.annotator_id,
                'selected_sense_id': result.selected_sense_id,
                'confidence': result.confidence,
                'annotation_time_seconds': result.annotation_time_seconds,
                'notes': result.notes,
                'timestamp': result.timestamp
            })
    
    # Save as JSON
    json_path = data_dir / 'annotation_tasks_final.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported annotation tasks to: {json_path}")
    print(f"📊 {len(json_data)} words, {sum(len(data['instances']) for data in json_data.values())} instances")
    
    # Also create a summary file
    summary_path = data_dir / 'annotation_summary.json'
    summary = {
        'dataset_info': {
            'total_words': len(json_data),
            'total_instances': sum(len(data['instances']) for data in json_data.values()),
            'instances_per_word': 5,
            'estimated_annotation_time': '45-60 minutes'
        },
        'words_by_difficulty': {
            'easy': [],
            'medium': [], 
            'hard': []
        },
        'word_details': {}
    }
    
    # Add word details
    for word, data in json_data.items():
        num_senses = len(data['instances'][0]['candidate_senses']) if data['instances'] else 0
        summary['word_details'][word] = {
            'instances': len(data['instances']),
            'senses': num_senses,
            'difficulty': 'easy' if num_senses <= 2 else 'medium' if num_senses <= 3 else 'hard'
        }
        
        # Categorize by difficulty
        difficulty = summary['word_details'][word]['difficulty']
        summary['words_by_difficulty'][difficulty].append(word)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Created summary at: {summary_path}")

if __name__ == '__main__':
    export_tasks_to_json()