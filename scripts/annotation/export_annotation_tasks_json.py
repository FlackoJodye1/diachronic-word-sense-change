#!/usr/bin/env python3
"""
Export annotation tasks pickle to human-readable JSON format.
"""

import json
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "evaluation"


def export_tasks_to_json() -> None:
    """Export pickle tasks to readable JSON and a summary file."""
    pickle_path = DATA_DIR / "annotation_tasks_final.pkl"

    with open(pickle_path, "rb") as f:
        tasks = pickle.load(f)

    # Convert to JSON-serializable format
    json_data = {}
    for word, task in tasks.items():
        json_data[word] = {
            "target_word": task.target_word,
            "instances": [
                {
                    "instance_id": instance.instance_id,
                    "target_word": instance.target_word,
                    "pos": instance.pos,
                    "sentence": instance.sentence,
                    "target_position": instance.target_position,
                    "candidate_senses": instance.candidate_senses,
                    "context_window": instance.context_window,
                }
                for instance in task.instances
            ],
            "results": [
                {
                    "instance_id": result.instance_id,
                    "annotator_id": result.annotator_id,
                    "selected_sense_id": result.selected_sense_id,
                    "confidence": result.confidence,
                    "annotation_time_seconds": result.annotation_time_seconds,
                    "notes": result.notes,
                    "timestamp": result.timestamp,
                }
                for result in task.results
            ],
        }

    json_path = DATA_DIR / "annotation_tasks_final.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    total_instances = sum(len(d["instances"]) for d in json_data.values())
    logger.info(f"Exported {len(json_data)} words, {total_instances} instances -> {json_path}")

    # Summary file
    summary = {
        "total_words": len(json_data),
        "total_instances": total_instances,
        "word_details": {},
        "words_by_difficulty": {"easy": [], "medium": [], "hard": []},
    }

    for word, data in json_data.items():
        num_senses = len(data["instances"][0]["candidate_senses"]) if data["instances"] else 0
        difficulty = "easy" if num_senses <= 2 else "medium" if num_senses <= 3 else "hard"
        summary["word_details"][word] = {
            "instances": len(data["instances"]),
            "senses": num_senses,
            "difficulty": difficulty,
        }
        summary["words_by_difficulty"][difficulty].append(word)

    summary_path = DATA_DIR / "annotation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary -> {summary_path}")


if __name__ == "__main__":
    export_tasks_to_json()