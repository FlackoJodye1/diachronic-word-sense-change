# -*- coding: utf-8 -*-
"""
Annotation task representation for WSD ground truth creation.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from pathlib import Path
import pandas as pd


@dataclass
class AnnotationInstance:
    """Single annotation instance for WSD."""
    
    instance_id: str
    target_word: str
    pos: str
    sentence: str
    target_position: int  # Position of target word in sentence
    candidate_senses: List[Dict[str, Any]]  # List of possible senses with metadata
    context_window: int = 20  # Words before/after target for context
    
    def get_context(self) -> str:
        """Get contextual window around target word."""
        words = self.sentence.split()
        start = max(0, self.target_position - self.context_window)
        end = min(len(words), self.target_position + self.context_window + 1)
        
        context_words = words[start:end]
        # Highlight target word
        target_idx = self.target_position - start
        if 0 <= target_idx < len(context_words):
            context_words[target_idx] = f"**{context_words[target_idx]}**"
            
        return " ".join(context_words)


@dataclass  
class AnnotationResult:
    """Result of annotation for a single instance."""
    
    instance_id: str
    annotator_id: str
    selected_sense_id: str
    confidence: float  # 0-1 confidence score
    annotation_time_seconds: Optional[float] = None
    notes: Optional[str] = None
    timestamp: Optional[str] = None


class AnnotationTask:
    """Represents a collection of annotation instances for a specific target word."""
    
    def __init__(self, target_word: str, instances: List[AnnotationInstance]):
        """Initialize annotation task.
        
        Args:
            target_word: The German word to annotate
            instances: List of sentences containing the target word
        """
        self.target_word = target_word
        self.instances = instances
        self.results: List[AnnotationResult] = []
        
    def add_result(self, result: AnnotationResult) -> None:
        """Add annotation result."""
        self.results.append(result)
        
    def get_results_df(self) -> pd.DataFrame:
        """Get results as DataFrame."""
        if not self.results:
            return pd.DataFrame()
            
        return pd.DataFrame([
            {
                'instance_id': r.instance_id,
                'target_word': self.target_word,
                'annotator_id': r.annotator_id,
                'selected_sense_id': r.selected_sense_id,
                'confidence': r.confidence,
                'annotation_time': r.annotation_time_seconds,
                'notes': r.notes,
                'timestamp': r.timestamp
            }
            for r in self.results
        ])
        
        
    def save_results(self, output_path: Path) -> None:
        """Save annotation results to CSV."""
        df = self.get_results_df()
        df.to_csv(output_path, index=False)
        
    @classmethod
    def from_sentences(cls, target_word: str, sentences_df: pd.DataFrame, 
                      senses_df: pd.DataFrame) -> 'AnnotationTask':
        """Create annotation task from sentences and senses DataFrames.
        
        Args:
            target_word: Target word to annotate
            sentences_df: DataFrame with columns: sentence, target_position
            senses_df: DataFrame with sense information for the target word
            
        Returns:
            AnnotationTask instance
        """
        instances = []
        
        # Get candidate senses for this word
        word_senses = senses_df[senses_df['lemma'] == target_word]
        candidate_senses = []
        
        for _, sense in word_senses.iterrows():
            candidate_senses.append({
                'synset_id': sense['synset_id'],
                'text_for_embedding': sense.get('text_for_embedding', ''),
                'class': sense.get('class', ''),
                'paraphrase': sense.get('paraphrase', ''),
                'sense_frequency': sense.get('sense_frequency', 1)
            })
            
        # Create instances
        for idx, row in sentences_df.iterrows():
            instance = AnnotationInstance(
                instance_id=f"{target_word}_{idx}",
                target_word=target_word,
                pos="NOUN",  # Default, could be extracted
                sentence=row['sentence'],
                target_position=row.get('target_position', 0),
                candidate_senses=candidate_senses
            )
            instances.append(instance)
            
        return cls(target_word, instances)