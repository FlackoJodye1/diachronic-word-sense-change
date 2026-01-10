# -*- coding: utf-8 -*-
"""
Detailed metrics calculation for WSD evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class WSDMetrics:
    """Calculate detailed evaluation metrics for WSD systems."""
    
    def calculate_detailed_metrics(self, eval_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive metrics for WSD evaluation.
        
        Args:
            eval_df: DataFrame with gold_sense_id and predicted_sense_id columns
            
        Returns:
            Dictionary of detailed metrics
        """
        y_true = eval_df['gold_sense_id']
        y_pred = eval_df['predicted_sense_id']
        
        # Basic accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Get unique labels
        all_senses = sorted(set(y_true) | set(y_pred))
        
        # Precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=all_senses, average=None, zero_division=0
        )
        
        # Macro and micro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        
        # Per-sense metrics
        sense_metrics = []
        for i, sense in enumerate(all_senses):
            sense_metrics.append({
                'sense_id': sense,
                'precision': precision[i],
                'recall': recall[i], 
                'f1': f1[i],
                'support': int(support[i]) if i < len(support) else 0
            })
            
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels=all_senses)
        
        return {
            'accuracy': accuracy,
            'macro_avg': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1': macro_f1
            },
            'micro_avg': {
                'precision': micro_precision,
                'recall': micro_recall,
                'f1': micro_f1
            },
            'per_sense_metrics': sense_metrics,
            'confusion_matrix': conf_matrix.tolist(),
            'sense_labels': all_senses,
            'total_instances': len(eval_df),
            'unique_senses': len(all_senses)
        }
        
    def calculate_polysemy_metrics(self, eval_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics based on word polysemy level.
        
        Args:
            eval_df: Evaluation DataFrame with target_word column
            
        Returns:
            Polysemy-based metrics
        """
        # Calculate polysemy (number of senses per word)
        word_polysemy = eval_df.groupby('target_word')['gold_sense_id'].nunique()
        eval_df = eval_df.copy()
        eval_df['polysemy'] = eval_df['target_word'].map(word_polysemy)
        
        # Group by polysemy level
        polysemy_results = []
        for polysemy_level in sorted(eval_df['polysemy'].unique()):
            subset = eval_df[eval_df['polysemy'] == polysemy_level]
            accuracy = (subset['gold_sense_id'] == subset['predicted_sense_id']).mean()
            
            polysemy_results.append({
                'polysemy_level': int(polysemy_level),
                'words': int(subset['target_word'].nunique()),
                'instances': len(subset),
                'accuracy': accuracy
            })
            
        return {
            'polysemy_breakdown': polysemy_results,
            'avg_polysemy': word_polysemy.mean(),
            'max_polysemy': word_polysemy.max(),
            'monosemous_words': (word_polysemy == 1).sum(),
            'polysemous_words': (word_polysemy > 1).sum()
        }
        
    def calculate_frequency_metrics(self, eval_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics based on sense frequency.
        
        Args:
            eval_df: Evaluation DataFrame with sense_frequency column (if available)
            
        Returns:
            Frequency-based metrics
        """
        if 'sense_frequency' not in eval_df.columns:
            return {"error": "sense_frequency column not available"}
            
        # Group by frequency bins
        eval_df = eval_df.copy()
        eval_df['freq_bin'] = pd.cut(
            eval_df['sense_frequency'],
            bins=[0, 1, 2, 3, 10],
            labels=['primary', 'secondary', 'tertiary', 'rare'],
            include_lowest=True
        )
        
        freq_results = []
        for freq_bin in eval_df['freq_bin'].cat.categories:
            subset = eval_df[eval_df['freq_bin'] == freq_bin]
            if len(subset) > 0:
                accuracy = (subset['gold_sense_id'] == subset['predicted_sense_id']).mean()
                freq_results.append({
                    'frequency_bin': freq_bin,
                    'instances': len(subset),
                    'accuracy': accuracy,
                    'avg_frequency': subset['sense_frequency'].mean()
                })
                
        return {
            'frequency_breakdown': freq_results,
            'primary_sense_accuracy': freq_results[0]['accuracy'] if freq_results else 0,
            'rare_sense_accuracy': freq_results[-1]['accuracy'] if len(freq_results) > 3 else 0
        }