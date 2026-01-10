# -*- coding: utf-8 -*-
"""
Error analysis for WSD evaluation.
"""

import pandas as pd
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict


class ErrorAnalyzer:
    """Analyzes prediction errors for WSD systems."""
    
    def analyze_errors(self, eval_df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive error analysis.
        
        Args:
            eval_df: DataFrame with gold_sense_id and predicted_sense_id
            
        Returns:
            Error analysis results
        """
        # Identify errors
        errors = eval_df[eval_df['gold_sense_id'] != eval_df['predicted_sense_id']].copy()
        correct = eval_df[eval_df['gold_sense_id'] == eval_df['predicted_sense_id']].copy()
        
        results = {
            'total_errors': len(errors),
            'total_correct': len(correct),
            'error_rate': len(errors) / len(eval_df) if len(eval_df) > 0 else 0
        }
        
        if len(errors) == 0:
            return results
            
        # Most confused sense pairs
        confusion_pairs = []
        for _, row in errors.iterrows():
            confusion_pairs.append((row['gold_sense_id'], row['predicted_sense_id']))
            
        pair_counts = Counter(confusion_pairs)
        top_confusions = pair_counts.most_common(10)
        
        results['confusion_pairs'] = [
            {
                'gold_sense': pair[0],
                'predicted_sense': pair[1], 
                'count': count,
                'percentage': count / len(errors)
            }
            for pair, count in top_confusions
        ]
        
        # Per-word error analysis
        word_errors = []
        for word in eval_df['target_word'].unique():
            word_data = eval_df[eval_df['target_word'] == word]
            word_error_data = errors[errors['target_word'] == word]
            
            if len(word_data) > 0:
                word_errors.append({
                    'word': word,
                    'total_instances': len(word_data),
                    'errors': len(word_error_data),
                    'error_rate': len(word_error_data) / len(word_data),
                    'accuracy': 1 - (len(word_error_data) / len(word_data))
                })
                
        word_errors_df = pd.DataFrame(word_errors).sort_values('error_rate', ascending=False)
        results['per_word_errors'] = word_errors_df.to_dict('records')
        results['high_error_words'] = word_errors_df.head(5)['word'].tolist()
        
        # Error patterns by sense frequency (if available)
        if 'sense_frequency' in eval_df.columns:
            freq_error_analysis = self._analyze_frequency_errors(eval_df, errors)
            results['frequency_error_patterns'] = freq_error_analysis
            
        # Error patterns by confidence (if available)
        if 'confidence' in eval_df.columns:
            conf_error_analysis = self._analyze_confidence_errors(eval_df, errors)
            results['confidence_error_patterns'] = conf_error_analysis
            
        return results
        
    def _analyze_frequency_errors(self, eval_df: pd.DataFrame, 
                                errors: pd.DataFrame) -> Dict[str, Any]:
        """Analyze errors by sense frequency patterns."""
        freq_patterns = {
            'primary_to_secondary': 0,  # Freq 1 -> Freq 2
            'secondary_to_primary': 0,  # Freq 2 -> Freq 1  
            'rare_to_frequent': 0,      # Freq >3 -> Freq <=2
            'frequent_to_rare': 0       # Freq <=2 -> Freq >3
        }
        
        # Note: This would need actual frequency mappings between senses
        # For now, return placeholder structure
        return {
            'patterns': freq_patterns,
            'most_common_pattern': 'primary_to_secondary',
            'analysis': 'Frequency-based error analysis requires sense frequency mappings'
        }
        
    def _analyze_confidence_errors(self, eval_df: pd.DataFrame,
                                 errors: pd.DataFrame) -> Dict[str, Any]:
        """Analyze errors by confidence levels."""
        if 'confidence' not in errors.columns:
            return {}
            
        # Confidence distribution of errors vs correct predictions  
        correct = eval_df[eval_df['gold_sense_id'] == eval_df['predicted_sense_id']]
        
        results = {
            'avg_confidence_errors': errors['confidence'].mean(),
            'avg_confidence_correct': correct['confidence'].mean(), 
            'high_confidence_errors': len(errors[errors['confidence'] >= 0.8]),
            'low_confidence_errors': len(errors[errors['confidence'] < 0.5])
        }
        
        # High-confidence errors (most concerning)
        high_conf_errors = errors[errors['confidence'] >= 0.8]
        if len(high_conf_errors) > 0:
            results['high_confidence_error_examples'] = high_conf_errors[[
                'target_word', 'gold_sense_id', 'predicted_sense_id', 'confidence'
            ]].head(5).to_dict('records')
            
        return results
        
    def find_systematic_errors(self, eval_df: pd.DataFrame) -> Dict[str, Any]:
        """Find systematic error patterns across the evaluation set."""
        errors = eval_df[eval_df['gold_sense_id'] != eval_df['predicted_sense_id']]
        
        if len(errors) == 0:
            return {"no_errors": True}
            
        # Pattern 1: Consistent bias towards certain senses
        predicted_sense_bias = errors['predicted_sense_id'].value_counts()
        gold_sense_distribution = eval_df['gold_sense_id'].value_counts()
        
        # Pattern 2: Consistent errors for certain words
        word_error_consistency = {}
        for word in errors['target_word'].unique():
            word_errors = errors[errors['target_word'] == word]
            if len(word_errors) > 1:
                # Check if same confusion pattern repeats
                confusion_pattern = word_errors.groupby(['gold_sense_id', 'predicted_sense_id']).size()
                repeated_confusions = confusion_pattern[confusion_pattern > 1]
                if not repeated_confusions.empty:
                    word_error_consistency[word] = repeated_confusions.to_dict()
                    
        # Pattern 3: Context-independent errors (would need sentence analysis)
        
        return {
            'sense_prediction_bias': predicted_sense_bias.head(5).to_dict(),
            'consistent_word_errors': word_error_consistency,
            'total_systematic_patterns': len(word_error_consistency)
        }
        
    def generate_error_report(self, error_analysis: Dict[str, Any]) -> str:
        """Generate human-readable error analysis report."""
        lines = [
            "WSD Error Analysis Report",
            "=" * 30,
            "",
            f"Total Errors: {error_analysis.get('total_errors', 0)}",
            f"Error Rate: {error_analysis.get('error_rate', 0):.3f}",
            ""
        ]
        
        # Top confusion pairs
        if 'confusion_pairs' in error_analysis:
            lines.extend([
                "Most Confused Sense Pairs:",
                "-" * 25
            ])
            
            for pair in error_analysis['confusion_pairs'][:5]:
                lines.append(
                    f"{pair['gold_sense']} → {pair['predicted_sense']} "
                    f"({pair['count']} errors, {pair['percentage']:.1%})"
                )
            lines.append("")
            
        # High-error words
        if 'high_error_words' in error_analysis:
            lines.extend([
                f"Words with Highest Error Rates:",
                "-" * 30
            ])
            
            for word in error_analysis['high_error_words']:
                word_info = next((w for w in error_analysis['per_word_errors'] 
                                if w['word'] == word), {})
                if word_info:
                    lines.append(
                        f"{word}: {word_info['error_rate']:.3f} "
                        f"({word_info['errors']}/{word_info['total_instances']})"
                    )
            lines.append("")
            
        # High-confidence errors
        if 'confidence_error_patterns' in error_analysis:
            conf_patterns = error_analysis['confidence_error_patterns']
            if 'high_confidence_errors' in conf_patterns:
                lines.extend([
                    "High-Confidence Errors (Most Concerning):",
                    "-" * 40,
                    f"Count: {conf_patterns['high_confidence_errors']}",
                    f"Avg confidence (errors): {conf_patterns.get('avg_confidence_errors', 0):.3f}",
                    f"Avg confidence (correct): {conf_patterns.get('avg_confidence_correct', 0):.3f}",
                    ""
                ])
                
        return "\n".join(lines)