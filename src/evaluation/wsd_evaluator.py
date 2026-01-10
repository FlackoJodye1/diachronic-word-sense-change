# -*- coding: utf-8 -*-
"""
WSD system evaluator using gold standard annotations.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from .metrics import WSDMetrics
from .error_analysis import ErrorAnalyzer


class WSDEvaluator:
    """Evaluates WSD systems against gold standard annotations."""
    
    def __init__(self, gold_standard_file: str):
        """Initialize evaluator with gold standard.
        
        Args:
            gold_standard_file: Path to gold standard CSV file
        """
        self.gold_standard_file = Path(gold_standard_file)
        self.gold_standard = self._load_gold_standard()
        self.metrics = WSDMetrics()
        self.error_analyzer = ErrorAnalyzer()
        self.logger = logging.getLogger(__name__)
        
    def _load_gold_standard(self) -> pd.DataFrame:
        """Load and validate gold standard data."""
        if not self.gold_standard_file.exists():
            raise FileNotFoundError(f"Gold standard file not found: {self.gold_standard_file}")
            
        df = pd.read_csv(self.gold_standard_file)
        
        # Validate required columns
        required_columns = ['instance_id', 'target_word', 'gold_sense_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in gold standard: {missing_columns}")
            
        return df
        
    def evaluate_predictions(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate WSD predictions against gold standard.
        
        Args:
            predictions: DataFrame with columns: instance_id, predicted_sense_id, confidence
            
        Returns:
            Evaluation results dictionary
        """
        # Validate predictions format
        required_columns = ['instance_id', 'predicted_sense_id']
        missing_columns = [col for col in required_columns if col not in predictions.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in predictions: {missing_columns}")
            
        # Merge predictions with gold standard
        eval_df = self.gold_standard.merge(
            predictions, 
            on='instance_id', 
            how='inner'
        )
        
        if eval_df.empty:
            raise ValueError("No matching instances found between predictions and gold standard")
            
        # Calculate metrics
        results = {
            'overview': {
                'total_instances': len(eval_df),
                'gold_standard_size': len(self.gold_standard),
                'predictions_size': len(predictions),
                'coverage': len(eval_df) / len(self.gold_standard)
            }
        }
        
        # Overall accuracy
        correct = eval_df['gold_sense_id'] == eval_df['predicted_sense_id']
        overall_accuracy = correct.mean()
        
        results['overall'] = {
            'accuracy': overall_accuracy,
            'correct_predictions': correct.sum(),
            'total_predictions': len(eval_df)
        }
        
        # Per-word analysis
        word_results = []
        for word in eval_df['target_word'].unique():
            word_df = eval_df[eval_df['target_word'] == word]
            word_correct = word_df['gold_sense_id'] == word_df['predicted_sense_id']
            
            word_results.append({
                'word': word,
                'instances': len(word_df),
                'accuracy': word_correct.mean(),
                'correct': word_correct.sum(),
                'senses_in_gold': word_df['gold_sense_id'].nunique(),
                'senses_predicted': word_df['predicted_sense_id'].nunique()
            })
            
        results['per_word'] = pd.DataFrame(word_results)
        
        # Confidence-based analysis (if available)
        if 'confidence' in eval_df.columns:
            confidence_analysis = self._analyze_confidence(eval_df)
            results['confidence'] = confidence_analysis
            
        # Error analysis
        error_analysis = self.error_analyzer.analyze_errors(eval_df)
        results['error_analysis'] = error_analysis
        
        # Detailed metrics
        detailed_metrics = self.metrics.calculate_detailed_metrics(eval_df)
        results['detailed_metrics'] = detailed_metrics
        
        self.logger.info(f"Evaluation complete: {overall_accuracy:.3f} accuracy on {len(eval_df)} instances")
        
        return results
        
    def _analyze_confidence(self, eval_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze prediction confidence vs accuracy correlation."""
        correct = eval_df['gold_sense_id'] == eval_df['predicted_sense_id']
        
        # Confidence bins
        eval_df['confidence_bin'] = pd.cut(
            eval_df['confidence'], 
            bins=[0, 0.5, 0.7, 0.8, 0.9, 1.0],
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )
        
        confidence_stats = eval_df.groupby('confidence_bin').agg({
            'instance_id': 'count',
            'gold_sense_id': lambda x: (eval_df.loc[x.index, 'gold_sense_id'] == 
                                      eval_df.loc[x.index, 'predicted_sense_id']).mean()
        }).round(3)
        
        confidence_stats.columns = ['count', 'accuracy']
        
        return {
            'confidence_accuracy_correlation': eval_df['confidence'].corr(correct.astype(int)),
            'confidence_bins': confidence_stats.to_dict('index'),
            'avg_confidence_correct': eval_df[correct]['confidence'].mean(),
            'avg_confidence_incorrect': eval_df[~correct]['confidence'].mean()
        }
        
    def evaluate_baseline(self, strategy: str = 'most_frequent') -> Dict[str, Any]:
        """Evaluate baseline strategy against gold standard.
        
        Args:
            strategy: Baseline strategy ('most_frequent', 'random')
            
        Returns:
            Baseline evaluation results
        """
        baseline_predictions = []
        
        for word in self.gold_standard['target_word'].unique():
            word_gold = self.gold_standard[self.gold_standard['target_word'] == word]
            
            if strategy == 'most_frequent':
                # Use most frequent sense in gold standard
                most_frequent_sense = word_gold['gold_sense_id'].mode().iloc[0]
                predictions = [most_frequent_sense] * len(word_gold)
            elif strategy == 'random':
                # Random selection from available senses
                available_senses = word_gold['gold_sense_id'].unique()
                predictions = np.random.choice(available_senses, size=len(word_gold))
            else:
                raise ValueError(f"Unknown baseline strategy: {strategy}")
                
            for instance_id, pred_sense in zip(word_gold['instance_id'], predictions):
                baseline_predictions.append({
                    'instance_id': instance_id,
                    'predicted_sense_id': pred_sense,
                    'confidence': 1.0 if strategy == 'most_frequent' else 0.5
                })
                
        baseline_df = pd.DataFrame(baseline_predictions)
        results = self.evaluate_predictions(baseline_df)
        results['baseline_strategy'] = strategy
        
        return results
        
    def compare_systems(self, system_predictions: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Compare multiple WSD systems.
        
        Args:
            system_predictions: Dict mapping system names to prediction DataFrames
            
        Returns:
            Comparison results
        """
        system_results = {}
        
        for system_name, predictions in system_predictions.items():
            try:
                results = self.evaluate_predictions(predictions)
                system_results[system_name] = results
            except Exception as e:
                self.logger.error(f"Error evaluating system '{system_name}': {e}")
                
        # Create comparison summary
        comparison_summary = []
        for system_name, results in system_results.items():
            summary = {
                'system': system_name,
                'overall_accuracy': results['overall']['accuracy'],
                'coverage': results['overview']['coverage'],
                'total_instances': results['overview']['total_instances']
            }
            
            if 'confidence' in results:
                summary['avg_confidence'] = results['confidence'].get('avg_confidence_correct', None)
                
            comparison_summary.append(summary)
            
        comparison_df = pd.DataFrame(comparison_summary).sort_values('overall_accuracy', ascending=False)
        
        return {
            'individual_results': system_results,
            'comparison_summary': comparison_df,
            'best_system': comparison_df.iloc[0]['system'] if not comparison_df.empty else None
        }
        
    def generate_report(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Generate human-readable evaluation report.
        
        Args:
            results: Evaluation results from evaluate_predictions
            output_file: Optional file to save report
            
        Returns:
            Report text
        """
        report_lines = [
            "WSD System Evaluation Report",
            "=" * 40,
            "",
            f"Overall Accuracy: {results['overall']['accuracy']:.3f}",
            f"Correct Predictions: {results['overall']['correct_predictions']} / {results['overall']['total_predictions']}",
            f"Coverage: {results['overview']['coverage']:.3f}",
            "",
            "Per-Word Results:",
            "-" * 20
        ]
        
        # Per-word results
        per_word = results['per_word'].sort_values('accuracy', ascending=False)
        for _, row in per_word.iterrows():
            report_lines.append(
                f"{row['word']:12} | {row['accuracy']:5.3f} | {row['correct']:3d}/{row['instances']:3d} instances | "
                f"{row['senses_in_gold']} gold senses"
            )
            
        # Error analysis summary
        if 'error_analysis' in results:
            report_lines.extend([
                "",
                "Error Analysis:",
                "-" * 15,
                f"Most confused sense pairs: {len(results['error_analysis'].get('confusion_pairs', []))} pairs",
                f"Words with highest error rates: {results['error_analysis'].get('high_error_words', [])[:5]}"
            ])
            
        # Confidence analysis
        if 'confidence' in results:
            conf = results['confidence']
            report_lines.extend([
                "",
                "Confidence Analysis:",
                "-" * 20,
                f"Confidence-accuracy correlation: {conf.get('confidence_accuracy_correlation', 0):.3f}",
                f"Avg confidence (correct): {conf.get('avg_confidence_correct', 0):.3f}",
                f"Avg confidence (incorrect): {conf.get('avg_confidence_incorrect', 0):.3f}"
            ])
            
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Report saved to {output_file}")
            
        return report_text