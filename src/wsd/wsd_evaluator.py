"""WSD evaluation framework using gold standard annotations."""

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

logger = logging.getLogger(__name__)


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

        accuracy = accuracy_score(y_true, y_pred)
        all_senses = sorted(set(y_true) | set(y_pred))

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=all_senses, average=None, zero_division=0
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )

        sense_metrics = [
            {
                'sense_id': sense,
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i]) if i < len(support) else 0,
            }
            for i, sense in enumerate(all_senses)
        ]

        conf_matrix = confusion_matrix(y_true, y_pred, labels=all_senses)

        return {
            'accuracy': accuracy,
            'macro_avg': {'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1},
            'micro_avg': {'precision': micro_precision, 'recall': micro_recall, 'f1': micro_f1},
            'per_sense_metrics': sense_metrics,
            'confusion_matrix': conf_matrix.tolist(),
            'sense_labels': all_senses,
            'total_instances': len(eval_df),
            'unique_senses': len(all_senses),
        }

    def calculate_polysemy_metrics(self, eval_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics based on word polysemy level."""
        word_polysemy = eval_df.groupby('target_word')['gold_sense_id'].nunique()
        eval_df = eval_df.copy()
        eval_df['polysemy'] = eval_df['target_word'].map(word_polysemy)

        polysemy_results = []
        for polysemy_level in sorted(eval_df['polysemy'].unique()):
            subset = eval_df[eval_df['polysemy'] == polysemy_level]
            accuracy = (subset['gold_sense_id'] == subset['predicted_sense_id']).mean()
            polysemy_results.append({
                'polysemy_level': int(polysemy_level),
                'words': int(subset['target_word'].nunique()),
                'instances': len(subset),
                'accuracy': accuracy,
            })

        return {
            'polysemy_breakdown': polysemy_results,
            'avg_polysemy': word_polysemy.mean(),
            'max_polysemy': word_polysemy.max(),
            'monosemous_words': (word_polysemy == 1).sum(),
            'polysemous_words': (word_polysemy > 1).sum(),
        }

    def calculate_frequency_metrics(self, eval_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics based on sense frequency."""
        if 'sense_frequency' not in eval_df.columns:
            return {"error": "sense_frequency column not available"}

        eval_df = eval_df.copy()
        eval_df['freq_bin'] = pd.cut(
            eval_df['sense_frequency'],
            bins=[0, 1, 2, 3, 10],
            labels=['primary', 'secondary', 'tertiary', 'rare'],
            include_lowest=True,
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
                    'avg_frequency': subset['sense_frequency'].mean(),
                })

        return {
            'frequency_breakdown': freq_results,
            'primary_sense_accuracy': freq_results[0]['accuracy'] if freq_results else 0,
            'rare_sense_accuracy': freq_results[-1]['accuracy'] if len(freq_results) > 3 else 0,
        }


class ErrorAnalyzer:
    """Analyzes prediction errors for WSD systems."""

    def analyze_errors(self, eval_df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive error analysis."""
        errors = eval_df[eval_df['gold_sense_id'] != eval_df['predicted_sense_id']].copy()
        correct = eval_df[eval_df['gold_sense_id'] == eval_df['predicted_sense_id']].copy()

        results = {
            'total_errors': len(errors),
            'total_correct': len(correct),
            'error_rate': len(errors) / len(eval_df) if len(eval_df) > 0 else 0,
        }

        if len(errors) == 0:
            return results

        # Most confused sense pairs
        confusion_pairs = [
            (row['gold_sense_id'], row['predicted_sense_id'])
            for _, row in errors.iterrows()
        ]
        pair_counts = Counter(confusion_pairs)
        results['confusion_pairs'] = [
            {
                'gold_sense': pair[0],
                'predicted_sense': pair[1],
                'count': count,
                'percentage': count / len(errors),
            }
            for pair, count in pair_counts.most_common(10)
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
                    'accuracy': 1 - (len(word_error_data) / len(word_data)),
                })

        word_errors_df = pd.DataFrame(word_errors).sort_values('error_rate', ascending=False)
        results['per_word_errors'] = word_errors_df.to_dict('records')
        results['high_error_words'] = word_errors_df.head(5)['word'].tolist()

        if 'sense_frequency' in eval_df.columns:
            results['frequency_error_patterns'] = self._analyze_frequency_errors(eval_df, errors)
        if 'confidence' in eval_df.columns:
            results['confidence_error_patterns'] = self._analyze_confidence_errors(eval_df, errors)

        return results

    def _analyze_frequency_errors(
        self, eval_df: pd.DataFrame, errors: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze errors by sense frequency patterns."""
        return {
            'patterns': {
                'primary_to_secondary': 0,
                'secondary_to_primary': 0,
                'rare_to_frequent': 0,
                'frequent_to_rare': 0,
            },
            'most_common_pattern': 'primary_to_secondary',
            'analysis': 'Frequency-based error analysis requires sense frequency mappings',
        }

    def _analyze_confidence_errors(
        self, eval_df: pd.DataFrame, errors: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze errors by confidence levels."""
        if 'confidence' not in errors.columns:
            return {}

        correct = eval_df[eval_df['gold_sense_id'] == eval_df['predicted_sense_id']]
        results = {
            'avg_confidence_errors': errors['confidence'].mean(),
            'avg_confidence_correct': correct['confidence'].mean(),
            'high_confidence_errors': len(errors[errors['confidence'] >= 0.8]),
            'low_confidence_errors': len(errors[errors['confidence'] < 0.5]),
        }

        high_conf_errors = errors[errors['confidence'] >= 0.8]
        if len(high_conf_errors) > 0:
            results['high_confidence_error_examples'] = high_conf_errors[
                ['target_word', 'gold_sense_id', 'predicted_sense_id', 'confidence']
            ].head(5).to_dict('records')

        return results

    def find_systematic_errors(self, eval_df: pd.DataFrame) -> Dict[str, Any]:
        """Find systematic error patterns across the evaluation set."""
        errors = eval_df[eval_df['gold_sense_id'] != eval_df['predicted_sense_id']]

        if len(errors) == 0:
            return {"no_errors": True}

        predicted_sense_bias = errors['predicted_sense_id'].value_counts()
        word_error_consistency = {}
        for word in errors['target_word'].unique():
            word_errors = errors[errors['target_word'] == word]
            if len(word_errors) > 1:
                confusion_pattern = word_errors.groupby(
                    ['gold_sense_id', 'predicted_sense_id']
                ).size()
                repeated_confusions = confusion_pattern[confusion_pattern > 1]
                if not repeated_confusions.empty:
                    word_error_consistency[word] = repeated_confusions.to_dict()

        return {
            'sense_prediction_bias': predicted_sense_bias.head(5).to_dict(),
            'consistent_word_errors': word_error_consistency,
            'total_systematic_patterns': len(word_error_consistency),
        }

    def generate_error_report(self, error_analysis: Dict[str, Any]) -> str:
        """Generate human-readable error analysis report."""
        lines = [
            "WSD Error Analysis Report",
            "=" * 30,
            "",
            f"Total Errors: {error_analysis.get('total_errors', 0)}",
            f"Error Rate: {error_analysis.get('error_rate', 0):.3f}",
            "",
        ]

        if 'confusion_pairs' in error_analysis:
            lines.extend(["Most Confused Sense Pairs:", "-" * 25])
            for pair in error_analysis['confusion_pairs'][:5]:
                lines.append(
                    f"{pair['gold_sense']} → {pair['predicted_sense']} "
                    f"({pair['count']} errors, {pair['percentage']:.1%})"
                )
            lines.append("")

        if 'high_error_words' in error_analysis:
            lines.extend(["Words with Highest Error Rates:", "-" * 30])
            for word in error_analysis['high_error_words']:
                word_info = next(
                    (w for w in error_analysis['per_word_errors'] if w['word'] == word), {}
                )
                if word_info:
                    lines.append(
                        f"{word}: {word_info['error_rate']:.3f} "
                        f"({word_info['errors']}/{word_info['total_instances']})"
                    )
            lines.append("")

        if 'confidence_error_patterns' in error_analysis:
            conf = error_analysis['confidence_error_patterns']
            if 'high_confidence_errors' in conf:
                lines.extend([
                    "High-Confidence Errors (Most Concerning):",
                    "-" * 40,
                    f"Count: {conf['high_confidence_errors']}",
                    f"Avg confidence (errors): {conf.get('avg_confidence_errors', 0):.3f}",
                    f"Avg confidence (correct): {conf.get('avg_confidence_correct', 0):.3f}",
                    "",
                ])

        return "\n".join(lines)


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

    def _load_gold_standard(self) -> pd.DataFrame:
        """Load and validate gold standard data."""
        if not self.gold_standard_file.exists():
            raise FileNotFoundError(f"Gold standard file not found: {self.gold_standard_file}")

        df = pd.read_csv(self.gold_standard_file)

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
        required_columns = ['instance_id', 'predicted_sense_id']
        missing_columns = [col for col in required_columns if col not in predictions.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in predictions: {missing_columns}")

        eval_df = self.gold_standard.merge(predictions, on='instance_id', how='inner')

        if eval_df.empty:
            raise ValueError("No matching instances found between predictions and gold standard")

        results = {
            'overview': {
                'total_instances': len(eval_df),
                'gold_standard_size': len(self.gold_standard),
                'predictions_size': len(predictions),
                'coverage': len(eval_df) / len(self.gold_standard),
            }
        }

        correct = eval_df['gold_sense_id'] == eval_df['predicted_sense_id']
        overall_accuracy = correct.mean()
        results['overall'] = {
            'accuracy': overall_accuracy,
            'correct_predictions': correct.sum(),
            'total_predictions': len(eval_df),
        }

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
                'senses_predicted': word_df['predicted_sense_id'].nunique(),
            })
        results['per_word'] = pd.DataFrame(word_results)

        if 'confidence' in eval_df.columns:
            results['confidence'] = self._analyze_confidence(eval_df)

        results['error_analysis'] = self.error_analyzer.analyze_errors(eval_df)
        results['detailed_metrics'] = self.metrics.calculate_detailed_metrics(eval_df)

        logger.info(
            f"Evaluation complete: {overall_accuracy:.3f} accuracy on {len(eval_df)} instances"
        )
        return results

    def _analyze_confidence(self, eval_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze prediction confidence vs accuracy correlation."""
        correct = eval_df['gold_sense_id'] == eval_df['predicted_sense_id']

        eval_df['confidence_bin'] = pd.cut(
            eval_df['confidence'],
            bins=[0, 0.5, 0.7, 0.8, 0.9, 1.0],
            labels=['very_low', 'low', 'medium', 'high', 'very_high'],
        )

        confidence_stats = eval_df.groupby('confidence_bin').agg({
            'instance_id': 'count',
            'gold_sense_id': lambda x: (
                eval_df.loc[x.index, 'gold_sense_id'] == eval_df.loc[x.index, 'predicted_sense_id']
            ).mean(),
        }).round(3)
        confidence_stats.columns = ['count', 'accuracy']

        return {
            'confidence_accuracy_correlation': eval_df['confidence'].corr(correct.astype(int)),
            'confidence_bins': confidence_stats.to_dict('index'),
            'avg_confidence_correct': eval_df[correct]['confidence'].mean(),
            'avg_confidence_incorrect': eval_df[~correct]['confidence'].mean(),
        }

    def evaluate_baseline(self, strategy: str = 'most_frequent') -> Dict[str, Any]:
        """Evaluate baseline strategy against gold standard.

        Args:
            strategy: Baseline strategy ('most_frequent', 'random')
        """
        baseline_predictions = []

        for word in self.gold_standard['target_word'].unique():
            word_gold = self.gold_standard[self.gold_standard['target_word'] == word]

            if strategy == 'most_frequent':
                most_frequent_sense = word_gold['gold_sense_id'].mode().iloc[0]
                predictions = [most_frequent_sense] * len(word_gold)
            elif strategy == 'random':
                available_senses = word_gold['gold_sense_id'].unique()
                predictions = np.random.choice(available_senses, size=len(word_gold))
            else:
                raise ValueError(f"Unknown baseline strategy: {strategy}")

            for instance_id, pred_sense in zip(word_gold['instance_id'], predictions):
                baseline_predictions.append({
                    'instance_id': instance_id,
                    'predicted_sense_id': pred_sense,
                    'confidence': 1.0 if strategy == 'most_frequent' else 0.5,
                })

        baseline_df = pd.DataFrame(baseline_predictions)
        results = self.evaluate_predictions(baseline_df)
        results['baseline_strategy'] = strategy
        return results

    def compare_systems(self, system_predictions: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Compare multiple WSD systems.

        Args:
            system_predictions: Dict mapping system names to prediction DataFrames
        """
        system_results = {}
        for system_name, predictions in system_predictions.items():
            try:
                system_results[system_name] = self.evaluate_predictions(predictions)
            except Exception as e:
                logger.error(f"Error evaluating system '{system_name}': {e}")

        comparison_summary = []
        for system_name, results in system_results.items():
            summary = {
                'system': system_name,
                'overall_accuracy': results['overall']['accuracy'],
                'coverage': results['overview']['coverage'],
                'total_instances': results['overview']['total_instances'],
            }
            if 'confidence' in results:
                summary['avg_confidence'] = results['confidence'].get('avg_confidence_correct')
            comparison_summary.append(summary)

        comparison_df = pd.DataFrame(comparison_summary).sort_values(
            'overall_accuracy', ascending=False
        )
        return {
            'individual_results': system_results,
            'comparison_summary': comparison_df,
            'best_system': comparison_df.iloc[0]['system'] if not comparison_df.empty else None,
        }

    def generate_report(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Generate human-readable evaluation report."""
        report_lines = [
            "WSD System Evaluation Report",
            "=" * 40,
            "",
            f"Overall Accuracy: {results['overall']['accuracy']:.3f}",
            f"Correct Predictions: {results['overall']['correct_predictions']} / {results['overall']['total_predictions']}",
            f"Coverage: {results['overview']['coverage']:.3f}",
            "",
            "Per-Word Results:",
            "-" * 20,
        ]

        per_word = results['per_word'].sort_values('accuracy', ascending=False)
        for _, row in per_word.iterrows():
            report_lines.append(
                f"{row['word']:12} | {row['accuracy']:5.3f} | {row['correct']:3d}/{row['instances']:3d} instances | "
                f"{row['senses_in_gold']} gold senses"
            )

        if 'error_analysis' in results:
            report_lines.extend([
                "",
                "Error Analysis:",
                "-" * 15,
                f"Most confused sense pairs: {len(results['error_analysis'].get('confusion_pairs', []))} pairs",
                f"Words with highest error rates: {results['error_analysis'].get('high_error_words', [])[:5]}",
            ])

        if 'confidence' in results:
            conf = results['confidence']
            report_lines.extend([
                "",
                "Confidence Analysis:",
                "-" * 20,
                f"Confidence-accuracy correlation: {conf.get('confidence_accuracy_correlation', 0):.3f}",
                f"Avg confidence (correct): {conf.get('avg_confidence_correct', 0):.3f}",
                f"Avg confidence (incorrect): {conf.get('avg_confidence_incorrect', 0):.3f}",
            ])

        report_text = "\n".join(report_lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_file}")

        return report_text