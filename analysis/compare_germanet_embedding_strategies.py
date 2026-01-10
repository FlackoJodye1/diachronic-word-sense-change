#!/usr/bin/env python3
"""
Compare different GermaNet embedding strategies to see which produces better semantic representations.
This doesn't require ground truth - just looks at embedding quality.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Error: sentence-transformers required")
    exit(1)

# Paths
DATA_DIR = Path("data")
GERMANET_DIR = DATA_DIR / "processed" / "germanet"
GERMANET_BASE_DIR = DATA_DIR / "processed" / "germanet_base"

def load_model():
    """Load BERT model once"""
    print("Loading multilingual BERT model...")
    return SentenceTransformer('distiluse-base-multilingual-cased')

def create_embedding_variants():
    """Create different text variants for comparison"""
    print("Creating different embedding text variants...")
    
    # Load enhanced data
    enhanced_data = pd.read_csv(GERMANET_DIR / "wsd_senses_enhanced.csv")
    
    # Load raw data for comparison
    sense_texts = pd.read_csv(GERMANET_BASE_DIR / "sysnets" / "sense_texts.csv")
    
    variants = {}
    
    # Variant 1: Our enhanced semantic relations
    variants['enhanced_relations'] = {
        'data': enhanced_data,
        'text_column': 'text_for_embedding',
        'description': 'Enhanced with semantic relations (our method)'
    }
    
    # Variant 2: Only examples (no relations)
    examples_only = enhanced_data[enhanced_data['text_source'] == 'example'].copy()
    if len(examples_only) > 0:
        variants['examples_only'] = {
            'data': examples_only,
            'text_column': 'text_for_embedding', 
            'description': 'Only real usage examples'
        }
    
    # Variant 3: Only relations (no examples)
    relations_only = enhanced_data[enhanced_data['text_source'] == 'direct_relations'].copy()
    if len(relations_only) > 0:
        variants['relations_only'] = {
            'data': relations_only,
            'text_column': 'text_for_embedding',
            'description': 'Only semantic relations'
        }
    
    # Variant 4: Just lemma + class (minimal)
    minimal_data = enhanced_data.copy()
    minimal_data['minimal_text'] = minimal_data['lemma'] + ' ist ein Begriff aus ' + minimal_data['class']
    variants['minimal'] = {
        'data': minimal_data,
        'text_column': 'minimal_text',
        'description': 'Minimal: lemma + semantic class only'
    }
    
    # Variant 5: Original sparse data (for comparison)
    original_merged = enhanced_data.merge(
        sense_texts[['synset_id', 'paraphrase']], 
        on='synset_id', 
        how='left'
    )
    
    # Use paraphrase if available, otherwise lemma
    def create_original_text(row):
        paraphrase = row.get('paraphrase', '')
        if pd.notna(paraphrase) and str(paraphrase).strip():
            return str(paraphrase).strip()
        else:
            return f"{row['lemma']} ({row['class']})"
    
    original_merged['original_text'] = original_merged.apply(create_original_text, axis=1)
    
    variants['original_sparse'] = {
        'data': original_merged,
        'text_column': 'original_text',
        'description': 'Original sparse GermaNet data'
    }
    
    print(f"Created {len(variants)} embedding variants:")
    for name, info in variants.items():
        print(f"  {name}: {len(info['data'])} senses - {info['description']}")
    
    return variants

def evaluate_semantic_coherence(embeddings, data, variant_name):
    """Evaluate how semantically coherent embeddings are"""
    print(f"Evaluating semantic coherence for {variant_name}...")
    
    results = {}
    
    # 1. Same-word similarity (senses of same word should be more similar)
    same_word_sims = []
    diff_word_sims = []
    
    for i in range(min(len(embeddings), 500)):  # Sample for efficiency
        for j in range(i+1, min(len(embeddings), 500)):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            
            if data.iloc[i]['lemma'] == data.iloc[j]['lemma']:
                same_word_sims.append(sim)
            else:
                diff_word_sims.append(sim)
    
    results['same_word_similarity'] = np.mean(same_word_sims) if same_word_sims else 0
    results['diff_word_similarity'] = np.mean(diff_word_sims) if diff_word_sims else 0
    results['discrimination_ratio'] = (results['same_word_similarity'] / 
                                      results['diff_word_similarity']) if results['diff_word_similarity'] > 0 else 0
    
    # 2. Semantic class coherence (same class should cluster together)
    class_sims = defaultdict(list)
    
    # Sample by semantic class
    for class_name in data['class'].unique()[:10]:  # Top 10 classes
        class_data = data[data['class'] == class_name]
        if len(class_data) >= 3:
            class_indices = class_data.index.tolist()[:20]  # Max 20 per class
            class_embeddings = [embeddings[i] for i in class_indices if i < len(embeddings)]
            
            if len(class_embeddings) >= 2:
                # Calculate intra-class similarities
                class_similarities = []
                for i in range(len(class_embeddings)):
                    for j in range(i+1, len(class_embeddings)):
                        sim = cosine_similarity([class_embeddings[i]], [class_embeddings[j]])[0][0]
                        class_similarities.append(sim)
                
                if class_similarities:
                    class_sims[class_name] = np.mean(class_similarities)
    
    results['class_coherence'] = np.mean(list(class_sims.values())) if class_sims else 0
    
    # 3. Embedding diversity (shouldn't be too similar across all)
    sample_embeddings = embeddings[:200] if len(embeddings) > 200 else embeddings
    all_similarities = []
    for i in range(len(sample_embeddings)):
        for j in range(i+1, len(sample_embeddings)):
            sim = cosine_similarity([sample_embeddings[i]], [sample_embeddings[j]])[0][0]
            all_similarities.append(sim)
    
    results['overall_similarity'] = np.mean(all_similarities)
    results['similarity_std'] = np.std(all_similarities)
    
    # 4. Clustering quality
    if len(embeddings) >= 10:
        n_clusters = min(10, len(set(data['class'].dropna())))
        if n_clusters >= 2:
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(sample_embeddings)
                silhouette_avg = silhouette_score(sample_embeddings, cluster_labels)
                results['silhouette_score'] = silhouette_avg
            except:
                results['silhouette_score'] = 0
        else:
            results['silhouette_score'] = 0
    else:
        results['silhouette_score'] = 0
    
    return results

def evaluate_text_quality(data, text_column):
    """Evaluate the quality of the text itself"""
    texts = data[text_column].fillna('')
    
    results = {
        'avg_length': np.mean([len(text.split()) for text in texts]),
        'coverage': (texts != '').sum() / len(texts),
        'unique_texts': len(set(texts)) / len(texts),
        'avg_characters': np.mean([len(text) for text in texts])
    }
    
    return results

def compare_variants(variants, model):
    """Compare all variants"""
    print(f"\n{'='*60}")
    print("🔬 COMPARING GERMANET EMBEDDING VARIANTS")
    print(f"{'='*60}")
    
    all_results = {}
    
    for variant_name, variant_info in variants.items():
        print(f"\n--- Processing {variant_name} ---")
        
        data = variant_info['data']
        text_column = variant_info['text_column']
        
        # Get texts and encode
        texts = data[text_column].fillna('').tolist()
        
        if not texts:
            print(f"No texts found for {variant_name}")
            continue
        
        print(f"Encoding {len(texts)} texts...")
        embeddings = model.encode(texts, show_progress_bar=False)
        
        # Evaluate
        semantic_results = evaluate_semantic_coherence(embeddings, data, variant_name)
        text_results = evaluate_text_quality(data, text_column)
        
        all_results[variant_name] = {
            'description': variant_info['description'],
            'num_senses': len(data),
            'semantic': semantic_results,
            'text_quality': text_results
        }
        
        print(f"✅ Completed {variant_name}")
    
    return all_results

def print_comparison_results(all_results):
    """Print detailed comparison"""
    print(f"\n{'='*80}")
    print("📊 GERMANET EMBEDDING VARIANT COMPARISON RESULTS")
    print(f"{'='*80}")
    
    # Create comparison table
    metrics = [
        ('Discrimination Ratio', 'semantic', 'discrimination_ratio', 'higher_better'),
        ('Same Word Similarity', 'semantic', 'same_word_similarity', 'higher_better'),
        ('Class Coherence', 'semantic', 'class_coherence', 'higher_better'),
        ('Silhouette Score', 'semantic', 'silhouette_score', 'higher_better'),
        ('Similarity Diversity', 'semantic', 'similarity_std', 'higher_better'),
        ('Text Coverage', 'text_quality', 'coverage', 'higher_better'),
        ('Avg Text Length', 'text_quality', 'avg_length', 'moderate_better'),
        ('Text Uniqueness', 'text_quality', 'unique_texts', 'higher_better')
    ]
    
    print(f"\n📋 DETAILED METRICS:")
    print(f"{'Variant':<20} {'Description':<35} {'Senses':<8} {'Discrim':<8} {'SameWord':<9} {'ClassCoh':<9} {'Silhouette':<11} {'Coverage':<9}")
    print("-" * 120)
    
    for variant_name, results in all_results.items():
        desc = results['description'][:35] if results['description'] else ''
        num_senses = results['num_senses']
        
        sem = results.get('semantic', {})
        text = results.get('text_quality', {})
        
        print(f"{variant_name:<20} {desc:<35} {num_senses:<8} "
              f"{sem.get('discrimination_ratio', 0):<8.3f} "
              f"{sem.get('same_word_similarity', 0):<9.3f} "
              f"{sem.get('class_coherence', 0):<9.3f} "
              f"{sem.get('silhouette_score', 0):<11.3f} "
              f"{text.get('coverage', 0):<9.3f}")
    
    # Find best performers
    print(f"\n🏆 BEST PERFORMERS:")
    
    key_metrics = [
        ('Best Discrimination', 'semantic', 'discrimination_ratio'),
        ('Best Class Coherence', 'semantic', 'class_coherence'), 
        ('Best Coverage', 'text_quality', 'coverage'),
        ('Best Silhouette', 'semantic', 'silhouette_score')
    ]
    
    for metric_name, category, key in key_metrics:
        best_variant = None
        best_score = -1
        
        for variant_name, results in all_results.items():
            score = results.get(category, {}).get(key, 0)
            if score > best_score:
                best_score = score
                best_variant = variant_name
        
        if best_variant:
            print(f"  {metric_name}: {best_variant} ({best_score:.3f})")
    
    # Overall recommendation
    print(f"\n💡 RECOMMENDATIONS:")
    
    # Score variants
    variant_scores = {}
    for variant_name, results in all_results.items():
        score = 0
        sem = results.get('semantic', {})
        text = results.get('text_quality', {})
        
        # Weight key metrics
        score += sem.get('discrimination_ratio', 0) * 3  # Most important
        score += sem.get('class_coherence', 0) * 2
        score += sem.get('silhouette_score', 0) * 2
        score += text.get('coverage', 0) * 1
        
        variant_scores[variant_name] = score
    
    # Sort by score
    ranked_variants = sorted(variant_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Ranking (higher is better):")
    for i, (variant_name, score) in enumerate(ranked_variants, 1):
        desc = all_results[variant_name]['description']
        print(f"  {i}. {variant_name}: {score:.3f} - {desc}")
    
    if ranked_variants:
        best_variant = ranked_variants[0][0]
        print(f"\n✅ RECOMMENDED: {best_variant}")
        print(f"   {all_results[best_variant]['description']}")

def main():
    """Main comparison function"""
    print("🔬 GERMANET EMBEDDING STRATEGY COMPARISON")
    print("=" * 60)
    
    # Load model
    model = load_model()
    
    # Create variants
    variants = create_embedding_variants()
    
    # Compare all variants
    results = compare_variants(variants, model)
    
    # Print results
    print_comparison_results(results)
    
    print(f"\n{'='*60}")
    print("🎯 This evaluation doesn't need ground truth!")
    print("   It measures intrinsic quality of semantic representations.")

if __name__ == "__main__":
    main()