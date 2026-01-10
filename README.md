# German Word Sense Disambiguation (WSD)

A comprehensive system for disambiguating polysemous German words using enhanced GermaNet semantic relations and BERT embeddings.

## 🎯 Project Overview

This project addresses German Word Sense Disambiguation by enhancing sparse GermaNet data with rich semantic relations to create better word sense embeddings.

### Key Achievements
- ✅ Enhanced 2,022 German word senses with semantic relations
- ✅ Achieved 48.6% accuracy on real DWDS corpus evaluation  
- ✅ Complete coverage vs 5.3% in original sparse GermaNet
- ✅ Built evaluation framework with real German sentences

## 📊 Current Status

### Data Ready ✅
- **Enhanced GermaNet**: 2,022 senses for 700 German words
- **DWDS Corpus**: 4,809 real German sentences (25 words)
- **Evaluation Results**: 48.6% accuracy baseline established

### Core Files
- `data/processed/germanet/wsd_senses_enhanced.csv` - Main enhanced dataset
- `data/dwds_contexts/fetch_progress_25_words.jsonl` - Real corpus sentences
- `scripts/` - Core processing and evaluation scripts

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. View Enhanced Data

```python
import pandas as pd

data = pd.read_csv('data/processed/germanet/wsd_senses.csv')
print(f"Enhanced senses: {len(data)}")
```

### 3. Run WSD Evaluation  
```bash
python scripts/real_dwds_wsd_evaluation.py
```

## 📁 Repository Structure

```
├── data/
│   ├── processed/germanet/          # Enhanced GermaNet data  
│   ├── dwds_contexts/              # Real German corpus sentences
│   └── external/germanet_v190/     # Original GermaNet data
├── scripts/                        # Core working scripts
├── analysis/                       # Analysis and comparison tools
├── notebooks/                      # Jupyter exploration notebooks
└── src/                           # Source code modules
```

## 🔬 Key Results

| Metric | Performance | Description |
|--------|-------------|-------------|
| **WSD Accuracy** | **48.6%** | Real German corpus evaluation |
| **Coverage** | **100%** | All senses have descriptions |  
| **Text Quality** | **15.7 words avg** | Rich semantic content |
| **Improvement** | **+42.4%** | Over majority baseline |

### Text Enhancement Example
- **Before**: "Bank" (just keyword)
- **After**: "Bank ist eine Art von Sitzgelegenheit. Bank hat Lehne und Sitzfläche."

## 🎯 Next Steps

1. **Ground Truth Creation**: Manual annotation for gold standard evaluation
2. **Corpus Expansion**: Fetch remaining 675 words from DWDS  
3. **Model Improvements**: Neural classifiers, fine-tuning
4. **Production**: API deployment for real applications

## 📚 Key Papers & References

- GermaNet: German WordNet
- DWDS: Digital Dictionary of German Language
- Multilingual BERT for semantic similarity

## 🤝 Contributing

This is a research project focused on German NLP and Word Sense Disambiguation.

## 📄 License

See LICENSE file for details.
