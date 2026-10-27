"""Builds the template baseline sense texts of senses_base.csv.

The script that originally produced senses_base.csv (Dec 2025) was never committed.
This is a reconstruction from the GermaNet base tables, verified to reproduce all
2,022 rows of senses_base.csv exactly (text, sense order, sense_frequency,
word count, has_examples). Re-check with
`python scripts/dwug/prepare_dwug_senses.py --verify-baseline`.

Rule per (lemma, synset) row of lemma_senses.csv, in table order:
- synset has GermaNet examples  -> text_source "example", text = sense_texts.text_for_embedding
- otherwise                     -> text_source "direct_relations", template sentences:
    "{L}: {paraphrase}"                              if a paraphrase exists
    "{L} wird auch genannt: {variants}"              if the synset has variants
    "{L} ist eine Art von {H}"                       first 2 hypernyms
    "{L} hat {M}" / "{M1} und {M2}" / "{M1}, {M2} und andere Teile"   component meronyms
    "{L} ist verwandt mit {R}"                       first 2 outgoing is_related_to
    "{L} ist ein Begriff aus {class}"
  joined by ". ". Related synsets are named by the first lemma in sense_texts.lemmas.
"""

import pandas as pd

BASE_COLUMNS = [
    'lemma', 'pos', 'synset_id', 'class', 'paraphrase', 'text_for_embedding',
    'text_source', 'sense_frequency', 'embedding_word_count', 'has_examples',
]


class BaselineTextBuilder:
    """Rebuilds senses_base.csv rows for arbitrary lemmas."""

    MAX_HYPERNYMS = 2
    MAX_RELATED = 2

    def __init__(self, lemma_senses_df: pd.DataFrame, sense_texts_df: pd.DataFrame,
                 relations_df: pd.DataFrame):
        self.lemma_senses = lemma_senses_df.fillna('')
        self.sense_texts = sense_texts_df.fillna('').set_index('synset_id')
        self.outgoing: dict[tuple[str, str], list[str]] = {}
        for rel_type, src, dst in zip(relations_df['relation_type'],
                                      relations_df['from_synset'], relations_df['to_synset']):
            self.outgoing.setdefault((src, rel_type), []).append(dst)

    def _name(self, synset_id: str) -> str:
        return self.sense_texts.at[synset_id, 'lemmas'].split(' ; ')[0]

    def _targets(self, synset_id: str, rel_type: str) -> list[str]:
        return self.outgoing.get((synset_id, rel_type), [])

    def relation_text(self, lemma: str, synset_id: str, paraphrase: str, synset_class: str) -> str:
        parts = []
        if paraphrase.strip():
            parts.append(f"{lemma}: {paraphrase.strip()}")
        variants = self.sense_texts.at[synset_id, 'variants']
        if variants:
            parts.append(f"{lemma} wird auch genannt: {variants}")
        for hyper in self._targets(synset_id, 'has_hypernym')[:self.MAX_HYPERNYMS]:
            parts.append(f"{lemma} ist eine Art von {self._name(hyper)}")
        meronyms = [self._name(m) for m in self._targets(synset_id, 'has_component_meronym')]
        if len(meronyms) == 1:
            parts.append(f"{lemma} hat {meronyms[0]}")
        elif len(meronyms) == 2:
            parts.append(f"{lemma} hat {meronyms[0]} und {meronyms[1]}")
        elif len(meronyms) > 2:
            parts.append(f"{lemma} hat {meronyms[0]}, {meronyms[1]} und andere Teile")
        for rel in self._targets(synset_id, 'is_related_to')[:self.MAX_RELATED]:
            parts.append(f"{lemma} ist verwandt mit {self._name(rel)}")
        parts.append(f"{lemma} ist ein Begriff aus {synset_class}")
        return '. '.join(parts)

    def build(self, lemmas: list[str]) -> pd.DataFrame:
        """Return senses_base-format rows for the given lemmas (GermaNet orthForm)."""
        rows = []
        wanted = set(lemmas)
        subset = self.lemma_senses[self.lemma_senses['lemma'].isin(wanted)]
        for lemma, group in subset.groupby('lemma', sort=True):
            for rank, row in enumerate(group.to_dict('records'), start=1):
                sid = row['synset_id']
                has_examples = self.sense_texts.at[sid, 'examples'] != ''
                if has_examples:
                    text, source = self.sense_texts.at[sid, 'text_for_embedding'], 'example'
                else:
                    text = self.relation_text(lemma, sid, row['paraphrase'], row['class'])
                    source = 'direct_relations'
                rows.append({
                    'lemma': lemma,
                    'pos': row['pos'],
                    'synset_id': sid,
                    'class': row['class'],
                    'paraphrase': row['paraphrase'],
                    'text_for_embedding': text,
                    'text_source': source,
                    'sense_frequency': rank,
                    'embedding_word_count': len(text.split()),
                    'has_examples': has_examples,
                })
        return pd.DataFrame(rows, columns=BASE_COLUMNS)
