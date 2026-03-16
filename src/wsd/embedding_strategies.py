"""
Helper functions for working with sense text representations.

Handles both baseline and LLM-enhanced sense files, and provides
utilities for creating different text representations for embedding.
"""

from typing import List, Literal
import pandas as pd
import numpy as np


RepresentationType = Literal[
    "baseline",           # GermaNet template text
    "llm_definition",     # LLM-generated definition only
    "core_gloss",         # Short denotative gloss — used for clustering only (not WSD)
    "llm_example_1",      # Single LLM example
    "llm_examples_concat", # All 5 examples concatenated
    "llm_examples_avg",   # Average embedding of 5 examples (requires embedder)
    "weighted"            # 0.5 * definition + 0.5 * avg(examples) - BEST METHOD (validated)
]


def get_text_for_embedding(
    sense_df: pd.DataFrame,
    representation: RepresentationType = "baseline"
) -> List[str]:
    """Extract text for embedding based on representation type.

    Args:
        sense_df: DataFrame with sense data (baseline or enhanced)
        representation: Which text representation to use

    Returns:
        List of text strings, one per sense

    Raises:
        ValueError: If required columns missing
    """
    if representation == "baseline":
        if 'text_for_embedding' not in sense_df.columns:
            raise ValueError("baseline requires 'text_for_embedding' column")
        return sense_df['text_for_embedding'].fillna("").tolist()

    elif representation == "llm_definition":
        if 'llm_definition' not in sense_df.columns:
            raise ValueError("llm_definition requires enhanced sense file")
        return sense_df['llm_definition'].fillna("").tolist()

    elif representation == "llm_example_1":
        if 'llm_example_1' not in sense_df.columns:
            raise ValueError("llm_example_1 requires enhanced sense file")
        return sense_df['llm_example_1'].fillna("").tolist()

    elif representation == "llm_examples_concat":
        # Concatenate all 5 examples with separator
        example_cols = [f'llm_example_{i}' for i in range(1, 6)]
        missing = [col for col in example_cols if col not in sense_df.columns]
        if missing:
            raise ValueError(f"llm_examples_concat requires columns: {missing}")

        texts = []
        for _, row in sense_df.iterrows():
            examples = [row[col] for col in example_cols if pd.notna(row[col]) and row[col]]
            texts.append(" ".join(examples))
        return texts

    else:
        raise ValueError(f"Unsupported representation: {representation}. "
                        f"Note: 'llm_examples_avg' and 'weighted' require embedder.")


def embed_with_representation(
    sense_df: pd.DataFrame,
    embedder,  # Embedder instance from wsd module
    representation: RepresentationType = "baseline",
    show_progress: bool = True,
    passage_prefix: str = "",
) -> np.ndarray:
    """Embed senses using specified representation.

    Handles special cases like averaging multiple examples or weighted combinations.

    Args:
        sense_df: DataFrame with sense data
        embedder: Embedder instance (from src.wsd.embedder)
        representation: Which text representation to use
        show_progress: Show progress bar
        passage_prefix: Prefix to prepend to all sense texts (e.g. "passage: " for E5 models)

    Returns:
        Numpy array of embeddings (n_senses, embedding_dim)
    """
    def _prefix(texts: List[str]) -> List[str]:
        return [f"{passage_prefix}{t}" for t in texts] if passage_prefix else texts

    # Simple cases: single text per sense
    if representation in ["baseline", "llm_definition", "core_gloss", "llm_example_1", "llm_examples_concat"]:
        if representation == "core_gloss":
            if "core_gloss" not in sense_df.columns:
                raise ValueError("core_gloss requires enhanced sense file with core_gloss column")
            texts = sense_df["core_gloss"].fillna("").tolist()
        else:
            texts = get_text_for_embedding(sense_df, representation)
        return embedder.encode(_prefix(texts), show_progress=show_progress)

    # Special case: average of 5 examples
    elif representation == "llm_examples_avg":
        example_cols = [f'llm_example_{i}' for i in range(1, 6)]
        missing = [col for col in example_cols if col not in sense_df.columns]
        if missing:
            raise ValueError(f"llm_examples_avg requires columns: {missing}")

        embeddings = []
        for _, row in sense_df.iterrows():
            examples = [row[col] for col in example_cols if pd.notna(row[col]) and row[col]]
            if not examples:
                example_embeddings = embedder.encode(_prefix([""]), show_progress=False)
            else:
                example_embeddings = embedder.encode(_prefix(examples), show_progress=False)

            avg_embedding = np.mean(example_embeddings, axis=0)
            embeddings.append(avg_embedding)

        return np.array(embeddings)

    # Special case: weighted combination (BEST METHOD — 54.8% accuracy)
    # 0.5 * llm_definition + 0.5 * avg(examples), α=0.5 validated via grid search
    # Note: core_gloss was tested as definition component but scored lower (best 51.6% at α=0.3)
    # Verbose llm_definition retained for WSD; core_gloss used separately for clustering only
    elif representation == "weighted":
        if 'llm_definition' not in sense_df.columns:
            raise ValueError("weighted requires enhanced sense file with llm_definition")

        # Embed definitions
        definitions = sense_df['llm_definition'].fillna("").tolist()
        def_embeddings = embedder.encode(_prefix(definitions), show_progress=show_progress)

        # Embed and average examples
        example_cols = [f'llm_example_{i}' for i in range(1, 6)]
        example_embeddings_list = []

        for _, row in sense_df.iterrows():
            examples = [row[col] for col in example_cols if pd.notna(row[col]) and row[col]]
            if not examples:
                example_embeddings_list.append(np.zeros_like(def_embeddings[0]))
            else:
                example_embs = embedder.encode(_prefix(examples), show_progress=False)
                avg_example_emb = np.mean(example_embs, axis=0)
                example_embeddings_list.append(avg_example_emb)

        example_embeddings = np.array(example_embeddings_list)

        # Weighted combination: 0.5 * definition + 0.5 * avg(examples)
        return 0.5 * def_embeddings + 0.5 * example_embeddings

    else:
        raise ValueError(f"Unknown representation: {representation}")


def validate_sense_file(sense_df: pd.DataFrame, representation: RepresentationType) -> bool:
    """Check if sense file has required columns for representation.

    Args:
        sense_df: DataFrame to validate
        representation: Representation type to check for

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_cols = {
        "baseline": ["text_for_embedding"],
        "llm_definition": ["llm_definition"],
        "core_gloss": ["core_gloss"],
        "llm_example_1": ["llm_example_1"],
        "llm_examples_concat": [f"llm_example_{i}" for i in range(1, 6)],
        "llm_examples_avg": [f"llm_example_{i}" for i in range(1, 6)],
        "weighted": ["llm_definition"] + [f"llm_example_{i}" for i in range(1, 6)],
    }

    if representation not in required_cols:
        raise ValueError(f"Unknown representation: {representation}")

    missing = [col for col in required_cols[representation] if col not in sense_df.columns]

    if missing:
        raise ValueError(
            f"Sense file missing columns for '{representation}' representation: {missing}\n"
            f"You may need to run: python scripts/data/enrich_senses_with_llm.py"
        )

    return True


def get_available_representations(sense_df: pd.DataFrame) -> List[str]:
    """List which representations are available for this sense file.

    Args:
        sense_df: DataFrame to check

    Returns:
        List of available representation names
    """
    available = []

    for rep in ["baseline", "llm_definition", "core_gloss", "llm_example_1",
                "llm_examples_concat", "llm_examples_avg", "weighted"]:
        try:
            validate_sense_file(sense_df, rep)
            available.append(rep)
        except ValueError:
            pass

    return available
