"""Generate sense text representations using LLMs."""

import logging
from pathlib import Path
from .client import LLMClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
PROMPTS_DIR = PROJECT_ROOT / "config" / "prompts"
POS_TO_GERMAN = {"NOUN": "Nomen", "VERB": "Verb", "ADJ": "Adjektiv"}


def load_prompt(name: str) -> str:
    """Load a prompt template from a file."""
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding='utf-8').strip()


class SenseTextGenerator:
    """Generate definitions and example sentences for word senses."""

    def __init__(self, client: LLMClient):
        self.client = client

    def generate_core_gloss(
        self,
        lemma: str,
        pos: str,
        sense_description: str,
        sibling_descriptions: list[str] | None = None,
    ) -> str:
        """Generate a short core gloss (up to 12 words, no boilerplate) for a word sense.

        The gloss should be compact but grounded: key nouns from the sense description
        (objects, tools, body parts, contexts) must be preserved. Abstract paraphrases
        like "etwas treffen oder bewegen" that apply to many senses are explicitly
        discouraged by the prompt.

        Args:
            sibling_descriptions: Baseline descriptions of other senses for the same lemma.
                When provided, the prompt instructs the model to distinguish this sense
                via a concrete object or specific context — not via abstract reformulation.
        """
        siblings = [d for d in (sibling_descriptions or []) if d.strip()]
        if siblings:
            lines = "\n".join(f"- {d}" for d in siblings)
            other_senses_block = (
                f'Andere Bedeutungen von "{lemma}" (nur zur Orientierung — nicht aufzählen, '
                f"sondern diese eine Bedeutung durch konkretes Objekt oder Kontext unterscheiden):\n"
                f"{lines}\n"
            )
        else:
            other_senses_block = ""

        prompt = load_prompt('core_gloss').format(
            lemma=lemma,
            pos_german=POS_TO_GERMAN.get(pos, pos),
            sense_description=sense_description,
            other_senses_block=other_senses_block,
        )
        text = self.client.generate(prompt, max_tokens=50)
        text = text.strip().strip('"').strip("'").rstrip('.').strip()
        return text

    def generate_definition(self, lemma: str, pos: str, sense_description: str) -> str:
        """Generate a natural language definition for a word sense."""
        prompt = load_prompt('definition').format(
            lemma=lemma,
            pos_german=POS_TO_GERMAN.get(pos, pos),
            sense_description=sense_description
        )
        return self.client.generate(prompt, max_tokens=150)

    def generate_examples(self, lemma: str, pos: str, sense_description: str, n: int = 5) -> list[str]:
        """Generate multiple example sentences for a word sense."""
        prompt = load_prompt('examples_multiple').format(
            lemma=lemma,
            pos_german=POS_TO_GERMAN.get(pos, pos),
            sense_description=sense_description,
            n=n
        )

        response = self.client.generate(prompt, max_tokens=300)

        # Parse numbered list
        examples = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                text = line.lstrip('0123456789').lstrip('.):- ').strip()
                if text:
                    examples.append(text)

        # Fallback: if parsing failed, split by newlines
        if len(examples) < n and '\n' in response:
            examples = [line.strip() for line in response.split('\n') if line.strip()]

        return examples[:n]

    def generate_all(
        self,
        lemma: str,
        pos: str,
        sense_description: str,
        n_examples: int = 5
    ) -> dict[str, object]:
        """Generate all text variants for a sense."""
        logger.info(f"Generating variants for {lemma}")

        return {
            'core_gloss': self.generate_core_gloss(lemma, pos, sense_description),
            'definition': self.generate_definition(lemma, pos, sense_description),
            'examples_5': self.generate_examples(lemma, pos, sense_description, n=n_examples)
        }