#backend/app/ingestion/preprocessors/spell_corrector.py
from spellchecker import SpellChecker


class SpellCorrector:
    """
    Corrects spelling mistakes in tokenized text.

    Uses pyspellchecker for lightweight spell correction
    and supports adding custom enterprise vocabulary.
    """

    def __init__(self):

        self.spell = SpellChecker()

        # Enterprise-specific words that should never be corrected.
        custom_words = [
            "supportmind",
            "neo4j",
            "qdrant",
            "langchain",
            "langgraph",
            "llamaindex",
            "fastapi",
            "autogen",
            "ragas",
            "deepeval",
            "mlflow",
            "graphrag",
            "bm25"
        ]

        self.spell.word_frequency.load_words(custom_words)

    def correct(self, tokens: list[str]) -> list[str]:
        """
        Correct spelling mistakes in a list of tokens.

        Parameters
        ----------
        tokens : list[str]

        Returns
        -------
        list[str]
        """

        corrected = []

        for token in tokens:

            # Ignore punctuation/numbers
            if not token.isalpha():
                corrected.append(token)
                continue

            suggestion = self.spell.correction(token.lower())

            corrected.append(suggestion if suggestion else token)

        return corrected