"""
Production Word Tokenizer

Uses spaCy's industrial-strength tokenizer.

SupportMind AI
"""

from typing import List

import spacy


class WordTokenizer:
    """
    Wrapper around spaCy tokenizer.

    Long documents are processed in safe character blocks
    to avoid spaCy's maximum document length limitation.
    """

    MAX_TEXT_LENGTH = 800_000

    def __init__(self):

        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=[
                "parser",
                "ner",
                "lemmatizer",
                "attribute_ruler",
                "tagger",
            ],
        )

    def tokenize(
        self,
        text: str,
    ) -> List[str]:

        if not text:
            return []

        if len(text) <= self.MAX_TEXT_LENGTH:

            doc = self.nlp(text)

            return [
                token.text
                for token in doc
            ]

        tokens = []

        for start in range(
            0,
            len(text),
            self.MAX_TEXT_LENGTH,
        ):

            block = text[
                start:
                start + self.MAX_TEXT_LENGTH
            ]

            doc = self.nlp(block)

            tokens.extend(
                token.text
                for token in doc
            )

        return tokens