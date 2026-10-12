#backend/app/ingestion/preprocessors/stopword_remover.py
from typing import Iterable


class StopwordRemover:
    """
    Removes common stopwords from token lists.

    Intended mainly for classical NLP pipelines such as:
    - TF-IDF
    - BM25
    - Topic Modeling
    - Traditional ML classifiers

    Not typically used before transformer models like
    BERT or GPT, where stopwords contribute to context.
    """

    DEFAULT_STOPWORDS = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "am",
        "be",
        "been",
        "being",
        "to",
        "of",
        "for",
        "at",
        "on",
        "in",
        "into",
        "with",
        "by",
        "from",
        "this",
        "that",
        "these",
        "those",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "as",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "their",
        "his",
        "her",
        "we",
        "our",
        "you",
        "your",
        "i",
        "me",
        "my",
        "mine",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "about",
        "after",
        "before",
        "during",
        "through",
        "over",
        "under",
        "again",
        "once",
        "very",
        "also",
        "just",
        "there",
        "here",
        "all",
        "any",
        "each",
        "few",
        "more",
        "most",
        "some",
        "such"
    }

    # Keep these because they change meaning
    NEGATION_WORDS = {
        "no",
        "not",
        "never",
        "none",
        "cannot",
        "can't",
        "won't",
        "don't",
        "didn't",
        "isn't",
        "aren't"
    }

    def __init__(
        self,
        extra_stopwords: Iterable[str] | None = None,
        preserve_negation: bool = True,
    ):

        self.stopwords = set(self.DEFAULT_STOPWORDS)

        if extra_stopwords:
            self.stopwords.update(word.lower() for word in extra_stopwords)

        if preserve_negation:
            self.stopwords -= self.NEGATION_WORDS

    def remove(self, tokens: list[str]) -> list[str]:
        """
        Removes stopwords while preserving punctuation and numbers.
        """

        cleaned = []

        for token in tokens:

            lower = token.lower()

            if lower in self.stopwords:
                continue

            cleaned.append(token)

        return cleaned