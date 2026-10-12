#backend/app/ingestion/preprocessors/stemmer.py
from nltk.stem import SnowballStemmer


class Stemmer:
    """
    Applies Snowball stemming to tokens.

    Stemming is useful for classical information retrieval
    (BM25, TF-IDF) where related word forms should match.

    Example
    -------
    running  -> run
    payments -> payment
    processed -> process
    """

    def __init__(self, language: str = "english"):
        self.stemmer = SnowballStemmer(language)

    def stem(self, tokens: list[str]) -> list[str]:
        """
        Stem a list of tokens.

        Parameters
        ----------
        tokens : list[str]

        Returns
        -------
        list[str]
        """

        stemmed = []

        for token in tokens:

            if token.isalpha():
                stemmed.append(self.stemmer.stem(token.lower()))
            else:
                stemmed.append(token)

        return stemmed