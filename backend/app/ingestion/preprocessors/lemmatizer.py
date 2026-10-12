#backend/app/ingestion/preprocessors/lemmatizer.py
import spacy


class Lemmatizer:
    """
    Production-ready lemmatizer using spaCy.

    Converts words into their dictionary form (lemma).

    Example:
        running  -> run
        processed -> process
        children -> child
    """

    def __init__(self):

        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=[
                "ner",
                "parser",
                "textcat"
            ]
        )

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """
        Lemmatize a list of tokens.

        Parameters
        ----------
        tokens : list[str]

        Returns
        -------
        list[str]
        """

        doc = self.nlp(" ".join(tokens))

        lemmas = []

        for token in doc:

            if token.is_space:
                continue

            lemmas.append(token.lemma_)

        return lemmas