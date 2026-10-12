"""
preprocessing_pipeline.py

Production preprocessing pipeline.

Runs all preprocessing components in the correct order.

Author: SupportMind AI
"""

from app.ingestion.preprocessors.preprocessor_registry import (
    PreprocessorRegistry,
)


class PreprocessingPipeline:
    """
    Executes the complete preprocessing workflow.
    """

    def __init__(self):

        self.registry = PreprocessorRegistry()

    def run(
        self,
        text: str,
        lowercase: bool = True,
        remove_stopwords: bool = False,
        spell_correction: bool = False,
        lemmatize: bool = True,
        stem: bool = False,
    ) -> dict:
        """
        Execute preprocessing pipeline.

        Returns
        -------
        dict
            Contains intermediate and final outputs.
        """

        # ----------------------------------------
        # Stage 1
        # Clean raw text
        # ----------------------------------------

        cleaned_text = (
            self.registry
            .get_text_cleaner()
            .clean(text)
        )

        # ----------------------------------------
        # Stage 2
        # Normalize text
        # ----------------------------------------

        normalized_text = (
            self.registry
            .get_text_normalizer()
            .normalize(
                cleaned_text,
                lowercase=lowercase,
            )
        )

        # ----------------------------------------
        # Stage 3
        # Detect language
        # ----------------------------------------

        language = (
            self.registry
            .get_language_detector()
            .detect(normalized_text)
        )

        # ----------------------------------------
        # Stage 4
        # Split into sentences
        # ----------------------------------------

        sentences = (
            self.registry
            .get_sentence_splitter()
            .split(normalized_text)
        )

        # ----------------------------------------
        # Stage 5
        # Tokenization
        # ----------------------------------------

        tokens = (
            self.registry
            .get_word_tokenizer()
            .tokenize(normalized_text)
        )

        # ----------------------------------------
        # Stage 6
        # Stopword Removal (Optional)
        # ----------------------------------------

        if remove_stopwords:

            tokens = (
                self.registry
                .get_stopword_remover()
                .remove(tokens)
            )

        # ----------------------------------------
        # Stage 7
        # Spell Correction (Optional)
        # ----------------------------------------

        if spell_correction:

            tokens = (
                self.registry
                .get_spell_corrector()
                .correct(tokens)
            )

        # ----------------------------------------
        # Stage 8
        # Lemmatization (Optional)
        # ----------------------------------------

        if lemmatize:

            tokens = (
                self.registry
                .get_lemmatizer()
                .lemmatize(tokens)
            )

        # ----------------------------------------
        # Stage 9
        # Stemming (Optional)
        # ----------------------------------------

        if stem:

            tokens = (
                self.registry
                .get_stemmer()
                .stem(tokens)
            )

        return {

            "cleaned_text": cleaned_text,

            "normalized_text": normalized_text,

            "language": language,

            "sentences": sentences,

            "tokens": tokens,

        }