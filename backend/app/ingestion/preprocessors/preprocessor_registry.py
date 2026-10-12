"""
preprocessor_registry.py

Central registry for all preprocessing components used by
SupportMind AI.

Unlike LoaderRegistry, these components are executed in a
fixed pipeline order.

Author: SupportMind AI
"""

from app.ingestion.preprocessors.text_cleaner import TextCleaner
from app.ingestion.preprocessors.text_normalizer import TextNormalizer
from app.ingestion.preprocessors.language_detector import LanguageDetector
from app.ingestion.preprocessors.sentence_splitter import SentenceSplitter
from app.ingestion.preprocessors.word_tokenizer import WordTokenizer
from app.ingestion.preprocessors.stopword_remover import StopwordRemover
from app.ingestion.preprocessors.spell_corrector import SpellCorrector
from app.ingestion.preprocessors.lemmatizer import Lemmatizer
from app.ingestion.preprocessors.stemmer import Stemmer


class PreprocessorRegistry:
    """
    Stores all preprocessing components.

    Components are instantiated once and reused
    throughout the application.
    """

    def __init__(self):

        self.text_cleaner = TextCleaner()

        self.text_normalizer = TextNormalizer()

        self.language_detector = LanguageDetector()

        self.sentence_splitter = SentenceSplitter()

        self.word_tokenizer = WordTokenizer()

        self.stopword_remover = StopwordRemover()

        self.spell_corrector = SpellCorrector()

        self.lemmatizer = Lemmatizer()

        self.stemmer = Stemmer()

    def get_text_cleaner(self):
        return self.text_cleaner

    def get_text_normalizer(self):
        return self.text_normalizer

    def get_language_detector(self):
        return self.language_detector

    def get_sentence_splitter(self):
        return self.sentence_splitter

    def get_word_tokenizer(self):
        return self.word_tokenizer

    def get_stopword_remover(self):
        return self.stopword_remover

    def get_spell_corrector(self):
        return self.spell_corrector

    def get_lemmatizer(self):
        return self.lemmatizer

    def get_stemmer(self):
        return self.stemmer