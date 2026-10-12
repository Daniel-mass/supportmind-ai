"""
orchestration_config.py

Configuration for SupportMind AI end-to-end
document ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrchestrationConfig:

    embedding_provider: str = "fastembed"

    vector_provider: str = "qdrant"

    collection_name: str = "supportmind_enterprise_knowledge"

    chunking_strategy: str = "sentence_window"

    lowercase: bool = True

    remove_stopwords: bool = False

    spell_correction: bool = False

    lemmatize: bool = False

    stem: bool = False