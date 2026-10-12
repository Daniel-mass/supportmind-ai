#exceptions.py
"""
Custom exceptions for the vector store layer.

These exceptions provide provider-independent error types
that can be raised by any vector database implementation
(Qdrant, Pinecone, Milvus, Weaviate, etc.).

SupportMind AI
"""

from __future__ import annotations


class VectorStoreError(Exception):
    """
    Base exception for all vector store errors.
    """


class VectorStoreConnectionError(VectorStoreError):
    """
    Raised when the vector database
    cannot be reached.
    """


class VectorCollectionNotFoundError(VectorStoreError):
    """
    Raised when a requested collection
    does not exist.
    """


class VectorCollectionAlreadyExistsError(VectorStoreError):
    """
    Raised when attempting to create
    an already existing collection.
    """


class VectorInsertionError(VectorStoreError):
    """
    Raised when inserting vectors fails.
    """


class VectorSearchError(VectorStoreError):
    """
    Raised when vector search fails.
    """


class VectorDeletionError(VectorStoreError):
    """
    Raised when deleting vectors fails.
    """


class VectorUpdateError(VectorStoreError):
    """
    Raised when updating vectors fails.
    """


class VectorConfigurationError(VectorStoreError):
    """
    Raised when the vector store
    configuration is invalid.
    """


class InvalidVectorDimensionError(VectorStoreError):
    """
    Raised when embedding dimensions
    do not match the collection.
    """


class InvalidVectorFilterError(VectorStoreError):
    """
    Raised when an unsupported
    metadata filter is supplied.
    """


class UnsupportedVectorOperationError(VectorStoreError):
    """
    Raised when the selected provider
    does not support an operation.
    """