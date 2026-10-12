"""
utils.py

Provider-agnostic utility functions for the SupportMind AI
vector store layer.

Responsibilities
----------------
1. Validate and normalize collection names.
2. Validate and normalize provider names.
3. Validate vector values.
4. Validate vector dimensions.
5. Validate search parameters.
6. Validate vector IDs.
7. Validate batches of IDs.
8. Validate metadata.
9. Detect duplicate IDs.
10. Provide reusable validation helpers.

This module must remain provider-agnostic.

It must NOT contain:
- Qdrant-specific logic
- Pinecone-specific logic
- Milvus-specific logic
- Database clients
- Network operations
- Pipeline orchestration

SupportMind AI
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Real
from typing import Any


# ==============================================================
# String Validation
# ==============================================================


def normalize_collection_name(
    collection_name: str,
) -> str:
    """
    Validate and normalize a vector collection name.

    Leading and trailing whitespace is removed.

    Parameters
    ----------
    collection_name:
        Collection name to validate.

    Returns
    -------
    str
        Normalized collection name.

    Raises
    ------
    ValueError
        If the collection name is invalid.
    """

    if not isinstance(
        collection_name,
        str,
    ):
        raise ValueError(
            "Collection name must be a string."
        )

    normalized_name = collection_name.strip()

    if not normalized_name:
        raise ValueError(
            "Collection name cannot be empty."
        )

    return normalized_name


def normalize_provider_name(
    provider_name: str,
) -> str:
    """
    Validate and normalize a vector provider name.

    Provider names are normalized to lowercase.

    Examples
    --------
    Qdrant -> qdrant
    Pinecone -> pinecone

    Parameters
    ----------
    provider_name:
        Provider name to validate.

    Returns
    -------
    str
        Normalized provider name.
    """

    if not isinstance(
        provider_name,
        str,
    ):
        raise ValueError(
            "Provider name must be a string."
        )

    normalized_name = provider_name.strip().lower()

    if not normalized_name:
        raise ValueError(
            "Provider name cannot be empty."
        )

    return normalized_name


# ==============================================================
# Vector Validation
# ==============================================================


def validate_vector(
    vector: list[float],
    *,
    name: str = "vector",
) -> None:
    """
    Validate a vector.

    Validation rules
    ----------------
    1. Must be a list.
    2. Must not be empty.
    3. Every value must be numeric.
    4. Values must be finite.
    5. Boolean values are rejected.

    Parameters
    ----------
    vector:
        Vector to validate.

    name:
        Human-readable vector name for error messages.

    Raises
    ------
    ValueError
        If the vector is invalid.
    """

    if not isinstance(
        vector,
        list,
    ):
        raise ValueError(
            f"{name} must be a list."
        )

    if not vector:
        raise ValueError(
            f"{name} cannot be empty."
        )

    for index, value in enumerate(vector):

        if isinstance(
            value,
            bool,
        ):
            raise ValueError(
                f"{name}[{index}] must be a numeric value."
            )

        if not isinstance(
            value,
            Real,
        ):
            raise ValueError(
                f"{name}[{index}] must be a numeric value."
            )

        if not math.isfinite(
            float(value)
        ):
            raise ValueError(
                f"{name}[{index}] must be finite."
            )


def validate_vector_dimension(
    vector: list[float],
    expected_dimension: int,
    *,
    name: str = "vector",
) -> None:
    """
    Validate that a vector has the expected dimension.

    Parameters
    ----------
    vector:
        Vector to validate.

    expected_dimension:
        Expected vector dimension.

    name:
        Human-readable vector name.

    Raises
    ------
    ValueError
        If dimension configuration is invalid.

    ValueError
        If vector dimension does not match.
    """

    if not isinstance(
        expected_dimension,
        int,
    ):
        raise ValueError(
            "Expected vector dimension must be an integer."
        )

    if expected_dimension <= 0:
        raise ValueError(
            "Expected vector dimension must be greater than zero."
        )

    validate_vector(
        vector,
        name=name,
    )

    actual_dimension = len(vector)

    if actual_dimension != expected_dimension:

        raise ValueError(
            f"{name} dimension mismatch. "
            f"Expected {expected_dimension}, "
            f"received {actual_dimension}."
        )


def validate_vector_dimensions(
    vectors: list[list[float]],
    expected_dimension: int,
    *,
    name: str = "vectors",
) -> None:
    """
    Validate a batch of vectors.

    Every vector must:
    - Be valid.
    - Have the expected dimension.

    The batch itself must not be empty.

    Parameters
    ----------
    vectors:
        List of vectors.

    expected_dimension:
        Expected dimension for every vector.

    name:
        Human-readable batch name.

    Raises
    ------
    ValueError
        If any vector is invalid.
    """

    if not isinstance(
        vectors,
        list,
    ):
        raise ValueError(
            f"{name} must be a list."
        )

    if not vectors:
        raise ValueError(
            f"{name} cannot be empty."
        )

    for index, vector in enumerate(vectors):

        validate_vector_dimension(
            vector,
            expected_dimension,
            name=f"{name}[{index}]",
        )


# ==============================================================
# Search Validation
# ==============================================================


def validate_top_k(
    top_k: int,
) -> None:
    """
    Validate the number of search results requested.

    Parameters
    ----------
    top_k:
        Number of results to retrieve.

    Raises
    ------
    ValueError
        If top_k is invalid.
    """

    if isinstance(
        top_k,
        bool,
    ):
        raise ValueError(
            "top_k must be an integer."
        )

    if not isinstance(
        top_k,
        int,
    ):
        raise ValueError(
            "top_k must be an integer."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )


def validate_search_parameters(
    query_vector: list[float],
    top_k: int,
) -> None:
    """
    Validate vector similarity search parameters.

    Parameters
    ----------
    query_vector:
        Query embedding vector.

    top_k:
        Number of results requested.

    Raises
    ------
    ValueError
        If query vector or top_k is invalid.
    """

    validate_vector(
        query_vector,
        name="query_vector",
    )

    validate_top_k(
        top_k
    )


# ==============================================================
# ID Validation
# ==============================================================


def normalize_vector_id(
    vector_id: str,
) -> str:
    """
    Validate and normalize a vector ID.

    Parameters
    ----------
    vector_id:
        Vector identifier.

    Returns
    -------
    str
        Normalized vector ID.

    Raises
    ------
    ValueError
        If the ID is invalid.
    """

    if not isinstance(
        vector_id,
        str,
    ):
        raise ValueError(
            "Vector ID must be a string."
        )

    normalized_id = vector_id.strip()

    if not normalized_id:
        raise ValueError(
            "Vector ID cannot be empty."
        )

    return normalized_id


def validate_vector_ids(
    ids: list[str],
) -> None:
    """
    Validate a batch of vector IDs.

    Parameters
    ----------
    ids:
        Vector IDs.

    Raises
    ------
    ValueError
        If the ID list or any ID is invalid.
    """

    if not isinstance(
        ids,
        list,
    ):
        raise ValueError(
            "Vector IDs must be provided as a list."
        )

    if not ids:
        raise ValueError(
            "Vector IDs cannot be empty."
        )

    for index, vector_id in enumerate(ids):

        try:

            normalize_vector_id(
                vector_id
            )

        except ValueError as exc:

            raise ValueError(
                f"Invalid vector ID at index "
                f"{index}: {exc}"
            ) from exc


def find_duplicate_ids(
    ids: Iterable[str],
) -> list[str]:
    """
    Find duplicate vector IDs.

    The returned duplicate list is deterministic
    and preserves first duplicate discovery order.

    Parameters
    ----------
    ids:
        Iterable of vector IDs.

    Returns
    -------
    list[str]
        Duplicate IDs.
    """

    seen: set[str] = set()
    duplicates: list[str] = []
    duplicate_set: set[str] = set()

    for vector_id in ids:

        normalized_id = normalize_vector_id(
            vector_id
        )

        if normalized_id in seen:

            if normalized_id not in duplicate_set:

                duplicates.append(
                    normalized_id
                )

                duplicate_set.add(
                    normalized_id
                )

        else:

            seen.add(
                normalized_id
            )

    return duplicates


def validate_unique_vector_ids(
    ids: list[str],
) -> None:
    """
    Validate that all vector IDs are unique.

    Parameters
    ----------
    ids:
        Vector IDs.

    Raises
    ------
    ValueError
        If duplicate IDs are detected.
    """

    validate_vector_ids(
        ids
    )

    duplicates = find_duplicate_ids(
        ids
    )

    if duplicates:

        raise ValueError(
            "Duplicate vector IDs detected: "
            f"{duplicates}"
        )


# ==============================================================
# Metadata Validation
# ==============================================================


def validate_metadata(
    metadata: dict[str, Any],
    *,
    name: str = "metadata",
) -> None:
    """
    Validate vector metadata.

    Metadata must be a dictionary.

    Empty metadata dictionaries are allowed.

    Parameters
    ----------
    metadata:
        Metadata dictionary.

    name:
        Human-readable metadata name.

    Raises
    ------
    ValueError
        If metadata is not a dictionary.
    """

    if not isinstance(
        metadata,
        dict,
    ):
        raise ValueError(
            f"{name} must be a dictionary."
        )


def validate_metadata_batch(
    metadata_list: list[dict[str, Any]],
    *,
    name: str = "metadata_list",
) -> None:
    """
    Validate a batch of metadata dictionaries.

    Parameters
    ----------
    metadata_list:
        List of metadata dictionaries.

    Raises
    ------
    ValueError
        If metadata entries are invalid.
    """

    if not isinstance(
        metadata_list,
        list,
    ):
        raise ValueError(
            f"{name} must be a list."
        )

    for index, metadata in enumerate(
        metadata_list
    ):

        validate_metadata(
            metadata,
            name=f"{name}[{index}]",
        )


# ==============================================================
# Batch Consistency Validation
# ==============================================================


def validate_batch_consistency(
    ids: list[str],
    vectors: list[list[float]],
) -> None:
    """
    Validate that IDs and vectors form a consistent batch.

    Rules
    -----
    1. IDs must be valid.
    2. IDs must be unique.
    3. Vectors must not be empty.
    4. Number of IDs must equal number of vectors.

    Parameters
    ----------
    ids:
        Vector IDs.

    vectors:
        Vector embeddings.

    Raises
    ------
    ValueError
        If batch consistency validation fails.
    """

    validate_unique_vector_ids(
        ids
    )

    if not isinstance(
        vectors,
        list,
    ):
        raise ValueError(
            "Vectors must be provided as a list."
        )

    if not vectors:
        raise ValueError(
            "Vectors cannot be empty."
        )

    if len(ids) != len(vectors):

        raise ValueError(
            "Batch size mismatch. "
            f"Received {len(ids)} IDs "
            f"and {len(vectors)} vectors."
        )


# ==============================================================
# Utility Conversion
# ==============================================================


def to_float_vector(
    vector: list[float],
) -> list[float]:
    """
    Convert a numeric vector into a list of floats.

    Parameters
    ----------
    vector:
        Numeric vector.

    Returns
    -------
    list[float]
        Float-converted vector.

    Raises
    ------
    ValueError
        If vector is invalid.
    """

    validate_vector(
        vector
    )

    return [
        float(value)
        for value in vector
    ]