"""
test_utils.py

Unit tests for provider-agnostic vector store utilities.

These tests are:
- Fast
- Deterministic
- Provider-independent
- Network-free

SupportMind AI
"""

from __future__ import annotations

import unittest

from app.vector_store.utils import (
    find_duplicate_ids,
    normalize_collection_name,
    normalize_provider_name,
    normalize_vector_id,
    to_float_vector,
    validate_batch_consistency,
    validate_metadata,
    validate_metadata_batch,
    validate_search_parameters,
    validate_top_k,
    validate_unique_vector_ids,
    validate_vector,
    validate_vector_dimension,
    validate_vector_dimensions,
    validate_vector_ids,
)


class TestStringNormalization(unittest.TestCase):
    """
    Tests for collection and provider name normalization.
    """

    # ==========================================================
    # Collection Names
    # ==========================================================

    def test_normalize_collection_name(self):

        result = normalize_collection_name(
            "  supportmind_documents  "
        )

        self.assertEqual(
            result,
            "supportmind_documents",
        )

    def test_normalize_collection_name_empty(self):

        with self.assertRaises(ValueError):

            normalize_collection_name(
                ""
            )

    def test_normalize_collection_name_whitespace(self):

        with self.assertRaises(ValueError):

            normalize_collection_name(
                "   "
            )

    def test_normalize_collection_name_invalid_type(self):

        with self.assertRaises(ValueError):

            normalize_collection_name(
                123
            )

    # ==========================================================
    # Provider Names
    # ==========================================================

    def test_normalize_provider_name(self):

        result = normalize_provider_name(
            "  QDRANT  "
        )

        self.assertEqual(
            result,
            "qdrant",
        )

    def test_normalize_provider_name_empty(self):

        with self.assertRaises(ValueError):

            normalize_provider_name(
                ""
            )

    def test_normalize_provider_name_invalid_type(self):

        with self.assertRaises(ValueError):

            normalize_provider_name(
                123
            )


class TestVectorValidation(unittest.TestCase):
    """
    Tests for vector validation.
    """

    def test_validate_vector_success(self):

        validate_vector(
            [
                1.0,
                0.0,
                0.5,
            ]
        )

    def test_validate_vector_integer_values(self):

        validate_vector(
            [
                1,
                2,
                3,
            ]
        )

    def test_validate_vector_empty(self):

        with self.assertRaises(ValueError):

            validate_vector(
                []
            )

    def test_validate_vector_invalid_type(self):

        with self.assertRaises(ValueError):

            validate_vector(
                "not-a-vector"
            )

    def test_validate_vector_non_numeric_value(self):

        with self.assertRaises(ValueError):

            validate_vector(
                [
                    1.0,
                    "invalid",
                    0.0,
                ]
            )

    def test_validate_vector_boolean_value(self):

        with self.assertRaises(ValueError):

            validate_vector(
                [
                    1.0,
                    True,
                    0.0,
                ]
            )

    def test_validate_vector_nan(self):

        with self.assertRaises(ValueError):

            validate_vector(
                [
                    1.0,
                    float("nan"),
                    0.0,
                ]
            )

    def test_validate_vector_positive_infinity(self):

        with self.assertRaises(ValueError):

            validate_vector(
                [
                    1.0,
                    float("inf"),
                    0.0,
                ]
            )

    def test_validate_vector_negative_infinity(self):

        with self.assertRaises(ValueError):

            validate_vector(
                [
                    1.0,
                    float("-inf"),
                    0.0,
                ]
            )


class TestVectorDimensionValidation(unittest.TestCase):
    """
    Tests for vector dimension validation.
    """

    def test_validate_vector_dimension_success(self):

        validate_vector_dimension(
            [
                1.0,
                0.0,
                0.0,
            ],
            3,
        )

    def test_validate_vector_dimension_mismatch(self):

        with self.assertRaises(ValueError):

            validate_vector_dimension(
                [
                    1.0,
                    0.0,
                ],
                3,
            )

    def test_validate_vector_dimension_invalid_expected_dimension(self):

        with self.assertRaises(ValueError):

            validate_vector_dimension(
                [
                    1.0,
                    0.0,
                ],
                0,
            )

    def test_validate_vector_dimension_non_integer_expected_dimension(self):

        with self.assertRaises(ValueError):

            validate_vector_dimension(
                [
                    1.0,
                    0.0,
                ],
                2.5,
            )

    def test_validate_vector_dimensions_success(self):

        validate_vector_dimensions(
            [
                [
                    1.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                ],
            ],
            2,
        )

    def test_validate_vector_dimensions_empty_batch(self):

        with self.assertRaises(ValueError):

            validate_vector_dimensions(
                [],
                3,
            )

    def test_validate_vector_dimensions_mismatch(self):

        with self.assertRaises(ValueError):

            validate_vector_dimensions(
                [
                    [
                        1.0,
                        0.0,
                    ],
                    [
                        1.0,
                        0.0,
                        0.0,
                    ],
                ],
                2,
            )


class TestSearchValidation(unittest.TestCase):
    """
    Tests for search parameter validation.
    """

    def test_validate_top_k_success(self):

        validate_top_k(
            5
        )

    def test_validate_top_k_zero(self):

        with self.assertRaises(ValueError):

            validate_top_k(
                0
            )

    def test_validate_top_k_negative(self):

        with self.assertRaises(ValueError):

            validate_top_k(
                -1
            )

    def test_validate_top_k_non_integer(self):

        with self.assertRaises(ValueError):

            validate_top_k(
                5.5
            )

    def test_validate_top_k_boolean(self):

        with self.assertRaises(ValueError):

            validate_top_k(
                True
            )

    def test_validate_search_parameters_success(self):

        validate_search_parameters(
            query_vector=[
                1.0,
                0.0,
                0.0,
            ],
            top_k=5,
        )

    def test_validate_search_parameters_invalid_vector(self):

        with self.assertRaises(ValueError):

            validate_search_parameters(
                query_vector=[],
                top_k=5,
            )

    def test_validate_search_parameters_invalid_top_k(self):

        with self.assertRaises(ValueError):

            validate_search_parameters(
                query_vector=[
                    1.0,
                    0.0,
                    0.0,
                ],
                top_k=0,
            )


class TestVectorIDValidation(unittest.TestCase):
    """
    Tests for vector ID validation.
    """

    def test_normalize_vector_id(self):

        result = normalize_vector_id(
            "  node_123  "
        )

        self.assertEqual(
            result,
            "node_123",
        )

    def test_normalize_vector_id_empty(self):

        with self.assertRaises(ValueError):

            normalize_vector_id(
                ""
            )

    def test_normalize_vector_id_whitespace(self):

        with self.assertRaises(ValueError):

            normalize_vector_id(
                "   "
            )

    def test_normalize_vector_id_invalid_type(self):

        with self.assertRaises(ValueError):

            normalize_vector_id(
                123
            )

    def test_validate_vector_ids_success(self):

        validate_vector_ids(
            [
                "node_1",
                "node_2",
                "node_3",
            ]
        )

    def test_validate_vector_ids_empty(self):

        with self.assertRaises(ValueError):

            validate_vector_ids(
                []
            )

    def test_validate_vector_ids_invalid_id(self):

        with self.assertRaises(ValueError):

            validate_vector_ids(
                [
                    "node_1",
                    "",
                ]
            )


class TestDuplicateIDs(unittest.TestCase):
    """
    Tests for duplicate vector ID detection.
    """

    def test_find_duplicate_ids(self):

        duplicates = find_duplicate_ids(
            [
                "node_1",
                "node_2",
                "node_1",
                "node_3",
                "node_2",
            ]
        )

        self.assertEqual(
            duplicates,
            [
                "node_1",
                "node_2",
            ],
        )

    def test_find_duplicate_ids_without_duplicates(self):

        duplicates = find_duplicate_ids(
            [
                "node_1",
                "node_2",
                "node_3",
            ]
        )

        self.assertEqual(
            duplicates,
            [],
        )

    def test_find_duplicate_ids_normalizes_whitespace(self):

        duplicates = find_duplicate_ids(
            [
                "node_1",
                " node_1 ",
            ]
        )

        self.assertEqual(
            duplicates,
            [
                "node_1",
            ],
        )

    def test_validate_unique_vector_ids_success(self):

        validate_unique_vector_ids(
            [
                "node_1",
                "node_2",
            ]
        )

    def test_validate_unique_vector_ids_duplicate(self):

        with self.assertRaises(ValueError):

            validate_unique_vector_ids(
                [
                    "node_1",
                    "node_2",
                    "node_1",
                ]
            )


class TestMetadataValidation(unittest.TestCase):
    """
    Tests for metadata validation.
    """

    def test_validate_metadata_success(self):

        validate_metadata(
            {
                "category": "billing",
                "page": 5,
            }
        )

    def test_validate_empty_metadata(self):

        validate_metadata(
            {}
        )

    def test_validate_metadata_invalid_type(self):

        with self.assertRaises(ValueError):

            validate_metadata(
                "invalid"
            )

    def test_validate_metadata_batch_success(self):

        validate_metadata_batch(
            [
                {
                    "category": "billing",
                },
                {
                    "category": "technical",
                },
            ]
        )

    def test_validate_metadata_batch_invalid_entry(self):

        with self.assertRaises(ValueError):

            validate_metadata_batch(
                [
                    {
                        "category": "billing",
                    },
                    "invalid",
                ]
            )


class TestBatchConsistency(unittest.TestCase):
    """
    Tests for ID/vector batch consistency.
    """

    def test_validate_batch_consistency_success(self):

        validate_batch_consistency(
            ids=[
                "node_1",
                "node_2",
            ],
            vectors=[
                [
                    1.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                ],
            ],
        )

    def test_validate_batch_consistency_size_mismatch(self):

        with self.assertRaises(ValueError):

            validate_batch_consistency(
                ids=[
                    "node_1",
                    "node_2",
                ],
                vectors=[
                    [
                        1.0,
                        0.0,
                    ],
                ],
            )

    def test_validate_batch_consistency_duplicate_ids(self):

        with self.assertRaises(ValueError):

            validate_batch_consistency(
                ids=[
                    "node_1",
                    "node_1",
                ],
                vectors=[
                    [
                        1.0,
                        0.0,
                    ],
                    [
                        0.0,
                        1.0,
                    ],
                ],
            )

    def test_validate_batch_consistency_empty_vectors(self):

        with self.assertRaises(ValueError):

            validate_batch_consistency(
                ids=[
                    "node_1",
                ],
                vectors=[],
            )


class TestFloatConversion(unittest.TestCase):
    """
    Tests for numeric vector conversion.
    """

    def test_to_float_vector(self):

        result = to_float_vector(
            [
                1,
                2,
                3,
            ]
        )

        self.assertEqual(
            result,
            [
                1.0,
                2.0,
                3.0,
            ],
        )

        self.assertTrue(
            all(
                isinstance(
                    value,
                    float,
                )
                for value in result
            )
        )

    def test_to_float_vector_already_float(self):

        result = to_float_vector(
            [
                0.1,
                0.2,
                0.3,
            ]
        )

        self.assertEqual(
            result,
            [
                0.1,
                0.2,
                0.3,
            ],
        )

    def test_to_float_vector_invalid(self):

        with self.assertRaises(ValueError):

            to_float_vector(
                [
                    1.0,
                    "invalid",
                ]
            )


# ==============================================================
# Test Runner
# ==============================================================


if __name__ == "__main__":

    unittest.main()