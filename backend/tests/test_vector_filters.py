"""
Unit tests for vector_filters.py

SupportMind AI
"""

from __future__ import annotations

import unittest

from app.vector_store.vector_filters import (
    FilterCondition,
    FilterOperator,
    VectorFilter,
)


class TestVectorFilters(unittest.TestCase):

    def test_create_condition(self):

        condition = FilterCondition(

            field="source",
            operator=FilterOperator.EQUAL,
            value="pdf",

        )

        self.assertEqual(condition.field, "source")
        self.assertEqual(condition.operator, FilterOperator.EQUAL)
        self.assertEqual(condition.value, "pdf")

    def test_empty_filter(self):

        vf = VectorFilter()

        self.assertTrue(vf.is_empty())

    def test_add_condition(self):

        vf = VectorFilter()

        vf.add(

            field="page",

            operator=FilterOperator.GREATER_EQUAL,

            value=5,

        )

        self.assertFalse(vf.is_empty())

        self.assertEqual(len(vf.conditions), 1)

        self.assertEqual(
            vf.conditions[0].field,
            "page",
        )

    def test_multiple_conditions(self):

        vf = VectorFilter()

        vf.add(
            "source",
            FilterOperator.EQUAL,
            "pdf",
        )

        vf.add(
            "category",
            FilterOperator.EQUAL,
            "refund",
        )

        self.assertEqual(
            len(vf.conditions),
            2,
        )

    def test_model_dump(self):

        vf = VectorFilter()

        vf.add(
            "priority",
            FilterOperator.EQUAL,
            "high",
        )

        dumped = vf.model_dump()

        self.assertEqual(

            dumped["conditions"][0]["field"],

            "priority",

        )

    def test_json(self):

        vf = VectorFilter()

        vf.add(
            "page",
            FilterOperator.GREATER_EQUAL,
            4,
        )

        json_data = vf.model_dump_json()

        self.assertIn("page", json_data)

    def test_enum_values(self):

        self.assertEqual(
            FilterOperator.EQUAL.value,
            "eq",
        )

        self.assertEqual(
            FilterOperator.IN.value,
            "in",
        )

    def test_condition_equality(self):

        a = FilterCondition(
            field="source",
            operator=FilterOperator.EQUAL,
            value="pdf",
        )

        b = FilterCondition(
            field="source",
            operator=FilterOperator.EQUAL,
            value="pdf",
        )

        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()