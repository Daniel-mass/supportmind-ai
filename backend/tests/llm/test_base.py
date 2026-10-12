"""
Unit tests for the base LLM provider contract.
"""

from typing import Any

import pytest

from app.llm.providers.base import BaseLLMProvider


class ConcreteTestProvider(BaseLLMProvider):
    """
    Minimal concrete provider used to test the abstract contract.
    """

    @property
    def provider_name(self) -> str:
        return "test-provider"

    @property
    def model_name(self) -> str:
        return "test-model"

    def create_model_client(self) -> Any:
        return "test-client"


def test_base_provider_cannot_be_instantiated():
    """
    BaseLLMProvider must remain abstract.
    """

    with pytest.raises(TypeError):
        BaseLLMProvider()


def test_concrete_provider_can_be_instantiated():
    """
    A provider implementing the complete contract should initialize.
    """

    provider = ConcreteTestProvider()

    assert provider is not None


def test_provider_name_contract():
    """
    provider_name should expose the provider identifier.
    """

    provider = ConcreteTestProvider()

    assert provider.provider_name == "test-provider"


def test_model_name_contract():
    """
    model_name should expose the configured model identifier.
    """

    provider = ConcreteTestProvider()

    assert provider.model_name == "test-model"


def test_create_model_client_contract():
    """
    create_model_client() should return the provider's model client.
    """

    provider = ConcreteTestProvider()

    client = provider.create_model_client()

    assert client == "test-client"


def test_provider_name_is_read_only_property():
    """
    Provider identity should be exposed through a property rather than
    requiring mutable public state.
    """

    provider = ConcreteTestProvider()

    with pytest.raises(
        AttributeError
    ):
        provider.provider_name = "another-provider"


def test_model_name_is_read_only_property():
    """
    Model identity should be exposed through a read-only property.
    """

    provider = ConcreteTestProvider()

    with pytest.raises(
        AttributeError
    ):
        provider.model_name = "another-model"


def test_incomplete_provider_cannot_be_instantiated():
    """
    A provider missing part of the required contract must remain abstract.
    """

    class IncompleteProvider(BaseLLMProvider):
        @property
        def provider_name(self) -> str:
            return "incomplete"

        @property
        def model_name(self) -> str:
            return "test-model"

        # create_model_client() intentionally omitted.

    with pytest.raises(TypeError):
        IncompleteProvider()


def test_provider_can_return_any_autogen_compatible_client():
    """
    The abstract contract should not constrain the concrete client type.
    """

    class ObjectProvider(BaseLLMProvider):
        @property
        def provider_name(self) -> str:
            return "object-provider"

        @property
        def model_name(self) -> str:
            return "object-model"

        def create_model_client(self) -> Any:
            return object()

    provider = ObjectProvider()

    client = provider.create_model_client()

    assert client is not None