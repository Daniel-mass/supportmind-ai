from unittest.mock import patch

import app.observability.phoenix


def test_setup_phoenix() -> None:
    """Verify Phoenix and AutoGen instrumentation are configured."""

    with patch(
        "app.observability.phoenix.register"
    ) as mock_register, patch(
        "app.observability.phoenix.AutogenAgentChatInstrumentor"
    ) as mock_instrumentor:

        mock_tracer_provider = object()
        mock_register.return_value = mock_tracer_provider

        from app.observability.phoenix import setup_phoenix

        result = setup_phoenix()

        assert result is mock_tracer_provider

        mock_register.assert_called_once()

        mock_instrumentor.return_value.instrument.assert_called_once_with(
            tracer_provider=mock_tracer_provider,
        )