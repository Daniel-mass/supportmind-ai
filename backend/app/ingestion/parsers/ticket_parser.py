###backend/app/ingestion/parsers/ticket_parser.py
"""
Ticket Parser

Standardizes ticket records coming from different support systems.

Examples:

Zendesk
Freshdesk
Jira
ServiceNow
Salesforce

All become one common schema used across SupportMind AI.
"""

from __future__ import annotations


class TicketParser:
    """Converts raw ticket dictionaries into a standardized schema."""

    FIELD_MAPPING = {
        "ticketid": "ticket_id",
        "ticket_id": "ticket_id",
        "casenumber": "ticket_id",
        "issueid": "ticket_id",

        "customer": "customer",
        "customer_name": "customer",
        "client": "customer",

        "category": "category",
        "issue_type": "category",

        "priority": "priority",

        "status": "status",

        "description": "description",
        "details": "description",
    }

    def parse(self, ticket: dict) -> dict:
        """
        Standardize ticket fields.
        """

        normalized = {}

        for key, value in ticket.items():

            normalized_key = self.FIELD_MAPPING.get(
                key.lower(),
                key.lower()
            )

            normalized[normalized_key] = value

        return normalized