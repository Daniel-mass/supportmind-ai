#backend/app/ingestion/parsers/email_parser.py
"""
Email Parser

Parses customer support emails into structured components.

Extracts:

- Greeting
- Issue Description
- Request
- Closing
- Signature

This structured representation is later used by:

- Intent Classification
- Ticket Generation
- NLP Pipeline
- AutoGen Agents
"""

from __future__ import annotations

import re


class EmailParser:
    """Parses an email body into structured sections."""

    CLOSINGS = (
        "regards",
        "thanks",
        "thank you",
        "best",
        "sincerely",
        "cheers",
    )

    def parse(self, text: str) -> dict:

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        greeting = ""
        issue = ""
        request = ""
        closing = ""
        signature = ""

        # ----------------------------
        # Greeting
        # ----------------------------

        if lines:

            if re.match(r"^(hi|hello|dear)\b", lines[0], re.IGNORECASE):
                greeting = lines[0]
                lines = lines[1:]

        # ----------------------------
        # Closing
        # ----------------------------

        closing_index = None

        for i, line in enumerate(lines):

            lower = line.lower()

            if any(lower.startswith(c) for c in self.CLOSINGS):

                closing = line
                closing_index = i

                break

        if closing_index is not None:

            body = lines[:closing_index]

            if closing_index + 1 < len(lines):
                signature = lines[closing_index + 1]

        else:

            body = lines

        # ----------------------------
        # Detect Request
        # ----------------------------

        request_lines = []

        issue_lines = []

        request_keywords = (
            "please",
            "kindly",
            "can you",
            "could you",
            "help",
        )

        for line in body:

            lower = line.lower()

            if any(keyword in lower for keyword in request_keywords):

                request_lines.append(line)

            else:

                issue_lines.append(line)

        issue = "\n".join(issue_lines).strip()

        request = "\n".join(request_lines).strip()

        return {

            "greeting": greeting,

            "issue": issue,

            "request": request,

            "closing": closing,

            "signature": signature,
        }