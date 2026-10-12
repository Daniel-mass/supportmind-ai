#ingestion/helpdesk_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class HelpDeskLoaderError(Exception):
    """Raised when a Help Desk source cannot be loaded."""


@dataclass(frozen=True)
class HelpDeskFileSet:
    """
    Configured Help Desk source files.

    Only the primary issues.csv file is mandatory.
    Other sources are optional because the Intelligence Layer
    should be able to load sources incrementally.
    """

    issues_csv: Path

    change_history_csv: Path | None = None
    snapshots_csv: Path | None = None
    sample_utterances_csv: Path | None = None
    scored_snapshot_xlsx: Path | None = None
    holidays_csv: Path | None = None


class HelpDeskLoader:
    """
    Loads raw Help Desk source files into pandas DataFrames.

    This component performs file loading only.
    Normalization and canonical schema mapping belong to
    the normalizer module.
    """

    CSV_SUFFIXES = {".csv"}
    EXCEL_SUFFIXES = {".xlsx", ".xls"}

    def _validate_file(self, path: Path) -> Path:
        """Validate that a source file exists and is a regular file."""

        path = Path(path)

        if not path.exists():
            raise HelpDeskLoaderError(
                f"Help Desk source file does not exist: {path}"
            )

        if not path.is_file():
            raise HelpDeskLoaderError(
                f"Help Desk source path is not a file: {path}"
            )

        return path

    def load_csv(self, path: Path) -> pd.DataFrame:
        """Load one CSV source into a DataFrame."""

        path = self._validate_file(path)

        if path.suffix.lower() not in self.CSV_SUFFIXES:
            raise HelpDeskLoaderError(
                f"Expected CSV file, received: {path}"
            )

        try:
            return pd.read_csv(path)
        except Exception as exc:
            raise HelpDeskLoaderError(
                f"Failed to load CSV source: {path}"
            ) from exc

    def load_excel(self, path: Path) -> pd.DataFrame:
        """Load one Excel source into a DataFrame."""

        path = self._validate_file(path)

        if path.suffix.lower() not in self.EXCEL_SUFFIXES:
            raise HelpDeskLoaderError(
                f"Expected Excel file, received: {path}"
            )

        try:
            return pd.read_excel(path)
        except Exception as exc:
            raise HelpDeskLoaderError(
                f"Failed to load Excel source: {path}"
            ) from exc

    def load_source(
        self,
        name: str,
        path: Path,
    ) -> pd.DataFrame:
        """
        Load one source based on its file extension.
        """

        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in self.CSV_SUFFIXES:
            return self.load_csv(path)

        if suffix in self.EXCEL_SUFFIXES:
            return self.load_excel(path)

        raise HelpDeskLoaderError(
            f"Unsupported Help Desk source format for '{name}': "
            f"{path.suffix or '<no extension>'}"
        )

    def load_all(
        self,
        files: HelpDeskFileSet,
    ) -> dict[str, pd.DataFrame]:
        """
        Load all configured Help Desk sources.

        The resulting dictionary contains only sources that
        are configured and successfully loaded.
        """

        sources: dict[str, pd.DataFrame] = {}

        configured_sources = {
            "issues": files.issues_csv,
            "change_history": files.change_history_csv,
            "snapshots": files.snapshots_csv,
            "sample_utterances": files.sample_utterances_csv,
            "scored_snapshot": files.scored_snapshot_xlsx,
            "holidays": files.holidays_csv,
        }

        for name, path in configured_sources.items():

            if path is None:
                continue

            sources[name] = self.load_source(
                name=name,
                path=path,
            )

        return sources