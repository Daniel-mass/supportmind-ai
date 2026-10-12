#helpdeskloadertest
from pathlib import Path

import pandas as pd
import pytest

from backend.app.intelligence.ingestion.helpdesk_loader import (
    HelpDeskFileSet,
    HelpDeskLoader,
    HelpDeskLoaderError,
)


def test_load_csv(tmp_path: Path):
    csv_path = tmp_path / "issues.csv"

    pd.DataFrame(
        {
            "issue_proj": ["C01", "C02"],
            "issue_num": [1, 2],
            "issue_priority": ["High", "Medium"],
        }
    ).to_csv(csv_path, index=False)

    loader = HelpDeskLoader()

    result = loader.load_csv(csv_path)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 3)
    assert list(result.columns) == [
        "issue_proj",
        "issue_num",
        "issue_priority",
    ]


def test_load_excel(tmp_path: Path):
    excel_path = tmp_path / "scored_snapshot.xlsx"

    pd.DataFrame(
        {
            "id": [1, 2],
            "Q1": [4, 5],
            "Q2": [4, 5],
            "Q3": [3, 5],
        }
    ).to_excel(
        excel_path,
        index=False,
    )

    loader = HelpDeskLoader()

    result = loader.load_excel(excel_path)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 4)
    assert list(result.columns) == [
        "id",
        "Q1",
        "Q2",
        "Q3",
    ]


def test_missing_file_is_rejected(tmp_path: Path):
    loader = HelpDeskLoader()

    missing_path = tmp_path / "missing.csv"

    with pytest.raises(HelpDeskLoaderError):
        loader.load_csv(missing_path)


def test_unsupported_file_format_is_rejected(tmp_path: Path):
    txt_path = tmp_path / "issues.txt"

    txt_path.write_text(
        "some content",
        encoding="utf-8",
    )

    loader = HelpDeskLoader()

    with pytest.raises(HelpDeskLoaderError):
        loader.load_source(
            name="issues",
            path=txt_path,
        )


def test_load_all_configured_sources(tmp_path: Path):
    issues_path = tmp_path / "issues.csv"
    history_path = tmp_path / "change_history.csv"

    pd.DataFrame(
        {
            "issue_num": [1],
            "issue_priority": ["High"],
        }
    ).to_csv(
        issues_path,
        index=False,
    )

    pd.DataFrame(
        {
            "issue_num": [1],
            "status": ["open"],
        }
    ).to_csv(
        history_path,
        index=False,
    )

    files = HelpDeskFileSet(
        issues_csv=issues_path,
        change_history_csv=history_path,
    )

    loader = HelpDeskLoader()

    result = loader.load_all(files)

    assert set(result.keys()) == {
        "issues",
        "change_history",
    }

    assert result["issues"].shape == (1, 2)
    assert result["change_history"].shape == (1, 2)


def test_optional_sources_can_be_omitted(tmp_path: Path):
    issues_path = tmp_path / "issues.csv"

    pd.DataFrame(
        {
            "issue_num": [1],
        }
    ).to_csv(
        issues_path,
        index=False,
    )

    files = HelpDeskFileSet(
        issues_csv=issues_path,
    )

    loader = HelpDeskLoader()

    result = loader.load_all(files)

    assert list(result.keys()) == ["issues"]
    assert result["issues"].shape == (1, 1)