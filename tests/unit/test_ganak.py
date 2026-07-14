"""Focused tests for the optional root GANAK adapter."""

from wfomc.errors import ExternalToolError, GanakError
from wfomc.ganak import GANAK_COMMIT, GANAK_REPO_URL, ganak_count


def test_ganak_count_empty_cnf_returns_one_without_binary():
    # No propositional variables means one empty model. This path must return
    # before locating GANAK because GANAK itself rejects empty input.
    assert ganak_count(0, [], {}) == 1


def test_ganak_error_belongs_to_external_tool_hierarchy():
    assert issubclass(GanakError, ExternalToolError)


def test_installer_uses_shared_ganak_metadata():
    from scripts.tools import install_ganak

    assert install_ganak.GANAK_COMMIT == GANAK_COMMIT
    assert install_ganak.GANAK_REPO_URL == GANAK_REPO_URL
