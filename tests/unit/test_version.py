"""Guards that ``continuum.__version__`` stays in sync with the distribution.

Regression test for the 0.2.1 bug where ``__version__`` was a hardcoded string
that drifted from the version in ``pyproject.toml``. The attribute is now derived
from the installed package metadata, so it can never disagree again.
"""

from importlib.metadata import version

import continuum


def test_version_matches_distribution_metadata() -> None:
    """``__version__`` must equal the installed distribution version."""
    assert continuum.__version__ == version("shyftlabs-continuum")


def test_version_is_not_a_placeholder() -> None:
    """The metadata lookup must succeed (package installed), not fall back."""
    assert continuum.__version__ != "0.0.0+unknown"
    assert continuum.__version__  # non-empty
