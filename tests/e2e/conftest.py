"""
End-to-end test fixtures - require all services running.
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = [pytest.mark.e2e]
