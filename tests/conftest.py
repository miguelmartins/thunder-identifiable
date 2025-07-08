# conftest.py
import os
import shutil
import tempfile

import pytest


@pytest.fixture(scope="session")
def temp_env_dir():
    """Creates a persistent temporary directory for the test session."""
    temp_dir = tempfile.mkdtemp()
    os.environ["THUNDER_BASE_DATA_FOLDER"] = temp_dir
    yield temp_dir
    shutil.rmtree(temp_dir)
