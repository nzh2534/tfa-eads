import shutil

import pytest

from tfa_eads.config import load_config


@pytest.fixture
def cfg(tmp_path):
    c = load_config(None)
    c["workdir"] = str(tmp_path / "work")
    return c


def pytest_collection_modifyitems(config, items):
    if shutil.which("xtb"):
        return
    skip = pytest.mark.skip(reason="xtb binary not on PATH")
    for item in items:
        if "xtb" in item.keywords:
            item.add_marker(skip)
