import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow: set RUN_SLOW=1 (needs Docker, a browser or a model server)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
