import pytest
from ai.schemas import Source


@pytest.fixture
def sample_source():
    return Source(
        title="Test Title",
        url="http://example.com",
        snippet="Test snippet",
        origin="wikipedia",
    )