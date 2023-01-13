import pytest

@pytest.fixture
def all_cases():
    return [1, 2, 3]

def test_all_cases(all_cases):
    assert all_cases in [1, 2, 3]

