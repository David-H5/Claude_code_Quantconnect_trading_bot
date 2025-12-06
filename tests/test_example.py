"""
Example test file demonstrating test structure.

Run with: pytest tests/
"""

import pytest


def test_example():
    """
    Example test that always passes.
    """
    assert True


def test_basic_math():
    """
    Example test for basic calculations.
    """
    assert 1 + 1 == 2
    assert 5 * 5 == 25


@pytest.mark.unit
def test_with_marker():
    """
    Example test with custom marker.
    Run only unit tests with: pytest -m unit
    """
    result = "hello".upper()
    assert result == "HELLO"


class TestExampleClass:
    """
    Example test class for organizing related tests.
    """

    def test_method_one(self):
        """Test method one."""
        assert len([1, 2, 3]) == 3

    def test_method_two(self):
        """Test method two."""
        assert max([1, 5, 3]) == 5


# TODO: Add tests for your algorithms, indicators, and utilities
# Example test structure for an algorithm:
#
# from algorithms.simple_momentum import SimpleMomentumAlgorithm
#
# class TestSimpleMomentumAlgorithm:
#     def test_initialization(self):
#         # Test algorithm initialization
#         pass
#
#     def test_rsi_calculation(self):
#         # Test RSI indicator setup
#         pass
#
#     def test_entry_signal(self):
#         # Test buy signal logic
#         pass
#
#     def test_exit_signal(self):
#         # Test sell signal logic
#         pass
