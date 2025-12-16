"""Tests for TimeRanges class."""

import pytest

from loadshift.time_ranges import TimeRanges


class TestTimeRanges:
    """Test cases for TimeRanges class."""

    def test_docstring_example(self) -> None:
        """Test the concrete example from the docstring.

        Example: n_lookback_hours=3, n_control_hours=5, n_lookahead_hours=9
        """
        # Create TimeRanges instance matching docstring example
        time_ranges = TimeRanges(n_lookback_hours=3)
        ranges = time_ranges.build(n_control_hours=5, n_lookahead_hours=9)

        # Test lookback indices [0,1,2]
        assert list(ranges.lookback_indices) == [0, 1, 2]

        # Test control indices [3,4,5,6,7]
        assert list(ranges.control_indices) == [3, 4, 5, 6, 7]

        # Test lookahead indices [3,4,5,6,7,8,9,10,11]
        assert list(ranges.lookahead_indices) == [3, 4, 5, 6, 7, 8, 9, 10, 11]

        # Test spillover indices [5,6,7]
        # (from n_control_hours to n_control_hours+n_lookback_hours)
        # With n_control_hours=5, n_lookback_hours=3: range(5, 5+3) = [5,6,7]
        assert list(ranges.spillover_indices) == [5, 6, 7]

        # Test local time [0,1,2,3,4,5,6,7,8]
        assert list(ranges.local_time) == [0, 1, 2, 3, 4, 5, 6, 7, 8]

        # Test all indices [0,1,2,3,4,5,6,7,8,9,10,11]
        assert list(ranges.all_indices) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        # Test index conversion from docstring
        assert ranges.time_to_global_index(0) == 3  # start of control period

    def test_validation(self) -> None:
        """Test validation of parameters."""
        # Test negative n_lookback_hours
        with pytest.raises(ValueError, match="n_lookback_hours must be non-negative"):
            TimeRanges(n_lookback_hours=-1)

        time_ranges = TimeRanges(n_lookback_hours=4)

        # Test negative n_control_hours
        with pytest.raises(ValueError, match="n_control_hours must be non-negative"):
            time_ranges.build(n_control_hours=-1, n_lookahead_hours=8)

        # Test negative n_lookahead_hours
        with pytest.raises(ValueError, match="n_lookahead_hours must be non-negative"):
            time_ranges.build(n_control_hours=6, n_lookahead_hours=-1)

        # Test n_control_hours > n_lookahead_hours
        with pytest.raises(
            ValueError,
            match=(
                "n_control_hours \\(10\\) must be less than or equal to "
                "n_lookahead_hours \\(8\\)"
            ),
        ):
            time_ranges.build(n_control_hours=10, n_lookahead_hours=8)
