"""Tests for moving_horizon module."""

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from loadshift.moving_horizon import (
    _compute_control_hours,
    _create_horizons,
    moving_horizon,
)


class TestCreateHorizons:
    """Test cases for _create_horizons function."""

    def test_docstring_example(self) -> None:
        """Test the exact example from the docstring."""
        # Create the index from the docstring example
        index = pd.date_range("2023-01-01 00:00", "2023-01-03 23:00", freq="h")

        # Call the function with docstring parameters
        horizons = _create_horizons(index, daily_decision_hour=6, horizon_length=24)

        # Expected output from docstring
        expected = [
            (pd.Timestamp("2023-01-01 00:00"), pd.Timestamp("2023-01-01 05:00")),
            (pd.Timestamp("2023-01-01 06:00"), pd.Timestamp("2023-01-02 05:00")),
            (pd.Timestamp("2023-01-02 06:00"), pd.Timestamp("2023-01-03 05:00")),
            (pd.Timestamp("2023-01-03 06:00"), pd.Timestamp("2023-01-03 23:00")),
        ]

        # Validate the result matches the docstring exactly
        assert len(horizons) == len(expected)

        for i, (actual, expected_horizon) in enumerate(
            zip(horizons, expected, strict=False)
        ):
            assert actual == expected_horizon, (
                f"Horizon {i}: expected {expected_horizon}, got {actual}"
            )

    def test_data_starts_at_decision_hour(self) -> None:
        """Test when data starts exactly at decision hour."""
        index = pd.date_range("2023-01-01 06:00", periods=48, freq="h")
        horizons = _create_horizons(index, daily_decision_hour=6, horizon_length=24)

        # Should have 2 horizons (no initial horizon since data starts at decision hour)
        # Second horizon goes until end of data since there's no third decision point
        expected = [
            (pd.Timestamp("2023-01-01 06:00"), pd.Timestamp("2023-01-02 05:00")),
            (pd.Timestamp("2023-01-02 06:00"), pd.Timestamp("2023-01-03 05:00")),
        ]

        assert len(horizons) == 2
        assert horizons == expected

    def test_daylight_saving_time_helsinki(self) -> None:
        """Test timezone handling during DST transitions in Helsinki."""
        # Test spring transition (clocks go forward): March 26, 2023
        # Night of March 25-26, 2023: 03:00 becomes 04:00 (23-hour day)
        spring_index = pd.date_range(
            "2023-03-25 00:00",
            "2023-03-27 23:00",
            freq="h",
            tz=ZoneInfo("Europe/Helsinki"),
        )

        spring_horizons = _create_horizons(
            spring_index, daily_decision_hour=6, horizon_length=24
        )

        # Should have 4 horizons: initial + 3 daily decisions (March 25, 26, 27)
        # Note: The day with DST transition has only 23 hours (2023-03-26)
        assert len(spring_horizons) == 4

        # Verify the transition day horizon handles the missing hour correctly
        # The 23-hour day affects the horizon from March 25 06:00 to March 26 05:00
        march_25_decision = pd.Timestamp(
            "2023-03-25 06:00", tz=ZoneInfo("Europe/Helsinki")
        )
        march_26_end = pd.Timestamp("2023-03-26 05:00", tz=ZoneInfo("Europe/Helsinki"))

        # Find the horizon that spans the DST transition
        # (starts March 25, ends March 26)
        dst_transition_horizon = None
        for start, end in spring_horizons:
            if start.date() == march_25_decision.date() and start.hour == 6:
                dst_transition_horizon = (start, end)
                break

        assert dst_transition_horizon is not None
        assert dst_transition_horizon[0] == march_25_decision
        assert dst_transition_horizon[1] == march_26_end

        # Count actual hours in the DST transition horizon
        # (23-hour period due to spring forward)
        dst_transition_hours = spring_index[
            (spring_index >= dst_transition_horizon[0])
            & (spring_index <= dst_transition_horizon[1])
        ]
        assert len(dst_transition_hours) == 23, (
            f"Expected 23 hours, got {len(dst_transition_hours)}"
        )

        # Test autumn transition (clocks go back): October 29, 2023
        # Night of October 28-29, 2023: 04:00 becomes 03:00 (25-hour day)
        autumn_index = pd.date_range(
            "2023-10-28 00:00",
            "2023-10-30 23:00",
            freq="h",
            tz=ZoneInfo("Europe/Helsinki"),
        )

        autumn_horizons = _create_horizons(
            autumn_index, daily_decision_hour=6, horizon_length=24
        )

        # Should have 4 horizons: initial + 3 daily decisions (Oct 28, 29, 30)
        # Note: The day with DST transition has 25 hours (2023-10-29)
        assert len(autumn_horizons) == 4

        # Verify the transition day horizon handles the extra hour correctly
        # The 25-hour day affects the horizon from October 28 06:00 to October 29 05:00
        oct_28_decision = pd.Timestamp(
            "2023-10-28 06:00", tz=ZoneInfo("Europe/Helsinki")
        )
        oct_29_end = pd.Timestamp("2023-10-29 05:00", tz=ZoneInfo("Europe/Helsinki"))

        # Find the horizon that spans the DST transition
        # (starts October 28, ends October 29)
        dst_transition_horizon_autumn = None
        for start, end in autumn_horizons:
            if start.date() == oct_28_decision.date() and start.hour == 6:
                dst_transition_horizon_autumn = (start, end)
                break

        assert dst_transition_horizon_autumn is not None
        assert dst_transition_horizon_autumn[0] == oct_28_decision
        assert dst_transition_horizon_autumn[1] == oct_29_end

        # Count actual hours in the DST transition horizon
        # (25-hour period due to fall back)
        dst_transition_hours_autumn = autumn_index[
            (autumn_index >= dst_transition_horizon_autumn[0])
            & (autumn_index <= dst_transition_horizon_autumn[1])
        ]
        assert len(dst_transition_hours_autumn) == 25, (
            f"Expected 25 hours, got {len(dst_transition_hours_autumn)}"
        )


class TestComputeControlHours:
    """Test cases for _compute_control_hours function."""

    def test_single_horizon(self) -> None:
        """Test _compute_control_hours with a single horizon."""
        # Create a single 24-hour horizon
        start = pd.Timestamp("2023-01-01 00:00")
        end = pd.Timestamp("2023-01-01 23:00")
        horizons = [(start, end)]

        # For a single horizon, should return control hours from start to end
        result = _compute_control_hours(horizons, 0)
        expected = 24  # 24 hours from 00:00 to 23:00 inclusive

        assert result == expected

    def test_daylight_saving_transitions(self) -> None:
        """Test _compute_control_hours handles DST transitions correctly."""
        # Test spring forward (23-hour day): March 26, 2023 in Helsinki
        spring_horizons = [
            (
                pd.Timestamp("2023-03-25 06:00", tz=ZoneInfo("Europe/Helsinki")),
                pd.Timestamp("2023-03-26 05:00", tz=ZoneInfo("Europe/Helsinki")),
            ),
            (
                pd.Timestamp("2023-03-26 06:00", tz=ZoneInfo("Europe/Helsinki")),
                pd.Timestamp("2023-03-27 05:00", tz=ZoneInfo("Europe/Helsinki")),
            ),
        ]

        # First horizon spans DST transition (spring forward - 23 hours)
        result_spring = _compute_control_hours(spring_horizons, 0)
        assert result_spring == 23, (
            f"Expected 23 hours for spring DST, got {result_spring}"
        )

        # Test fall back (25-hour day): October 29, 2023 in Helsinki
        autumn_horizons = [
            (
                pd.Timestamp("2023-10-28 06:00", tz=ZoneInfo("Europe/Helsinki")),
                pd.Timestamp("2023-10-29 05:00", tz=ZoneInfo("Europe/Helsinki")),
            ),
            (
                pd.Timestamp("2023-10-29 06:00", tz=ZoneInfo("Europe/Helsinki")),
                pd.Timestamp("2023-10-30 05:00", tz=ZoneInfo("Europe/Helsinki")),
            ),
        ]

        # First horizon spans DST transition (fall back - 25 hours)
        result_autumn = _compute_control_hours(autumn_horizons, 0)
        assert result_autumn == 25, (
            f"Expected 25 hours for autumn DST, got {result_autumn}"
        )


class TestMovingHorizon:
    """Test cases for moving_horizon function."""

    @pytest.mark.parametrize(
        "max_delay,expected_spillover_hour,description",
        [
            (1, 0, "max_delay = 1 hour"),
            (2, 1, "max_delay = 2 hours"),
            (3, 2, "max_delay = 3 hours"),
        ],
    )
    def test_delay_spillover_moving_horizon(
        self,
        solver: str,
        max_delay: int,
        expected_spillover_hour: int,
        description: str,
    ) -> None:
        """Test moving horizon spillover with 24-hour decision intervals.

        This test uses 36 hours of data with 24-hour decision intervals to create
        two optimization horizons. The demand is placed at hour 23 of the first
        horizon and should spill over to the second horizon based on max_delay.

        Setup:
        - 36 hours of data with 24-hour decision intervals (two horizons)
        - First horizon: hours 0-23, second horizon: hours 24-35
        - Linearly decreasing prices from 36 to 1 (cheapest at end)
        - Original demand: 1 kWh at hour 23 (last hour of first horizon)
        - Variable max_delay (1, 2, or 3 hours)

        Expected behavior:
        - First horizon: demand at hour 23 should spill over (beyond control period)
        - Second horizon: receives spillover and places it optimally based on delay
        """
        # Create datetime index for 36 hours with 24-hour decision intervals
        datetime_index = pd.date_range("2023-01-01 00:00", periods=36, freq="h")

        # Create linearly decreasing price data (36 to 1)
        prices = np.arange(36, 0, -1, dtype=float)  # [36, 35, 34, ..., 2, 1]
        price_data = pd.DataFrame(
            {"price": prices},
            index=datetime_index,
        )

        # Create demand data: single 1 kWh demand at hour 23 (last hour of first)
        demand_values = np.zeros(36)
        demand_values[23] = 1.0  # Peak at hour 23
        demand_data = pd.DataFrame({"demand": demand_values}, index=datetime_index)

        # Configuration for moving horizon optimization with 24-hour decision intervals
        config = {
            "daily_decision_hour": 0,  # Decision at 00:00 each day
            "n_lookahead_hours": 36,  # 36-hour lookahead
            "virtual_storage": {
                "max_demand_advance": 0,
                "max_demand_delay": max_delay,
                "max_hourly_purchase": 1.0,
                "max_rate": 1.0,
                "solver": solver,
            },
        }

        # Run moving horizon optimization
        result = moving_horizon(price_data, demand_data, config)
        results_df = result["results"]

        # Extract optimized demands
        optimized_demand = results_df["demand"].values

        # First horizon: hours 0-23
        first_horizon_demand = optimized_demand[:24]

        # Second horizon: hours 24-35
        second_horizon_demand = optimized_demand[24:]

        # With parametrized max_delay, demand from hour 23 can be delayed by max_delay
        # So it can go to hours 24, 25, 26... (i.e., 24 + (max_delay - 1))
        # In the second horizon control period, these correspond to hours 0, 1, 2...
        # With decreasing prices, cheapest within max_delay is at expected_hour
        expected_second_demand = np.zeros(len(second_horizon_demand))
        expected_second_demand[expected_spillover_hour] = 1.0

        # First horizon: all optimal demands should be zero (spillover)
        assert np.allclose(first_horizon_demand, 0.0), (
            f"{description}: First horizon demand should be all zeros, "
            f"but got {first_horizon_demand}"
        )

        # Second horizon: should match expected spillover placement
        assert np.allclose(second_horizon_demand, expected_second_demand), (
            f"{description}: Second horizon demand should be {expected_second_demand}, "
            f"but got {second_horizon_demand}"
        )

    @pytest.mark.parametrize(
        "max_advance,expected_spillover_hour,description",
        [
            (1, 23, "max_advance = 1 hour"),
            (2, 22, "max_advance = 2 hours"),
            (3, 21, "max_advance = 3 hours"),
        ],
    )
    def test_advance_spillover_moving_horizon(
        self,
        solver: str,
        max_advance: int,
        expected_spillover_hour: int,
        description: str,
    ) -> None:
        """Test moving horizon spillover with advance capability.

        This test uses 36 hours of data with 24-hour decision intervals to create
        two optimization horizons. The demand is placed at hour 0 of the second
        horizon and should spill back to the first horizon based on max_advance.

        Setup:
        - 36 hours of data with 24-hour decision intervals (two horizons)
        - First horizon: hours 0-23, second horizon: hours 24-35
        - Linearly increasing prices from 1 to 36 (cheapest at start)
        - Original demand: 1 kWh at hour 0 of second horizon (hour 24 globally)
        - Variable max_advance (1, 2, or 3 hours)

        Expected behavior:
        - Second horizon: demand should spill back to first horizon
        - First horizon: receives spillover and places it optimally based on advance
        """
        # Create datetime index for 36 hours with 24-hour decision intervals
        datetime_index = pd.date_range("2023-01-01 00:00", periods=36, freq="h")

        # Create linearly increasing price data (1 to 36, cheapest at start)
        prices = np.arange(0, 36, dtype=float)  # [0, 1, 2, 3, ..., 35]
        price_data = pd.DataFrame(
            {"price": prices},
            index=datetime_index,
        )

        # Create demand data: single 1 kWh demand at hour 0 of second horizon (hour 24)
        demand_values = np.zeros(36)
        demand_values[24] = 1.0  # Peak at hour 24 (first hour of second horizon)
        demand_data = pd.DataFrame({"demand": demand_values}, index=datetime_index)

        # Configuration for moving horizon optimization
        config = {
            "daily_decision_hour": 0,  # Decision at 00:00 each day
            "n_lookahead_hours": 36,  # 36-hour lookahead
            "virtual_storage": {
                "max_demand_advance": max_advance,
                "max_demand_delay": 0,
                "max_hourly_purchase": 1.0,
                "max_rate": 1.0,
                "solver": solver,
            },
        }

        # Run moving horizon optimization
        result = moving_horizon(price_data, demand_data, config)
        results_df = result["results"]

        # Extract optimized demands
        optimized_demand = results_df["demand"].values

        # First horizon: hours 0-23
        first_horizon_demand = optimized_demand[:24]

        # Second horizon: hours 24-35
        second_horizon_demand = optimized_demand[24:]

        # With increasing prices, cheapest within max_advance is at expected_hour
        expected_first_demand = np.zeros(len(first_horizon_demand))
        expected_first_demand[expected_spillover_hour] = 1.0

        # First horizon: should match expected spillover placement
        assert np.allclose(first_horizon_demand, expected_first_demand), (
            f"{description}: First horizon demand should be {expected_first_demand}, "
            f"but got {first_horizon_demand}"
        )

        # Second horizon: all optimal demands should be zero (spillover back)
        assert np.allclose(second_horizon_demand, 0.0), (
            f"{description}: Second horizon demand should be all zeros, "
            f"but got {second_horizon_demand}"
        )
