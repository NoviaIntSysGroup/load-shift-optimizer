"""Tests for virtual_storage module."""

import numpy as np
import pytest

from loadshift.virtual_storage import VirtualStorage


class TestVirtualStorage:
    """Test cases for VirtualStorage optimization."""

    @pytest.mark.parametrize(
        "max_advance,max_delay,expected_time,description",
        [
            (0, 0, 4, "Base case - no advance/delay capability"),
            (1, 0, 3, "Advance only (1 hour)"),
            (0, 1, 5, "Delay only (1 hour)"),
            (4, 0, 0, "Advance only (4 hours)"),
            (0, 4, 8, "Delay only (4 hours)"),
        ],
    )
    def test_pyramid_price_demand_shifting(
        self,
        solver: str,
        max_advance: int,
        max_delay: int,
        expected_time: int,
        description: str,
    ) -> None:
        """Test demand shifting with pyramid price profile and single peak demand.

        This test verifies that the virtual storage system correctly shifts demand
        based on advance/delay parameters with a pyramid price structure.

        Test Setup:
        =======================

        Price Profile (9 values, pyramid shape):
        Time:    0  1  2  3  4  5  6  7  8
        Price:  [1, 2, 3, 4, 5, 4, 3, 2, 1]
                             ▲

        Original Demand Profile (1 kWh peak at center):
        Time:    0  1  2  3  4  5  6  7  8
        Demand: [0, 0, 0, 0, 1, 0, 0, 0, 0]
                             ▲
        """
        # Create pyramid price profile (9 values, peak in middle)
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0])

        # Create single peak demand at the center (time index 4)
        demand = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        storage = VirtualStorage(
            max_demand_advance=max_advance,
            max_demand_delay=max_delay,
            max_hourly_purchase=1.0,
            max_rate=1.0,
            solver=solver,
        )

        result = storage.optimize_demand(prices, demand)
        optimized_demand = result["optimal_demand"]

        # For base case, demand should remain unchanged
        if max_advance == 0 and max_delay == 0:
            assert np.allclose(optimized_demand, demand), (
                f"{description}: should return original demand profile"
            )
        else:
            # Demand should be shifted away from time 4 (original peak)
            assert np.isclose(optimized_demand[4], 0.0), (
                f"{description}: demand should be shifted away from time 4, "
                f"but got {optimized_demand[4]}"
            )
            # Demand should appear at expected time
            assert np.isclose(optimized_demand[expected_time], 1.0), (
                f"{description}: demand should be shifted to time {expected_time}, "
                f"but got {optimized_demand[expected_time]}"
            )
            # Total demand must be conserved
            assert np.isclose(np.sum(optimized_demand), 1.0), (
                f"{description}: total demand must be conserved"
            )

    @pytest.mark.parametrize(
        "max_advance,max_delay,max_hourly_purchase,max_rate,expected_demand,description",
        [
            (
                2,
                0,
                1.0,
                1.0,
                [0.0, 0.0, 1.0, 0.0, 1.0],
                "max_hourly_purchase=1.0 limits to 1 kWh transfer",
            ),
            (
                2,
                0,
                1.0,
                2.0,
                [0.0, 0.0, 1.0, 1.0, 0.0],
                "max_rate=2.0 allows both kWh to move",
            ),
            (
                2,
                0,
                2.0,
                2.0,
                [0.0, 0.0, 2.0, 0.0, 0.0],
                "max_hourly_purchase=2.0 allows all energy to optimal time",
            ),
        ],
    )
    def test_max_rate_and_max_hourly_purchase_constraint(
        self,
        solver: str,
        max_advance: int,
        max_delay: int,
        max_hourly_purchase: float,
        max_rate: float,
        expected_demand: list,
        description: str,
    ) -> None:
        """Test max_hourly_purchase and max_rate constraints with linearly
        decreasing prices.

        This test verifies that the virtual storage system respects both the
        max_hourly_purchase
        and max_rate constraints when optimizing energy transfers.

        Test Setup:
        =======================

        Price Profile (5 values, linearly decreasing):
        Time:    0  1  2  3  4
        Price:  [1, 2, 3, 4, 5]

        Original Demand Profile (2 kWh peak at end):
        Time:    0  1  2  3  4
        Demand: [0, 0, 0, 0, 2]

        Expected behavior varies based on constraint parameters.
        """
        # Create linearly decreasing price profile (time 0 is cheapest)
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Create 2 kWh peak demand at time 4 (most expensive) to test constraint
        demand = np.array([0.0, 0.0, 0.0, 0.0, 2.0])

        storage = VirtualStorage(
            max_demand_advance=max_advance,
            max_demand_delay=max_delay,
            max_hourly_purchase=max_hourly_purchase,
            max_rate=max_rate,
            solver=solver,
        )

        result = storage.optimize_demand(prices, demand)
        optimized_demand = result["optimal_demand"]

        expected_demand_array = np.array(expected_demand)
        assert np.allclose(optimized_demand, expected_demand_array), (
            f"{description} failed. Expected: {expected_demand_array}, "
            f"Got: {optimized_demand}"
        )

    @pytest.mark.parametrize(
        "max_delay,expected_remove_spillover_idx,description",
        [
            (3, 0, "max_delay = 3 hours"),
            (4, 1, "max_delay = 4 hours"),
            (5, 2, "max_delay = 5 hours"),
        ],
    )
    def test_spillover_with_varying_max_delay(
        self,
        solver: str,
        max_delay: int,
        expected_remove_spillover_idx: int,
        description: str,
    ) -> None:
        """Test spillover functionality with 6h lookahead, 3h control period,
        and varying max_delay.

        This test verifies spillover behavior when demand can be delayed beyond
        the control horizon.
        The setup uses:
        - 6 hours lookahead period
        - 3 hours control period (times 0-2)
        - Original demand peak of 1 at first hour (time 0)
        - Linearly decreasing prices
        - Three test cases with max_delay of 3, 4, and 5 hours (to force spillover)

        Expected behavior:
        - With high max_delay, demand should be moved beyond the lookahead horizon
        - Remove from spillover should track where energy was moved to
        """
        # 6-hour lookahead with linearly decreasing prices
        prices = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])  # Price decreases each hour

        # Original demand has peak of 1 at first hour (time 0)
        demand = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Control period of 3 hours (indices 0-2)
        n_control_hours = 3

        storage = VirtualStorage(
            max_demand_advance=0,
            max_demand_delay=max_delay,
            max_hourly_purchase=1.0,
            max_rate=1.0,
            solver=solver,
        )

        result = storage.optimize_demand(
            prices, demand, n_control_hours=n_control_hours
        )
        optimized_demand = result["optimal_demand"]
        remove_spillover = result["remove_spillover"]
        add_spillover = result["add_spillover"]

        # Demand should move beyond control horizon (0-2), so control period
        # should be all zeros
        expected_demand = np.array([0.0, 0.0, 0.0])
        assert np.allclose(optimized_demand, expected_demand), (
            f"{description}: demand should move beyond control horizon"
        )

        # Verify spillover: demand moved from time 0 should appear in
        # remove_from_spillover
        assert np.isclose(remove_spillover[expected_remove_spillover_idx], 1.0), (
            f"{description}: remove spillover should have 1 at position "
            f"{expected_remove_spillover_idx}"
        )

        # All other positions in remove_spillover should be zero
        other_indices = [
            i
            for i in range(len(remove_spillover))
            if i != expected_remove_spillover_idx
        ]
        assert np.allclose(remove_spillover[other_indices], 0.0), (
            f"{description}: remove spillover should be zero elsewhere"
        )

        # Add spillover should be all zeros
        assert np.allclose(add_spillover, 0.0), (
            f"{description}: add spillover should be all zeros"
        )

    @pytest.mark.parametrize(
        "max_advance,expected_demand_idx,description",
        [
            (3, 2, "max_advance = 3 hours"),
            (4, 1, "max_advance = 4 hours"),
            (5, 0, "max_advance = 5 hours"),
        ],
    )
    def test_spillover_with_varying_max_advance(
        self, solver: str, max_advance: int, expected_demand_idx: int, description: str
    ) -> None:
        """Test spillover functionality with 6h lookahead, 3h control period,
        and varying max_advance.

        This test verifies spillover behavior when demand can be advanced from
        beyond the control horizon.
        The setup uses:
        - 6 hours lookahead period
        - 3 hours control period (times 0-2)
        - Original demand peak of 1 at last hour (time 5)
        - Linearly increasing prices (incentivizes moving demand earlier)
        - Three test cases with max_advance of 3, 4, and 5 hours

        Expected behavior:
        - Demand from time 5 should be moved to earlier times with lower prices
        - Add to spillover should track where energy was added within control period
        """
        # 6-hour lookahead with linearly increasing prices
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # Price increases each hour

        # Original demand has peak of 1 at last hour (time 5)
        demand = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        # Control period of 3 hours (indices 0-2)
        n_control_hours = 3

        storage = VirtualStorage(
            max_demand_advance=max_advance,
            max_demand_delay=0,
            max_hourly_purchase=1.0,
            max_rate=1.0,
            solver=solver,
        )

        result = storage.optimize_demand(
            prices, demand, n_control_hours=n_control_hours
        )
        optimized_demand = result["optimal_demand"]
        remove_spillover = result["remove_spillover"]
        add_spillover = result["add_spillover"]

        # Demand should be moved to the expected time within control period
        expected_demand = np.array([0.0, 0.0, 0.0])
        expected_demand[expected_demand_idx] = 1.0
        assert np.allclose(optimized_demand, expected_demand), (
            f"{description}: demand should move to time {expected_demand_idx}"
        )

        # Verify spillover: demand added to control period should appear in
        # add_spillover at index 2
        # (based on the spillover indexing scheme which appears to be related to
        # lookback hours)
        assert np.isclose(add_spillover[2], 1.0), (
            f"{description}: add spillover should have 1 at position 2"
        )

        # All other positions in add_spillover should be zero
        other_indices = [i for i in range(len(add_spillover)) if i != 2]
        assert np.allclose(add_spillover[other_indices], 0.0), (
            f"{description}: add spillover should be zero elsewhere"
        )

        # Remove spillover should be all zeros (no demand removed from control period)
        assert np.allclose(remove_spillover, 0.0), (
            f"{description}: remove spillover should be all zeros"
        )
