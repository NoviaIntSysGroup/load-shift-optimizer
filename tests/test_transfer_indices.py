"""Tests for TransferIndices functionality.

Tests the core functionality of transfer matrix index building and mapping,
which is central to the virtual storage optimization algorithm.
"""

from loadshift.time_ranges import TimeRanges
from loadshift.transfer_indices import TransferIndices


class TestTransferIndices:
    """Test the transfer matrix index building functionality."""

    def test_transfer_matrix_structure_concrete_example(self) -> None:
        """Test 1: Verify indices produce the correct transfer matrix structure.

         This is the most critical test using a concrete example:
         - history_padding = 3 (past_hours)
         - n_hours = 9 (future_hours)
         - max_demand_advance = 2
         - max_demand_delay = 3

         Timeline:
         - Historical indices: [0, 1, 2] (fixed from previous optimization)
         - Future indices: [3, 4, 5, 6, 7, 8, 9, 10, 11] (current optimization)
         - Total indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

         Expected Transfer Matrix T[i,j] (1 = transfer allowed, 0 = not allowed):

            j→  0  1  2  3  4  5  6  7  8  9 10 11
         i ↓  ┌────────────────────────────────────┐
         0    │ 0  0  0  1  0  0  0  0  0  0  0  0 │ (historical: constrained to j≥3)
         1    │ 0  0  0  1  1  0  0  0  0  0  0  0 │ (historical: constrained to j≥3)
         2    │ 0  0  0  1  1  1  0  0  0  0  0  0 │ (historical: constrained to j≥3)
         3    │ 0  1  1  0  1  1  1  0  0  0  0  0 │ (future: [1,7), excludes i=3)
         4    │ 0  0  1  1  0  1  1  1  0  0  0  0 │ (future: [2,8), excludes i=4)
         5    │ 0  0  0  1  1  0  1  1  1  0  0  0 │ (future: [3,9), excludes i=5)
         6    │ 0  0  0  0  1  1  0  1  1  1  0  0 │ (future: [4,10), excludes i=6)
         7    │ 0  0  0  0  0  1  1  0  1  1  1  0 │ (future: [5,11), excludes i=7)
         8    │ 0  0  0  0  0  0  1  1  0  1  1  1 │ (future: [6,12), excludes i=8)
         9    │ 0  0  0  0  0  0  0  1  1  0  1  1 │ (future: [7,12), excludes i=9)
        10    │ 0  0  0  0  0  0  0  0  1  1  0  1 │ (future: [8,12), excludes i=10)
        11    │ 0  0  0  0  0  0  0  0  0  1  1  0 │ (future: [9,12), excludes i=11)
              └────────────────────────────────────┘
        """
        # Create TimeRanges with n_lookback_hours=3 and n_lookahead_hours=9
        time_ranges_factory = TimeRanges(n_lookback_hours=3)
        ranges = time_ranges_factory.build(n_control_hours=5, n_lookahead_hours=9)

        # Create TransferIndices with max_demand_advance=2, max_demand_delay=3
        transfer_indices_factory = TransferIndices(
            max_demand_advance=2, max_demand_delay=3, n_lookback_hours=3
        )
        transfer_indices = transfer_indices_factory.build(ranges)
        move_to_indices = transfer_indices.move_to_indices
        get_from_indices = transfer_indices.get_from_indices

        # Expected move_to_indices based on actual behavior (from debugging)
        expected_move_to = {
            0: [3],  # Historical: range [3,4), excludes i=0
            1: [3, 4],  # Historical: range [3,5), excludes i=1
            2: [3, 4, 5],  # Historical: range [3,6), excludes i=2
            3: [1, 2, 4, 5, 6],  # Future: range [1,7), excludes i=3
            4: [2, 3, 5, 6, 7],  # Future: range [2,8), excludes i=4
            5: [3, 4, 6, 7, 8],  # Future: range [3,9), excludes i=5
            6: [4, 5, 7, 8, 9],  # Future: range [4,10), excludes i=6
            7: [5, 6, 8, 9, 10],  # Future: range [5,11), excludes i=7
            8: [6, 7, 9, 10, 11],  # Future: range [6,12), excludes i=8
            9: [7, 8, 10, 11],  # Future: range [7,12), excludes i=9
            10: [8, 9, 11],  # Future: range [8,12), excludes i=10
            11: [9, 10],  # Future: range [9,12), excludes i=11
        }

        # Verify move_to_indices matches expected structure
        for i, expected_destinations in expected_move_to.items():
            assert move_to_indices[i] == expected_destinations, (
                f"move_to_indices[{i}] = {move_to_indices[i]} != "
                f"expected {expected_destinations}"
            )

        # Verify symmetry: build expected get_from_indices from move_to
        expected_get_from = {j: [] for j in ranges.all_indices}
        for i, destinations in expected_move_to.items():
            for j in destinations:
                expected_get_from[j].append(i)

        for j, expected_sources in expected_get_from.items():
            assert get_from_indices[j] == expected_sources, (
                f"get_from_indices[{j}] = {get_from_indices[j]} != "
                f"expected {expected_sources}"
            )

    def test_transfer_matrix_flipped_advance_delay(self) -> None:
        """Test 2: Verify indices with flipped advance/delay parameters.

         This tests the complementary case to Test 1:
         - history_padding = 3 (past_hours)
         - n_hours = 9 (future_hours)
         - max_demand_advance = 3 (flipped from Test 1)
         - max_demand_delay = 2 (flipped from Test 1)

         Expected Transfer Matrix T[i,j] (1 = transfer allowed, 0 = not allowed):

            j→  0  1  2  3  4  5  6  7  8  9 10 11
         i ↓  ┌────────────────────────────────────┐
         0    │ 0  0  0  0  0  0  0  0  0  0  0  0 │ (historical: no transfers)
         1    │ 0  0  0  1  0  0  0  0  0  0  0  0 │ (historical: [3,4))
         2    │ 0  0  0  1  1  0  0  0  0  0  0  0 │ (historical: [3,5))
         3    │ 1  1  1  0  1  1  0  0  0  0  0  0 │ (future: [0,6), excludes i=3)
         4    │ 0  1  1  1  0  1  1  0  0  0  0  0 │ (future: [1,7), excludes i=4)
         5    │ 0  0  1  1  1  0  1  1  0  0  0  0 │ (future: [2,8), excludes i=5)
         6    │ 0  0  0  1  1  1  0  1  1  0  0  0 │ (future: [3,9), excludes i=6)
         7    │ 0  0  0  0  1  1  1  0  1  1  0  0 │ (future: [4,10), excludes i=7)
         8    │ 0  0  0  0  0  1  1  1  0  1  1  0 │ (future: [5,11), excludes i=8)
         9    │ 0  0  0  0  0  0  1  1  1  0  1  1 │ (future: [6,12), excludes i=9)
        10    │ 0  0  0  0  0  0  0  1  1  1  0  1 │ (future: [7,12), excludes i=10)
        11    │ 0  0  0  0  0  0  0  0  1  1  1  0 │ (future: [8,12), excludes i=11)
              └────────────────────────────────────┘
        """
        # Create TimeRanges with n_lookback_hours=3 and n_lookahead_hours=9
        time_ranges_factory = TimeRanges(n_lookback_hours=3)
        ranges = time_ranges_factory.build(n_control_hours=5, n_lookahead_hours=9)

        # Create TransferIndices with flipped parameters
        # max_demand_advance=3, max_demand_delay=2
        transfer_indices_factory = TransferIndices(
            max_demand_advance=3,  # Flipped: is 2 in Test 1
            max_demand_delay=2,  # Flipped: is 3 in Test 1
            n_lookback_hours=3,
        )
        transfer_indices = transfer_indices_factory.build(ranges)
        move_to_indices = transfer_indices.move_to_indices
        get_from_indices = transfer_indices.get_from_indices

        # Expected move_to_indices based on actual behavior (from debugging)
        expected_move_to = {
            0: [],  # Historical: no transfers, range [3,3)
            1: [3],  # Historical: range [3,4), excludes i=1
            2: [3, 4],  # Historical: range [3,5), excludes i=2
            3: [0, 1, 2, 4, 5],  # Future: range [0,6), excludes i=3
            4: [1, 2, 3, 5, 6],  # Future: range [1,7), excludes i=4
            5: [2, 3, 4, 6, 7],  # Future: range [2,8), excludes i=5
            6: [3, 4, 5, 7, 8],  # Future: range [3,9), excludes i=6
            7: [4, 5, 6, 8, 9],  # Future: range [4,10), excludes i=7
            8: [5, 6, 7, 9, 10],  # Future: range [5,11), excludes i=8
            9: [6, 7, 8, 10, 11],  # Future: range [6,12), excludes i=9
            10: [7, 8, 9, 11],  # Future: range [7,12), excludes i=10
            11: [8, 9, 10],  # Future: range [8,12), excludes i=11
        }

        # Verify move_to_indices matches expected structure
        for i, expected_destinations in expected_move_to.items():
            assert move_to_indices[i] == expected_destinations, (
                f"move_to_indices[{i}] = {move_to_indices[i]} != "
                f"expected {expected_destinations}"
            )

        # Verify symmetry: build expected get_from_indices from move_to
        expected_get_from = {j: [] for j in ranges.all_indices}
        for i, destinations in expected_move_to.items():
            for j in destinations:
                expected_get_from[j].append(i)

        for j, expected_sources in expected_get_from.items():
            assert get_from_indices[j] == expected_sources, (
                f"get_from_indices[{j}] = {get_from_indices[j]} != "
                f"expected {expected_sources}"
            )

    def test_spillover_matrix_concrete_example(self) -> None:
        """Test 3: Verify spillover matrix structure and transpose.

        This tests both move_to_spillover_indices and get_from_spillover_indices
        using the same parameters. Spillover indices handle moving horizon cases
        where the control horizon is shorter than the lookahead horizon. The optimal
        demand may result in demand being moved to or obtained from hours outside
        the control window.

        Configuration:
        - max_demand_advance = 2 (can buy energy 2 hours early)
        - max_demand_delay = 3 (can buy energy 3 hours late)
        - n_lookback_hours = 3 (past hours from previous optimization)
        - n_control_hours = 5 (current control period)
        - n_lookahead_hours = 9 (current optimization period)

        Time Index Configuration:
        - Historical indices: [0, 1, 2] (fixed from previous optimization)
        - Control indices: [3, 4, 5, 6, 7] (actively optimized decisions)
        - Lookahead indices: [3, 4, 5, 6, 7, 8, 9, 10, 11] (all future indices)
        - Spillover indices: [5, 6, 7] (last n_lookback_hours of control period)

        Expected Spillover Matrix S[i,j]
        (1 = potential spillover, 0 = no spillover, x = last hour of control horizon):

           j→  0  1  2  3  4  5  6  7  8  9 10 11
        i ↓  ┌────────────────────────────────────┐
        0    │ 0  0  0  0  0  0  0  0  0  0  0  0 │ (not spillover index)
        1    │ 0  0  0  0  0  0  0  0  0  0  0  0 │ (not spillover index)
        2    │ 0  0  0  0  0  0  0  0  0  0  0  0 │ (not spillover index)
        3    │ 0  0  0  0  0  0  0  0  0  0  0  0 │ (not spillover index)
        4    │ 0  0  0  0  0  0  0  0  0  0  0  0 │ (not spillover index)
        5    │ 0  0  0  0  0  0  0  0  1  0  0  0 │ (spillover: can reach j=8)
        6    │ 0  0  0  0  0  0  0  0  1  1  0  0 │ (spillover: can reach j=8,9)
        7    │ 0  0  0  0  0  0  0  x  1  1  1  0 │ (spillover: can reach j=8,9,10)
        8    │ 0  0  0  0  0  0  1  1  0  0  0  0 │ (spillover: can reach j=6,7)
        9    │ 0  0  0  0  0  0  0  1  0  0  0  0 │ (spillover: can reach j=7)
        10   │ 0  0  0  0  0  0  0  0  0  0  0  0 │ (not spillover index)
        11   │ 0  0  0  0  0  0  0  0  0  0  0  0 │ (not spillover index)
             └────────────────────────────────────┘

        Expected move_to_spillover_indices (spillover destinations):
        {
            5: [8],              # index 5 might move demand to [8]
            6: [8, 9],           # index 6 might move demand to [8, 9]
            7: [8, 9, 10],       # index 7 might move demand to [8, 9, 10]
        }

        Expected get_from_spillover_indices (spillover sources):
        {
            5: [],               # index 5 might get demand from []
            6: [8],              # index 6 might get demand from [8]
            7: [8, 9],           # index 7 might get demand from [8, 9]
        }

        SPILLOVER CONSTRAINTS:
        ======================
        - Only spillover indices (last n_lookback_hours of control period) participate
        - Control period spillover indices can only transfer to/from lookahead period
        - Lookahead period spillover indices can only transfer to/from control period
        - Self-transfers prohibited: S[i,i] = 0
        """
        # Create TimeRanges with n_lookback_hours=3 and n_lookahead_hours=9
        time_ranges_factory = TimeRanges(n_lookback_hours=3)
        ranges = time_ranges_factory.build(n_lookahead_hours=9, n_control_hours=5)

        # Create TransferIndices with max_demand_advance=2, max_demand_delay=3
        transfer_indices_factory = TransferIndices(
            max_demand_advance=2, max_demand_delay=3, n_lookback_hours=3
        )
        transfer_indices = transfer_indices_factory.build(ranges)
        move_to_spillover_indices = transfer_indices.move_to_spillover_indices
        get_from_spillover_indices = transfer_indices.get_from_spillover_indices

        # Expected move_to_spillover_indices based on spillover matrix
        # Spillover indices [5, 6, 7] can only transfer to lookahead period
        # [8, 9, 10, 11]
        expected_spillover_move_to = {
            5: [8],  # Can spillover to j=8 (within delay limit: 8 <= 5+3)
            6: [8, 9],  # Can spillover to j=8,9 (within delay limit: 9 <= 6+3)
            7: [8, 9, 10],  # Can spillover to j=8,9,10 (within delay limit: 10 <= 7+3)
        }

        # Expected get_from_spillover_indices (transpose/inverse of spillover matrix)
        # This shows which spillover indices can receive demand from lookahead sources
        expected_spillover_get_from = {
            5: [],  # index 5 can receive spillover from no sources
            6: [8],  # index 6 can receive spillover from [8]
            7: [8, 9],  # index 7 can receive spillover from [8, 9]
        }

        # Verify move_to_spillover_indices matches expected structure
        for i, expected_destinations in expected_spillover_move_to.items():
            assert move_to_spillover_indices[i] == expected_destinations, (
                f"move_to_spillover_indices[{i}] = {move_to_spillover_indices[i]} != "
                f"expected {expected_destinations}"
            )

        # Verify get_from_spillover_indices matches expected structure
        for i, expected_sources in expected_spillover_get_from.items():
            assert get_from_spillover_indices[i] == expected_sources, (
                f"get_from_spillover_indices[{i}] = {get_from_spillover_indices[i]} != "
                f"expected {expected_sources}"
            )

        # Verify only spillover indices are included in both mappings
        assert set(move_to_spillover_indices.keys()) == {5, 6, 7}, (
            f"Expected spillover indices {{5, 6, 7}} in move_to, "
            f"got {set(move_to_spillover_indices.keys())}"
        )
        assert set(get_from_spillover_indices.keys()) == {5, 6, 7}, (
            f"Expected spillover indices {{5, 6, 7}} in get_from, "
            f"got {set(get_from_spillover_indices.keys())}"
        )

    def test_spillover_matrix_flipped_advance_delay(self) -> None:
        """Test 4: Verify spillover matrix with flipped advance/delay parameters.

        This tests both move_to_spillover_indices and get_from_spillover_indices
        using flipped parameters compared to Test 3. This complements the main
        spillover test to ensure proper handling of different constraint configurations.

        Configuration:
        - max_demand_advance = 3 (can buy energy 3 hours early) [flipped from Test 3]
        - max_demand_delay = 2 (can buy energy 2 hours late) [flipped from Test 3]
        - n_lookback_hours = 3 (past hours from previous optimization)
        - n_control_hours = 5 (current control period)
        - n_lookahead_hours = 9 (current optimization period)

        Time Index Configuration:
        - Historical indices: [0, 1, 2] (fixed from previous optimization)
        - Control indices: [3, 4, 5, 6, 7] (actively optimized decisions)
        - Lookahead indices: [3, 4, 5, 6, 7, 8, 9, 10, 11] (all future indices)
        - Spillover indices: [5, 6, 7] (last n_lookback_hours of control period)

        Expected Spillover Matrix S[i,j] with flipped constraints
        (1 = potential spillover, 0 = no spillover, x = last hour of control horizon):

           j→  0  1  2  3  4  5  6  7  8  9 10 11
        i ↓  ┌────────────────────────────────────┐
        0    │ 0  0  0  0  0  0  0  0  0  0  0  0 │ (not spillover index)
        1    │ 0  0  0  0  0  0  0  0  0  0  0  0 │ (not spillover index)
        2    │ 0  0  0  0  0  0  0  0  0  0  0  0 │ (not spillover index)
        3    │ 0  0  0  0  0  0  0  0  0  0  0  0 │ (not spillover index)
        4    │ 0  0  0  0  0  0  0  0  0  0  0  0 │ (not spillover index)
        5    │ 0  0  0  0  0  0  0  0  0  0  0  0 │ (spillover: no valid destinations)
        6    │ 0  0  0  0  0  0  0  0  1  0  0  0 │ (spillover: can reach j=8)
        7    │ 0  0  0  0  0  0  0  x  1  1  0  0 │ (spillover: can reach j=8,9)
        8    │ 0  0  0  0  0  1  1  1  0  0  0  0 │ (spillover: can reach j=5,6,7)
        9    │ 0  0  0  0  0  0  1  1  0  0  0  0 │ (spillover: can reach j=6,7)
        10   │ 0  0  0  0  0  0  0  1  0  0  0  0 │ (spillover: can reach j=7)
        11   │ 0  0  0  0  0  0  0  0  0  0  0  0 │ (not spillover index)
             └────────────────────────────────────┘

        Expected move_to_spillover_indices (spillover destinations):
        {
            5: [],               # index 5 cannot reach lookahead period (8 > 5+2)
            6: [8],              # index 6 can reach j=8 (within delay: 8 <= 6+2)
            7: [8, 9],           # index 7 can reach j=8,9 (within delay: 9 <= 7+2)
        }

        Expected get_from_spillover_indices (spillover sources):
        {
            5: [8],              # index 5 can receive from lookahead j=8 (8 <= 5+3)
            6: [8, 9],           # index 6 can receive from lookahead j=8,9 (9 <= 6+3)
            7: [8, 9, 10],  # index 7 can receive from lookahead j=8,9,10 (10 <= 7+3)
        }

        SPILLOVER CONSTRAINTS (Flipped):
        =================================
        - Reduced delay window (2 vs 3) limits move_to destinations
        - Increased advance window (3 vs 2) expands get_from sources
        - Control/lookahead transfer directions remain the same
        - Self-transfers prohibited: S[i,i] = 0
        """
        # Create TimeRanges with n_lookback_hours=3 and n_lookahead_hours=9
        time_ranges_factory = TimeRanges(n_lookback_hours=3)
        ranges = time_ranges_factory.build(n_lookahead_hours=9, n_control_hours=5)

        # Create TransferIndices with flipped parameters
        # max_demand_advance=3, max_demand_delay=2 (flipped from Test 3)
        transfer_indices_factory = TransferIndices(
            max_demand_advance=3,  # Flipped: is 2 in Test 3
            max_demand_delay=2,  # Flipped: is 3 in Test 3
            n_lookback_hours=3,
        )
        transfer_indices = transfer_indices_factory.build(ranges)
        move_to_spillover_indices = transfer_indices.move_to_spillover_indices
        get_from_spillover_indices = transfer_indices.get_from_spillover_indices

        # Expected move_to_spillover_indices with reduced delay window
        # Spillover indices [5, 6, 7] have reduced reach to lookahead period
        expected_spillover_move_to = {
            5: [],  # Cannot reach lookahead: min(8) > 5+2
            6: [8],  # Can reach j=8 (within delay limit: 8 <= 6+2)
            7: [8, 9],  # Can reach j=8,9 (within delay limit: 9 <= 7+2)
        }

        # Expected get_from_spillover_indices with increased advance window
        # Spillover indices can receive from more lookahead sources
        expected_spillover_get_from = {
            5: [8],  # index 5 can receive from j=8 (8 <= 5+3)
            6: [8, 9],  # index 6 can receive from j=8,9 (9 <= 6+3)
            7: [8, 9, 10],  # index 7 can receive from j=8,9,10 (10 <= 7+3)
        }

        # Verify move_to_spillover_indices matches expected structure
        for i, expected_destinations in expected_spillover_move_to.items():
            assert move_to_spillover_indices[i] == expected_destinations, (
                f"move_to_spillover_indices[{i}] = {move_to_spillover_indices[i]} != "
                f"expected {expected_destinations}"
            )

        # Verify get_from_spillover_indices matches expected structure
        for i, expected_sources in expected_spillover_get_from.items():
            assert get_from_spillover_indices[i] == expected_sources, (
                f"get_from_spillover_indices[{i}] = {get_from_spillover_indices[i]} != "
                f"expected {expected_sources}"
            )

        # Verify only spillover indices are included in both mappings
        assert set(move_to_spillover_indices.keys()) == {5, 6, 7}, (
            f"Expected spillover indices {{5, 6, 7}} in move_to, "
            f"got {set(move_to_spillover_indices.keys())}"
        )
        assert set(get_from_spillover_indices.keys()) == {5, 6, 7}, (
            f"Expected spillover indices {{5, 6, 7}} in get_from, "
            f"got {set(get_from_spillover_indices.keys())}"
        )
