"""Transfer matrix index mappings for virtual storage optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .time_ranges import TimeRanges


class TransferIndices:
    """Builds transfer matrix index mappings for virtual storage optimization.

     This class determines which energy transfers T[i,j] are allowed in the transfer
     matrix, where energy originally demanded at time i can be purchased at time j
     to minimize costs while respecting physical timing constraints.

     CONCRETE EXAMPLE:
     ================

     Configuration:
     - max_demand_advance = 2 (can buy energy 2 hours early)
     - max_demand_delay = 3 (can buy energy 3 hours late)
     - n_lookback_hours = 3 (past hours from previous optimization)
     - n_control_hours = 5 (current control period)
     - n_lookahead_hours = 9 (current optimization period)

     Expected Transfer Matrix T[i,j] (1 = transfer allowed, 0 = not allowed):

        j→  0  1  2  3  4  5  6  7  8  9 10 11
     i ↓  ┌────────────────────────────────────┐
     0    │ 0  0  0  1  0  0  0  0  0  0  0  0 │ (historical: constrained to j≥3)
     1    │ 0  0  0  1  1  0  0  0  0  0  0  0 │ (historical: constrained to j≥3)
     2    │ 0  0  0  1  1  1  0  0  0  0  0  0 │ (historical: constrained to j≥3)
     3    │ 0  1  1  0  1  1  1  0  0  0  0  0 │ (future: range [1,7), excludes i=3)
     4    │ 0  0  1  1  0  1  1  1  0  0  0  0 │ (future: range [2,8), excludes i=4)
     5    │ 0  0  0  1  1  0  1  1  1  0  0  0 │ (future: range [3,9), excludes i=5)
     6    │ 0  0  0  0  1  1  0  1  1  1  0  0 │ (future: range [4,10), excludes i=6)
     7    │ 0  0  0  0  0  1  1  0  1  1  1  0 │ (future: range [5,11), excludes i=7)
     8    │ 0  0  0  0  0  0  1  1  0  1  1  1 │ (future: range [6,12), excludes i=8)
     9    │ 0  0  0  0  0  0  0  1  1  0  1  1 │ (future: range [7,12), excludes i=9)
    10    │ 0  0  0  0  0  0  0  0  1  1  0  1 │ (future: range [8,12), excludes i=10)
    11    │ 0  0  0  0  0  0  0  0  0  1  1  0 │ (future: range [9,12), excludes i=11)
          └────────────────────────────────────┘

     Example move_to_indices result:
     {
         0: [3],                    # Can move energy from hour 0 to hour 3
         1: [3, 4],                 # Can move energy from hour 1 to hours 3,4
         2: [3, 4, 5],              # Can move energy from hour 2 to hours 3,4,5
         3: [1, 2, 4, 5, 6],        # Can move energy from hour 3 to hours 1,2,4,5,6
         4: [2, 3, 5, 6, 7],        # Can move energy from hour 4 to hours 2,3,5,6,7
         ...
     }

     Example get_from_indices result:
     {
         0: [],                     # Hour 0 cannot receive energy from any source
         1: [3],                    # Hour 1 can receive energy from hour 3
         2: [3, 4],                 # Hour 2 can receive energy from hours 3,4
         3: [0, 1, 2, 4],           # Hour 3 can receive energy from hours 0,1,2,4
         4: [1, 2, 3, 5],           # Hour 4 can receive energy from hours 1,2,3,5
         ...
     }

     SPILLOVER SUBSET MATRIX:
     =========================

     Spillover indices handles moving horizon cases stemming from the control horizon
     being shorter than the lookahead horizon. In such cases, the optimal demand may
     result in demand being moved to or obtained from hours that are outside the
     control window. Such spillover demand has to be accounted for and the spillover
     indices indicate which elements in the transfer matrix that may move or obtain
     demand from hours outside the control window.

     Spillover Transfer Matrix S[i,j]
     (1 = potential spillover, 0 = no spillover, x = last hour of the control horizon):

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
    10    │ 0  0  0  0  0  0  0  0  0  0  0  0 │ (not spillover index)
    11    │ 0  0  0  0  0  0  0  0  0  0  0  0 │ (not spillover index)
          └────────────────────────────────────┘

    Example move_to_spillover_indices result:
    {
        5: [8],                 # index 5 might move demand to [8]
        6: [8, 9],              # index 6 might move demand to [8, 9]
        7: [8, 9, 10],          # index 7 might move demand to [8, 9, 10]
    }

    Example get_from_spillover_indices result:
    {
        5: [],                  # index 5 might get demand from []
        6: [8],                 # index 6 might get demand from [8]
        7: [8, 9],              # index 7 might get demand from [8, 9]
    }

     TRANSFER MATRIX CONSTRAINTS:
     ============================
     - Historical periods (i < n_lookback_hours): Only forward transfers
       j ≥ n_lookback_hours
     - Future periods (i ≥ n_lookback_hours): Bidirectional transfers within
       advance/delay windows
     - Self-transfers prohibited: T[i,i] = 0
    """

    def __init__(
        self, max_demand_advance: int, max_demand_delay: int, n_lookback_hours: int
    ):
        self.max_demand_advance = max_demand_advance
        self.max_demand_delay = max_demand_delay
        self.n_lookback_hours = n_lookback_hours

    def build(self, ranges: TimeRanges) -> TransferIndices:
        """Build transfer matrix index mappings for virtual storage optimization.

        Args:
            ranges: TimeRanges object containing time index definitions.

        Returns:
            TransferIndices instance with computed T[i,j] mappings.
        """
        # Create new instance with all the configuration
        instance = TransferIndices(
            max_demand_advance=self.max_demand_advance,
            max_demand_delay=self.max_demand_delay,
            n_lookback_hours=self.n_lookback_hours,
        )
        # Add the computed mappings
        instance.move_to_indices = self.build_move_to_indices(ranges)
        instance.get_from_indices = self.build_get_from_indices(ranges)
        instance.move_to_spillover_indices = self.build_move_to_spillover_indices(
            ranges
        )
        instance.get_from_spillover_indices = self.build_get_from_spillover_indices(
            ranges
        )
        return instance

    def build_move_to_indices(self, ranges: TimeRanges) -> dict[int, list[int]]:
        """Build valid T[i,j] destination indices for energy transfers.

        For each source index i, determines valid destination indices j where
        energy originally demanded at time i can be purchased at time j.

        Args:
            ranges: TimeRanges object containing time index definitions.

        Returns:
            Dictionary mapping source index i to list of valid destination indices j.
        """
        return {
            i: [
                j
                for j in range(*self._get_j_range_for_i(i, ranges.n_lookahead_hours))
                if j != i
            ]
            for i in ranges.all_indices
        }

    def build_move_to_spillover_indices(
        self, ranges: TimeRanges
    ) -> dict[int, list[int]]:
        """Build spillover T[i,j] destination indices for control period transfers.

        Args:
            ranges: TimeRanges object containing time index definitions.

        Returns:
            Dictionary mapping spillover source indices to lookahead destinations.
        """
        return {
            i: [
                j
                for j in range(
                    *self._get_spillover_j_range_for_i(
                        i, ranges.n_control_hours, ranges.n_lookahead_hours
                    )
                )
                if j != i
            ]
            for i in ranges.spillover_indices
        }

    def build_get_from_indices(self, ranges: TimeRanges) -> dict[int, list[int]]:
        """Build valid T[i,j] source indices for energy transfers.

        For each destination index j, determines valid source indices i where
        energy can be transferred from time i to be purchased at time j.
        This is the inverse mapping of build_move_to_indices.

        Args:
            ranges: TimeRanges object containing time index definitions.

        Returns:
            Dictionary mapping destination index j to list of valid source indices i.
        """
        return {
            j: [
                i
                for i in range(*self._get_i_range_for_j(j, ranges.n_lookahead_hours))
                if i != j
            ]
            for j in ranges.all_indices
        }

    def build_get_from_spillover_indices(
        self, ranges: TimeRanges
    ) -> dict[int, list[int]]:
        """Build spillover T[i,j] source indices for control period transfers.

        Args:
            ranges: TimeRanges object containing time index definitions.

        Returns:
            Dictionary mapping spillover destination indices to lookahead sources.
        """
        return {
            i: [
                j
                for j in range(
                    *self._get_spillover_i_range_for_j(
                        i, ranges.n_control_hours, ranges.n_lookahead_hours
                    )
                )
                if j != i
            ]
            for i in ranges.spillover_indices
        }

    def _get_j_range_for_i(self, i: int, n_lookahead_hours: int) -> tuple[int, int]:
        """Get valid T[i,j] destination range for source index i.

        Args:
            i: Source time index for energy transfer.
            n_lookahead_hours: Total lookahead hours in optimization.

        Returns:
            Tuple (start, end) for valid destination indices j.
        """
        if i < self.n_lookback_hours:
            return (
                max(
                    self.n_lookback_hours,
                    i - self.max_demand_advance,
                ),
                min(
                    n_lookahead_hours + self.n_lookback_hours,
                    i + self.max_demand_delay + 1,
                ),
            )
        else:
            return (
                max(0, i - self.max_demand_advance),
                min(
                    n_lookahead_hours + self.n_lookback_hours,
                    i + self.max_demand_delay + 1,
                ),
            )

    def _get_i_range_for_j(self, j: int, n_lookahead_hours: int) -> tuple[int, int]:
        """Get valid T[i,j] source range for destination index j.

        Args:
            j: Destination time index for energy transfer.
            n_lookahead_hours: Total lookahead hours in optimization.

        Returns:
            Tuple (start, end) for valid source indices i.
        """
        if j < self.n_lookback_hours:
            return (
                max(
                    self.n_lookback_hours,
                    j - self.max_demand_delay,
                ),
                min(
                    n_lookahead_hours + self.n_lookback_hours,
                    j + self.max_demand_advance + 1,
                ),
            )
        else:
            return (
                max(0, j - self.max_demand_delay),
                min(
                    n_lookahead_hours + self.n_lookback_hours,
                    j + self.max_demand_advance + 1,
                ),
            )

    def _get_spillover_j_range_for_i(
        self, i: int, n_control_hours: int, n_lookahead_hours: int
    ) -> tuple[int, int]:
        """Get spillover T[i,j] destination range for control period source i.

        Args:
            i: Control period source index for spillover transfer.
            n_control_hours: Number of control hours in optimization.
            n_lookahead_hours: Total lookahead hours in optimization.

        Returns:
            Tuple (start, end) for valid spillover destination indices j.
        """
        return (
            n_control_hours + self.n_lookback_hours,
            min(
                n_lookahead_hours + self.n_lookback_hours,
                i + self.max_demand_delay + 1,
            ),
        )

    def _get_spillover_i_range_for_j(
        self, j: int, n_control_hours: int, n_lookahead_hours: int
    ) -> tuple[int, int]:
        """Get spillover T[i,j] source range for control period destination j.

        Args:
            j: Control period destination index for spillover transfer.
            n_control_hours: Number of control hours in optimization.
            n_lookahead_hours: Total lookahead hours in optimization.

        Returns:
            Tuple (start, end) for valid spillover source indices i.
        """
        return (
            n_control_hours + self.n_lookback_hours,
            min(
                n_lookahead_hours + self.n_lookback_hours,
                j + self.max_demand_advance + 1,
            ),
        )
