class TimeRanges:
    """Manages time indexing for optimization problems with historical constraints.

    This class handles the dual indexing system needed for moving-horizon optimization:

    INDEXING SYSTEM:
    ===============

    Two coordinate systems are used:
    1. LOCAL TIME (t): Hours relative to current time [0, n_lookahead_hours-1]
    2. GLOBAL INDEX (i,j): Absolute array positions [0, total_hours-1]

    Timeline Structure with Control/Lookahead Hierarchy:
    --------------------------------------------------------------------------------
    Global Index:  [0, 1, 2] [3, 4, 5, 6, 7, 8, 9, 10, 11]
    Lookback:      [0, 1, 2]
    Lookahead:               [3, 4, 5, 6, 7, 8, 9, 10, 11]
    Control:                 [3, 4, 5, 6, 7]
    Spillover:                     [5, 6, 7]
    Local Time:              [0, 1, 2, 3, 4, 5, 6,  7,  8]

    CONCRETE EXAMPLE (n_lookback_hours=3, n_control_hours=5, n_lookahead_hours=9):
    =============================================================================

    Time Periods:
    - Lookback indices: [0,1,2] (fixed from previous optimization)
    - Control indices: [3,4,5,6,7] (actively optimized decision variables)
    - Lookahead indices: [3,4,5,6,7,8,9,10,11] (all future indices including control)
    - Spillover indices: [5,6,7]
      (range(n_control_hours, n_control_hours + n_lookback_hours))
    - Local time: [0,1,2,3,4,5,6,7,8] (hours from present moment)

    Index Conversion:
    - Local time t=0 (now) → Global index i=3 (start of control period)

    Spillover Logic: The spillover indices represent hours that can transfer energy
    to or from non-control lookahead hours.

    Args:
        n_lookback_hours: Number of historical hours used as constraints.

    Note:
        The n_control_hours and n_lookahead_hours attributes are set at build time
        via the build() method, along with the associated properties that
        depend on them.
    """

    def __init__(self, n_lookback_hours: int) -> None:
        if n_lookback_hours < 0:
            raise ValueError("n_lookback_hours must be non-negative")
        self.n_lookback_hours = n_lookback_hours

    def build(self, n_control_hours: int, n_lookahead_hours: int) -> "TimeRanges":
        """Build a new TimeRanges instance with control/lookahead split.

        Args:
            n_control_hours: Number of hours in the control period (actively optimized).
            n_lookahead_hours: Total number of future hours (control + spillover).

        Returns:
            TimeRanges instance with all time periods configured.

        Raises:
            ValueError: If n_control_hours or n_lookahead_hours is negative.
            ValueError: If n_control_hours > n_lookahead_hours.
        """
        if n_control_hours < 0:
            raise ValueError("n_control_hours must be non-negative")
        if n_lookahead_hours < 0:
            raise ValueError("n_lookahead_hours must be non-negative")
        if n_control_hours > n_lookahead_hours:
            raise ValueError(
                f"n_control_hours ({n_control_hours}) must be less than or equal to "
                f"n_lookahead_hours ({n_lookahead_hours})"
            )

        # Create a new instance with both dimensions
        instance = TimeRanges(n_lookback_hours=self.n_lookback_hours)
        instance.n_control_hours = n_control_hours
        instance.n_lookahead_hours = n_lookahead_hours
        return instance

    def time_to_global_index(self, t: int) -> int:
        return t + self.n_lookback_hours

    @property
    def local_time(self) -> range:
        """Local time indices [0, n_lookahead_hours-1] for current optimization."""
        return range(self.n_lookahead_hours)

    @property
    def lookback_indices(self) -> range:
        """Historical global indices [0, n_lookback_hours-1] from previous
        optimization."""
        return range(self.n_lookback_hours)

    @property
    def lookahead_indices(self) -> range:
        """All future global indices (includes control period)."""
        return range(
            self.n_lookback_hours, self.n_lookback_hours + self.n_lookahead_hours
        )

    @property
    def spillover_indices(self) -> range:
        """Last n_lookback_hours of control or lookback period that can spill
        effects forward."""
        return range(self.n_control_hours, self.n_control_hours + self.n_lookback_hours)

    @property
    def control_indices(self) -> range:
        """Control global indices for active optimization decisions."""
        return range(
            self.n_lookback_hours, self.n_lookback_hours + self.n_control_hours
        )

    @property
    def all_indices(self) -> range:
        """All global indices (lookback + lookahead periods)."""
        return range(self.n_lookback_hours + self.n_lookahead_hours)
