"""Virtual storage optimization for demand response using MILP.

Supports both python-mip (open-source) and Gurobi (commercial) solvers.
"""

from typing import Any, Dict, Optional

import numpy as np

from .solver_adapters import create_solver_adapter
from .time_ranges import TimeRanges
from .transfer_indices import TransferIndices
from .utils import get_logger

logger = get_logger(__name__)


class VirtualStorage:
    """Virtual storage system for energy demand response optimization.

    This class models demand response as a virtual battery that can store and
    release energy across time to minimize total purchasing costs. The storage
    size is constrained implicitly through max_demand_advance, max_demand_delay,
    and max_rate. There is therefore no need to model and constrain the storage
    level separately.

    OBJECTIVE FUNCTION:
    ==================
    Minimize: Σ[purchase[t] * price[t]]

    Where:
    - purchase[t] = actual energy purchased at time t
    - price[t] = spot price (e.g., day-ahead market price)

    TRANSFER MATRIX CONCEPT:
    =======================

    The optimization uses a transfer matrix T[i,j] to determine how demand should
    be modified to lower the total cost. The allowed indices indicate between
    which hours energy can be moved around.

    T[i,j] = Amount of energy originally demanded at time i,
             but actually purchased at time j to lower costs.

    MATHEMATICAL INTERPRETATION:
    ---------------------------

    Original demand: D = [d₀, d₁, d₂, d₃]  (kWh)
    Energy prices:   P = [p₀, p₁, p₂, p₃]  (ct/kWh)

    Transfer Matrix T:
         j→  0   1   2   3  (purchase times)
    i ↓  ┌─────────────────┐
    0    │ 0  t₀₁ t₀₂  0   │
    1    │t₁₀  0  t₁₂ t₁₃  │
    2    │ 0  t₂₁  0  t₂₃  │
    3    │ 0   0  t₃₂  0   │
         └─────────────────┘

    Row sums: Σⱼ T[i,j] = total energy removed from hour i in original demand profile
    Column sums: Σᵢ T[i,j] = total energy added to hour j compared to original
        demand profile

    COST OPTIMIZATION EXAMPLE:
    --------------------------
    Original:  [10, 15, 8, 12] kWh demand
    Prices:    [30, 80, 20, 40] ct/kWh
    Original cost: 10*30 + 15*80 + 8*20 + 12*40 = 2140 ct

    Optimal strategy: Move expensive hour 1 demand to cheap hour 2
    T[1,2] = 10 kWh (move 10 kWh from hour 1 to hour 2)

    Modified purchases: [10, 5, 18, 12] kWh
    New cost: 10*30 + 5*80 + 18*20 + 12*40 = 1540 ct
    Savings: 600 ct (28% reduction)

    CONSTRAINTS ON ENERGY MOVEMENT:
    ------------------------------
    - Non-negativity: T[i,j] ≥ 0 (all transfer matrix elements must be positive)
    - Row sum ≤ original demand: Σⱼ T[i,j] ≤ dᵢ (can't move more than available)
    - Transfer limits: Only between hours within max_demand_advance/delay windows
    - Transfer rates: Energy movement ≤ max_rate per hour
    - Diagonal = 0: T[i,i] = 0 (can't transfer to same time)
    - Mutual exclusivity: Can either add to OR remove from storage during an
        hour, not both
    """

    def __init__(
        self,
        max_demand_advance: int,
        max_demand_delay: int,
        max_hourly_purchase: float,
        max_rate: float,
        enforce_charge_direction: bool = False,
        solver: str = "auto",
    ) -> None:
        """Initialize the virtual storage object.

        Args:
            max_demand_advance: Hours before demand energy can be purchased (hours).
            max_demand_delay: Hours after demand energy can be purchased (hours).
            max_hourly_purchase: Maximum energy that can be purchased in a
                single hour (kWh).
            max_rate: Maximum rate at which energy can be added or removed (kW).
            enforce_charge_direction: If True, enforce mutual exclusivity between
                charging and discharging (default False). Setting to False allows
                simultaneous add_to and remove_from, which may speed up optimization
                but could produce less physically realistic results.
            solver: Solver backend to use. Options:
                - "auto": Try Gurobi first, fall back to MIP (default)
                - "gurobi": Use Gurobi (requires gurobipy)
                - "mip": Use python-mip with CBC solver

        Raises:
            ValueError: If max_demand_advance or max_demand_delay is negative.
            ValueError: If max_hourly_purchase or max_rate is negative or zero.
            ValueError: If solver is not one of "auto", "gurobi", "mip".
            ImportError: If requested solver is not available.
        """

        self.max_demand_advance = max_demand_advance
        self.max_demand_delay = max_demand_delay
        self.max_hourly_purchase = max_hourly_purchase
        self.max_rate = max_rate
        self.big_m = max_rate
        self.enforce_charge_direction = enforce_charge_direction

        # Handle historical items in storage for moving horizon dynamics
        self.n_lookback_hours = max(self.max_demand_advance, self.max_demand_delay)

        # Create helper instances to avoid recreating them in optimize_demand
        self.time_ranges = TimeRanges(n_lookback_hours=self.n_lookback_hours)
        self.transfer_indices = TransferIndices(
            max_demand_advance=self.max_demand_advance,
            max_demand_delay=self.max_demand_delay,
            n_lookback_hours=self.n_lookback_hours,
        )

        # Create solver adapter
        self._solver = create_solver_adapter(solver)

    def optimize_demand(
        self,
        price: np.ndarray,
        demand: np.ndarray,
        remove_from_history: Optional[np.ndarray] = None,
        add_to_history: Optional[np.ndarray] = None,
        n_control_hours: Optional[int] = None,
        debug: bool = False,
    ) -> dict:
        """Optimize energy consumption using the transfer matrix approach.

        Implements the MILP optimization to find the optimal T[i,j] transfer matrix
        that minimizes total purchasing costs. Uses the constraints on energy
        movement to ensure physical feasibility.

        Timeline and Indexing:
        ---------------------
        - Global indices i,j: Used for transfer matrix T[i,j] (as in class docstring)
        - Local time t: Hours from present [0, n_lookahead_hours-1] for current
          optimization
        - Historical period: [0...n_lookback_hours-1] fixed from previous optimization
        - Optimization period: [n_lookback_hours...end] current decision variables

        The method solves for T[i,j] values that represent energy originally demanded
        at global time i but purchased at global time j to achieve cost savings.

        Args:
            price: Array of energy prices per interval (ct/kWh).
            demand: Array of energy demand per interval (kWh).
            remove_from_history: Historical energy removals for continuity (kWh).
                Defaults to zeros if None.
            add_to_history: Historical energy additions for continuity (kWh).
                Defaults to zeros if None.
            n_control_hours: Number of intervals to actively optimize.
                If None, optimizes the entire lookahead horizon.
                Must be <= len(demand).
            debug: If True, include detailed debug information in return dict.
                Defaults to False.

        Returns:
            Dictionary with the following keys:
            - "optimal_demand": Array of optimized demand for control period (kWh)
            - "optimal_shift": Array of demand shifts (add_to - remove_from) for
              control period (kWh)
            - "remove_spillover": Energy removed from current period that spills
              to future periods (kWh)
            - "add_spillover": Energy added to current period from past periods
              that spills over (kWh)
            - "debug_info": Dictionary containing detailed optimization variables
              (only present if debug=True):
                - "purchase": Full purchase array for all lookahead hours (kWh)
                - "add_to": Energy added to storage per hour (kWh)
                - "remove_from": Energy removed from storage per hour (kWh)
                - "charge_direction": Binary charge direction
                  (1=charging, 0=discharging)
                - "transfer_matrix": Full T[i,j] transfer matrix

        Raises:
            ValueError: If price and demand arrays have different lengths.
            ValueError: If input arrays are empty or contain invalid values (NaN, inf).
            ValueError: If remove_from_history or add_to_history arrays have
                incorrect size.
            TypeError: If remove_from_history or add_to_history are not numpy
                arrays when provided.

        Note:
            If the solver returns non-optimal status (infeasible, unbounded,
            not solved),
            warnings are logged but the method continues execution. In such cases:
            - obj_value will be float('inf') for failed optimizations
            - optimized_demand will contain zeros or partial solutions
            - Check logs for specific failure reasons and constraint suggestions
        """

        # Validate and initialize optional arrays
        remove_from_history = self._validate_optional_array(
            remove_from_history, self.n_lookback_hours, "remove_from_history"
        )
        add_to_history = self._validate_optional_array(
            add_to_history, self.n_lookback_hours, "add_to_history"
        )

        # Initialize and validate time parameters
        n_lookahead_hours, n_control_hours = self._init_time_params(
            price, demand, n_control_hours
        )

        # Build time ranges and transfer indices
        ranges = self.time_ranges.build(n_control_hours, n_lookahead_hours)
        transfer_indices = self.transfer_indices.build(ranges)

        # Create the MILP problem using solver adapter
        model = self._solver.create_model()

        # Create decision variables
        variables = self._create_decision_variables(model, ranges, transfer_indices)

        # Add objective function
        self._add_objective_function(model, variables, price, ranges)

        # Add constraints
        self._add_spillover_constraints(
            model,
            variables,
            ranges,
            transfer_indices,
            remove_from_history,
            add_to_history,
        )
        self._add_control_constraints(
            model, variables, ranges, transfer_indices, demand
        )

        if self.enforce_charge_direction:
            self._add_charge_direction_constraints(model, variables, ranges)

        # Solve the problem
        self._solver.solve(model)

        # Determine potential spillover to lookahead hours beyond the control horizon
        remove_from_spillover, add_to_spillover = self._compute_spillover(
            variables, transfer_indices
        )

        # Extract optimization results
        s = self._solver
        optimal_demand = np.array(
            [s.get_value(variables["purchase"][t]) for t in range(n_control_hours)]
        )

        # Compute optimal_shift for control period (add_to - remove_from)
        add_to_control = np.array(
            [s.get_value(variables["add_to"][t]) for t in range(n_control_hours)]
        )
        remove_from_control = np.array(
            [s.get_value(variables["remove_from"][t]) for t in range(n_control_hours)]
        )
        optimal_shift = add_to_control - remove_from_control

        # Build return dictionary
        result = {
            "optimal_demand": optimal_demand,
            "optimal_shift": optimal_shift,
            "remove_spillover": remove_from_spillover,
            "add_spillover": add_to_spillover,
        }

        # Add debug info if requested
        if debug:
            debug_info = self._extract_debug_info(variables, ranges, transfer_indices)
            result["debug_info"] = debug_info

        return result

    # ============================================================================
    # DECISION VARIABLES
    # ============================================================================

    def _create_decision_variables(
        self,
        model: Any,
        ranges: TimeRanges,
        transfer_indices: TransferIndices,
    ) -> Dict[str, dict]:
        """Create all decision variables for the transfer matrix optimization.

        Uses the solver adapter to create variables in a solver-agnostic way.

        Args:
            model: The optimization model instance.
            ranges: TimeRanges object for time indexing.
            transfer_indices: TransferIndices object with computed mappings.
        """
        s = self._solver  # shorthand
        variables = {}

        # Transfer Matrix T[i,j]
        variables["transfer"] = {
            (i, j): s.add_var(model, f"T_{i}_{j}", lb=0)
            for i in ranges.all_indices
            for j in transfer_indices.move_to_indices[i]
        }

        # Purchase variables
        variables["purchase"] = {
            t: s.add_var(model, f"purchase_{t}", lb=0, ub=self.max_hourly_purchase)
            for t in ranges.local_time
        }

        # Storage variables
        variables["remove_from"] = {
            t: s.add_var(model, f"remove_from_{t}", lb=0, ub=self.max_rate)
            for t in ranges.local_time
        }
        variables["add_to"] = {
            t: s.add_var(model, f"add_to_{t}", lb=0, ub=self.max_rate)
            for t in ranges.local_time
        }

        # Charge direction binary (only if enforcing mutual exclusivity)
        if self.enforce_charge_direction:
            variables["charge_direction"] = {
                t: s.add_var(model, f"charge_dir_{t}", vtype="binary")
                for t in ranges.local_time
            }

        return variables

    # ============================================================================
    # OBJECTIVE FUNCTION AND CONSTRAINTS
    # ============================================================================

    def _add_objective_function(
        self,
        model: Any,
        variables: Dict[str, dict],
        price: np.ndarray,
        ranges: TimeRanges,
    ) -> None:
        """Minimize total cost: purchase*price.

        Implements cost minimization objective.

        Args:
            model: The optimization model instance.
            variables: Dictionary of decision variables.
            price: Array of energy prices per interval (ct/kWh).
            ranges: TimeRanges object for time indexing.
        """
        s = self._solver

        # Minimize purchase cost
        objective = s.sum(
            [variables["purchase"][t] * price[t] for t in ranges.local_time]
        )

        s.set_objective(model, objective, sense="minimize")

    def _add_spillover_constraints(
        self,
        model: Any,
        variables: Dict[str, dict],
        ranges: TimeRanges,
        transfer_indices: TransferIndices,
        remove_from_history: np.ndarray,
        add_to_history: np.ndarray,
    ) -> None:
        """Add constraints for historical period continuity.

        Ensures continuity between optimization periods by constraining the
        T[i,j] transfer matrix elements for historical times to match the
        previous optimization results. This maintains consistency in the
        moving-horizon optimization approach.

        Args:
            model: The optimization model instance.
            variables: Dictionary of decision variables.
            ranges: TimeRanges object for time indexing.
            transfer_indices: TransferIndices object with computed mappings.
            remove_from_history: Historical energy removals (kWh).
            add_to_history: Historical energy additions (kWh).
        """
        s = self._solver

        for i in ranges.lookback_indices:
            if transfer_indices.move_to_indices[i]:
                s.add_constraint(
                    model,
                    remove_from_history[i]
                    == s.sum(
                        [
                            variables["transfer"][(i, j)]
                            for j in transfer_indices.move_to_indices[i]
                        ]
                    ),
                    name=f"Remove_from_historical_{i}",
                )

        for j in ranges.lookback_indices:
            if transfer_indices.get_from_indices[j]:
                s.add_constraint(
                    model,
                    add_to_history[j]
                    == s.sum(
                        [
                            variables["transfer"][(i, j)]
                            for i in transfer_indices.get_from_indices[j]
                        ]
                    ),
                    name=f"Add_to_historical_{j}",
                )

    def _add_control_constraints(
        self,
        model: Any,
        variables: Dict[str, dict],
        ranges: TimeRanges,
        transfer_indices: TransferIndices,
        demand: np.ndarray,
    ) -> None:
        """Add constraints for the optimization period.

        Implements all constraints from the class docstring for the current
        optimization period, including:
        - Row sum constraints: Σⱼ T[i,j] ≤ demand[i] (can't move more than available)
        - Column sum calculations: Σᵢ T[i,j] = energy added to time j
        - Purchase balance: purchase[t] = demand[t] + added[t] - removed[t]

        Args:
            model: The optimization model instance.
            variables: Dictionary of decision variables.
            ranges: TimeRanges object for time indexing.
            transfer_indices: TransferIndices object with computed mappings.
            demand: Array of energy demand per interval (kWh).
        """
        s = self._solver

        # Remove From Constraints:
        # remove from equals the row sum of the transfer matrix, smaller than demand.
        for t in ranges.local_time:
            i = ranges.time_to_global_index(t)
            s.add_constraint(
                model,
                variables["remove_from"][t]
                == s.sum(
                    [
                        variables["transfer"][(i, j)]
                        for j in transfer_indices.move_to_indices[i]
                    ]
                ),
                name=f"Remove_from_{t}",
            )
            s.add_constraint(
                model,
                variables["remove_from"][t] <= demand[t],
                name=f"Remove_must_be_smaller_than_demand_{t}",
            )

        # Add To Constraints: add to is equal to the column sum of the transfer matrix.
        for t in ranges.local_time:
            j = ranges.time_to_global_index(t)
            s.add_constraint(
                model,
                variables["add_to"][t]
                == s.sum(
                    [
                        variables["transfer"][(i, j)]
                        for i in transfer_indices.get_from_indices[j]
                    ]
                ),
                name=f"Add_to_{t}",
            )

        # Purchase Constraints: purchase equals demand plus added minus removed.
        for t in ranges.local_time:
            s.add_constraint(
                model,
                variables["purchase"][t]
                == (demand[t] + variables["add_to"][t] - variables["remove_from"][t]),
                name=f"Purchase_at_{t}",
            )

    def _add_charge_direction_constraints(
        self,
        model: Any,
        variables: Dict[str, dict],
        ranges: TimeRanges,
    ) -> None:
        """Add charge direction mutual exclusivity constraints.

        Enforces that energy can only be added OR removed at each time step,
        not both simultaneously. This constraint uses binary variables and
        increases problem complexity but ensures physically realistic behavior.

        Args:
            model: The optimization model instance.
            variables: Dictionary of decision variables.
            ranges: TimeRanges object for time indexing.
        """
        s = self._solver

        for t in ranges.local_time:
            s.add_constraint(
                model,
                variables["add_to"][t] <= self.big_m * variables["charge_direction"][t],
                name=f"Add_limit_{t}",
            )
            s.add_constraint(
                model,
                variables["remove_from"][t]
                <= (self.big_m * (1 - variables["charge_direction"][t])),
                name=f"Remove_limit_{t}",
            )

    # ============================================================================
    # HELPER METHODS
    # ============================================================================

    def _validate_optional_array(
        self,
        array: Optional[np.ndarray],
        expected_size: int,
        name: str,
        default_value: float = 0.0,
    ) -> np.ndarray:
        """Validate and initialize optional numpy array.

        Generic validator for all optional array inputs. Provides default array
        (filled with default_value) if None, otherwise validates type and size.

        Args:
            array: Optional numpy array to validate.
            expected_size: Expected array size (first dimension).
            name: Parameter name for error messages.
            default_value: Value to use when creating default array (default 0.0).

        Returns:
            Validated array, or default array if input was None.

        Raises:
            TypeError: If array is not None and not a numpy array.
            ValueError: If array size doesn't match expected_size.
        """
        if array is None:
            return np.full(expected_size, default_value, dtype=float)

        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(array).__name__}")

        if array.shape[0] != expected_size:
            raise ValueError(
                f"{name} has size {array.shape[0]}, expected {expected_size}"
            )

        return array

    def _init_time_params(
        self,
        price: np.ndarray,
        demand: np.ndarray,
        n_control_hours: Optional[int],
    ) -> tuple[int, int]:
        """Initialize and validate time parameters for optimization."""
        # Validate array lengths match
        if price.shape[0] != demand.shape[0]:
            raise ValueError(
                f"price length ({price.shape[0]}) must equal "
                f"demand length ({demand.shape[0]})"
            )

        # Calculate and validate control parameters
        n_lookahead_hours = demand.shape[0]
        if n_control_hours is None:
            n_control_hours = n_lookahead_hours

        if n_control_hours > n_lookahead_hours:
            raise ValueError(
                f"n_control_hours ({n_control_hours}) must be less than or equal to "
                f"n_lookahead_hours ({n_lookahead_hours})"
            )

        return n_lookahead_hours, n_control_hours

    def _compute_spillover(
        self, variables: dict, transfer_indices: TransferIndices
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute energy spillover to future periods beyond control horizon."""
        s = self._solver

        # Compute energy removed from current period that spills to future periods
        move_to_spillover_indices = transfer_indices.move_to_spillover_indices
        remove_from_spillover = np.array(
            [
                sum(
                    s.get_value(variables["transfer"][(i, j)]) or 0
                    for j in move_to_spillover_indices[i]
                )
                for i in sorted(move_to_spillover_indices.keys())
            ]
        )

        # Compute energy added to current period from past periods that spills over
        get_from_spillover_indices = transfer_indices.get_from_spillover_indices
        add_to_spillover = np.array(
            [
                sum(
                    s.get_value(variables["transfer"][(i, j)]) or 0
                    for i in get_from_spillover_indices[j]
                )
                for j in sorted(get_from_spillover_indices.keys())
            ]
        )

        return remove_from_spillover, add_to_spillover

    def _extract_debug_info(
        self,
        variables: dict,
        ranges: TimeRanges,
        transfer_indices: TransferIndices,
    ) -> dict:
        """Extract detailed optimization variables for debugging.

        Args:
            variables: Dictionary of optimization variables.
            ranges: TimeRanges object for time indexing.
            transfer_indices: TransferIndices object with computed mappings.

        Returns:
            Dictionary containing detailed debug information.
        """
        s = self._solver

        # Extract full lookahead results
        purchase = np.array(
            [s.get_value(variables["purchase"][t]) for t in ranges.local_time]
        )
        add_to = np.array(
            [s.get_value(variables["add_to"][t]) for t in ranges.local_time]
        )
        remove_from = np.array(
            [s.get_value(variables["remove_from"][t]) for t in ranges.local_time]
        )

        # Extract charge_direction only if it was created
        if self.enforce_charge_direction:
            charge_direction = np.array(
                [
                    s.get_value(variables["charge_direction"][t])
                    for t in ranges.local_time
                ]
            )
        else:
            charge_direction = None

        # Extract transfer matrix
        n_total_hours = ranges.n_lookback_hours + ranges.n_lookahead_hours
        transfer_matrix = np.zeros((n_total_hours, n_total_hours))
        for i in ranges.all_indices:
            for j in transfer_indices.move_to_indices[i]:
                transfer_matrix[i, j] = s.get_value(variables["transfer"][(i, j)])

        return {
            "purchase": purchase,
            "add_to": add_to,
            "remove_from": remove_from,
            "charge_direction": charge_direction,
            "transfer_matrix": transfer_matrix,
        }
