"""Solver adapters for virtual storage optimization.

Provides a unified interface for different MILP solvers (Gurobi, python-mip/CBC).
"""

from typing import Any

from .utils import get_logger

logger = get_logger(__name__)


class SolverAdapter:
    """Base adapter interface for MILP solvers."""

    def create_model(self) -> Any:
        """Create and return a new optimization model."""
        raise NotImplementedError

    def add_var(
        self,
        model: Any,
        name: str,
        lb: float = 0,
        ub: float | None = None,
        vtype: str = "continuous",
    ) -> Any:
        """Add a variable to the model.

        Args:
            model: The optimization model.
            name: Variable name.
            lb: Lower bound (default 0).
            ub: Upper bound (None for unbounded).
            vtype: Variable type ('continuous' or 'binary').

        Returns:
            Variable object.
        """
        raise NotImplementedError

    def add_constraint(
        self, model: Any, constraint: Any, name: str | None = None
    ) -> None:
        """Add a constraint to the model.

        Args:
            model: The optimization model.
            constraint: The constraint expression.
            name: Optional constraint name.
        """
        raise NotImplementedError

    def sum(self, variables: list[Any]) -> Any:
        """Create a sum expression from variables.

        Args:
            variables: List of variables to sum.

        Returns:
            Sum expression.
        """
        raise NotImplementedError

    def set_objective(
        self, model: Any, expression: Any, sense: str = "minimize"
    ) -> None:
        """Set the optimization objective.

        Args:
            model: The optimization model.
            expression: The objective expression.
            sense: 'minimize' or 'maximize'.
        """
        raise NotImplementedError

    def solve(self, model: Any) -> None:
        """Solve the optimization problem.

        Args:
            model: The optimization model.
        """
        raise NotImplementedError

    def get_value(self, var: Any) -> float:
        """Get the solution value of a variable.

        Args:
            var: The variable.

        Returns:
            Solution value.
        """
        raise NotImplementedError


class GurobiAdapter(SolverAdapter):
    """Adapter for Gurobi solver."""

    def create_model(self) -> Any:
        """Create a Gurobi model."""
        import gurobipy as gp

        model = gp.Model("Virtual_Storage_Optimization")
        model.setParam("OutputFlag", 0)  # Silent mode
        return model

    def add_var(
        self,
        model: Any,
        name: str,
        lb: float = 0,
        ub: float | None = None,
        vtype: str = "continuous",
    ) -> Any:
        """Add a variable to the Gurobi model."""
        from gurobipy import GRB

        vtype_map = {"continuous": GRB.CONTINUOUS, "binary": GRB.BINARY}
        kwargs = {"lb": lb, "vtype": vtype_map[vtype], "name": name}
        if ub is not None:
            kwargs["ub"] = ub
        return model.addVar(**kwargs)

    def add_constraint(
        self, model: Any, constraint: Any, name: str | None = None
    ) -> None:
        """Add a constraint to the Gurobi model."""
        model.addConstr(constraint, name=name)

    def sum(self, variables: list[Any]) -> Any:
        """Create a sum expression using Gurobi's quicksum."""
        import gurobipy as gp

        return gp.quicksum(variables)

    def set_objective(
        self, model: Any, expression: Any, sense: str = "minimize"
    ) -> None:
        """Set the objective function for Gurobi model."""
        from gurobipy import GRB

        sense_map = {"minimize": GRB.MINIMIZE, "maximize": GRB.MAXIMIZE}
        model.setObjective(expression, sense_map[sense])

    def solve(self, model: Any) -> None:
        """Solve the Gurobi optimization problem."""
        model.optimize()
        self._check_status(model.Status)

    def get_value(self, var: Any) -> float:
        """Get the solution value from a Gurobi variable."""
        return var.X

    def _check_status(self, status: int) -> None:
        """Check and log Gurobi solver status."""
        from gurobipy import GRB

        if status == GRB.OPTIMAL:
            logger.debug("Solver status: Optimal")
        elif status == GRB.INFEASIBLE:
            logger.warning(
                "Optimization problem is infeasible. Check constraints "
                "compatibility: storage_size, max_rate, transfer windows, "
                "and demand levels."
            )
        elif status == GRB.UNBOUNDED:
            logger.warning(
                "Optimization problem is unbounded. Check objective function."
            )
        elif status == GRB.INF_OR_UNBD:
            logger.warning("Optimization problem is infeasible or unbounded.")
        else:
            logger.warning("Solver returned non-optimal status: %s", status)


class MipAdapter(SolverAdapter):
    """Adapter for python-mip solver (CBC backend)."""

    def create_model(self) -> Any:
        """Create a python-mip model."""
        from mip import MINIMIZE, Model

        return Model(sense=MINIMIZE, solver_name="CBC")

    def add_var(
        self,
        model: Any,
        name: str,
        lb: float = 0,
        ub: float | None = None,
        vtype: str = "continuous",
    ) -> Any:
        """Add a variable to the python-mip model."""
        from mip import BINARY, CONTINUOUS

        vtype_map = {"continuous": CONTINUOUS, "binary": BINARY}
        kwargs = {"name": name, "lb": lb, "var_type": vtype_map[vtype]}
        if ub is not None:
            kwargs["ub"] = ub
        return model.add_var(**kwargs)

    def add_constraint(
        self, model: Any, constraint: Any, name: str | None = None
    ) -> None:
        """Add a constraint to the python-mip model."""
        model += constraint

    def sum(self, variables: list[Any]) -> Any:
        """Create a sum expression using Python's built-in sum."""
        return sum(variables)

    def set_objective(
        self, model: Any, expression: Any, sense: str = "minimize"
    ) -> None:
        """Set the objective function for python-mip model."""
        from mip import MAXIMIZE, MINIMIZE

        sense_map = {"minimize": MINIMIZE, "maximize": MAXIMIZE}
        model.objective = expression
        model.sense = sense_map[sense]

    def solve(self, model: Any) -> None:
        """Solve the python-mip optimization problem."""
        status = model.optimize()
        self._check_status(status)

    def get_value(self, var: Any) -> float:
        """Get the solution value from a python-mip variable."""
        return var.x

    def _check_status(self, status: Any) -> None:
        """Check and log python-mip solver status."""
        from mip import OptimizationStatus

        logger.debug("Solver status: %s", status)

        if status != OptimizationStatus.OPTIMAL:
            if status == OptimizationStatus.INFEASIBLE:
                logger.warning(
                    "Optimization problem is infeasible. Check constraints "
                    "compatibility: storage_size, max_rate, transfer windows, "
                    "and demand levels."
                )
            elif status == OptimizationStatus.UNBOUNDED:
                logger.warning(
                    "Optimization problem is unbounded. Check objective function."
                )
            elif status == OptimizationStatus.ERROR:
                logger.warning(
                    "Solver failed to find a solution. Try different solver parameters."
                )
            else:
                logger.warning("Solver returned non-optimal status: %s", status)


def create_solver_adapter(solver: str) -> SolverAdapter:
    """Factory function to create the appropriate solver adapter.

    Args:
        solver: Solver name ('auto', 'gurobi', or 'mip').

    Returns:
        SolverAdapter instance.

    Raises:
        ValueError: If solver name is invalid.
        ImportError: If requested solver is not available.
    """
    if solver == "auto":
        try:
            import gurobipy

            logger.debug("Using Gurobi solver backend")
            return GurobiAdapter()
        except ImportError:
            logger.debug("Gurobi not available, falling back to MIP")
            return MipAdapter()
    elif solver == "gurobi":
        import gurobipy  # noqa: F401

        logger.debug("Using Gurobi solver backend")
        return GurobiAdapter()
    elif solver == "mip":
        from mip import Model  # noqa: F401

        logger.debug("Using MIP solver backend")
        return MipAdapter()
    else:
        raise ValueError(f"Unknown solver: {solver}. Use 'auto', 'gurobi', or 'mip'")
