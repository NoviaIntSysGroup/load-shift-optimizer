"""Pytest configuration for loadshift tests."""

from typing import Any


def pytest_generate_tests(metafunc: Any) -> None:
    """Automatically parametrize all tests with both solvers.

    This hook runs for every test function and adds solver parametrization
    if the test accepts a 'solver' parameter.

    Tests will run with both "mip" and "gurobi" solvers. If Gurobi is not
    available, only "mip" tests will run.
    """
    if "solver" in metafunc.fixturenames:
        # Check if Gurobi is available
        try:
            import gurobipy  # noqa: F401

            solvers = ["mip", "gurobi"]
        except ImportError:
            solvers = ["mip"]

        metafunc.parametrize("solver", solvers)
