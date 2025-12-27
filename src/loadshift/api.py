"""FastAPI application for load-shift-optimizer REST API."""

from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import (
    ErrorResponse,
    HealthResponse,
    MovingHorizonRequest,
    MovingHorizonResponse,
    OptimizationRequest,
    OptimizationResponse,
)
from .moving_horizon import moving_horizon
from .virtual_storage import VirtualStorage

# API version
API_VERSION = "1.0.0"

# Create FastAPI app
app = FastAPI(
    title="Load Shift Optimizer API",
    description="""
    REST API for optimizing electrical load shifting based on price signals.
    
    This API provides endpoints to:
    - Optimize energy demand based on known prices and flexibility constraints
    - Run moving horizon optimization for time-series data
    - Minimize electricity costs through intelligent load shifting
    
    ## Features
    - **Basic Optimization**: Shift loads optimally for a given price and demand profile
    - **Moving Horizon**: Day-by-day optimization mimicking real-world scenarios
    - **Flexible Constraints**: Configure advance/delay hours, power limits, and transfer rates
    
    ## Getting Started
    1. Use the `/api/v1/optimize` endpoint for basic optimization
    2. Use the `/api/v1/optimize/moving-horizon` endpoint for time-series optimization
    3. Explore the interactive documentation below to test endpoints
    """,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware for browser-based testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            detail=f"Internal server error: {str(exc)}", error_type=type(exc).__name__
        ).model_dump(),
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns the service status and API version.
    """
    return HealthResponse(status="healthy", version=API_VERSION)


@app.post(
    "/api/v1/optimize",
    response_model=OptimizationResponse,
    tags=["Optimization"],
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Optimization failed"},
    },
)
async def optimize_demand(request: OptimizationRequest) -> OptimizationResponse:
    """
    Optimize energy demand based on price signals and flexibility constraints.
    
    This endpoint performs basic load shifting optimization to minimize electricity costs.
    It finds the optimal transfer matrix T[i,j] that shifts energy consumption from
    expensive to cheaper time periods while respecting all constraints.
    
    ## Parameters
    - **price**: Energy prices per time interval (ct/kWh)
    - **demand**: Energy demand per time interval (kWh)
    - **max_demand_advance**: Hours before demand can be purchased (flexibility)
    - **max_demand_delay**: Hours after demand can be purchased (flexibility)
    - **max_hourly_purchase**: Maximum energy per hour (kWh)
    - **max_rate**: Maximum transfer rate (kW)
    - **enforce_charge_direction**: Only add OR remove energy at each time step
    - **solver**: Solver to use ('auto', 'mip', or 'gurobi')
    - **n_control_hours**: Control period length (optional, for moving horizon)
    
    ## Returns
    - **optimal_demand**: Optimized demand profile
    - **optimal_shift**: Demand shift at each time step
    - **original_cost**: Cost without optimization
    - **optimized_cost**: Cost after optimization
    - **cost_savings**: Absolute cost savings
    - **cost_savings_percent**: Percentage cost savings
    
    ## Example
    ```json
    {
        "price": [30, 80, 20, 40, 35, 25],
        "demand": [10, 15, 8, 12, 10, 9],
        "max_demand_advance": 2,
        "max_demand_delay": 3,
        "max_hourly_purchase": 20,
        "max_rate": 10
    }
    ```
    """
    try:
        # Convert lists to numpy arrays
        price = np.array(request.price)
        demand = np.array(request.demand)

        # Create optimizer
        optimizer = VirtualStorage(
            max_demand_advance=request.max_demand_advance,
            max_demand_delay=request.max_demand_delay,
            max_hourly_purchase=request.max_hourly_purchase,
            max_rate=request.max_rate,
            enforce_charge_direction=request.enforce_charge_direction,
            solver=request.solver,
        )

        # Run optimization
        result = optimizer.optimize_demand(
            price=price, demand=demand, n_control_hours=request.n_control_hours
        )

        # Calculate costs
        original_cost = float(np.sum(price * demand))
        optimized_cost = float(np.sum(price * result["optimal_demand"]))
        cost_savings = original_cost - optimized_cost
        cost_savings_percent = (
            (cost_savings / original_cost * 100) if original_cost > 0 else 0.0
        )

        # Build response
        response = OptimizationResponse(
            optimal_demand=result["optimal_demand"].tolist(),
            optimal_shift=result["optimal_shift"].tolist(),
            original_cost=original_cost,
            optimized_cost=optimized_cost,
            cost_savings=cost_savings,
            cost_savings_percent=cost_savings_percent,
        )

        # Add spillover if present
        if "remove_spillover" in result:
            response.remove_spillover = result["remove_spillover"].tolist()
        if "add_spillover" in result:
            response.add_spillover = result["add_spillover"].tolist()

        return response

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request parameters: {str(e)}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}",
        ) from e


@app.post(
    "/api/v1/optimize/moving-horizon",
    response_model=MovingHorizonResponse,
    tags=["Optimization"],
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Optimization failed"},
    },
)
async def optimize_moving_horizon(
    request: MovingHorizonRequest,
) -> MovingHorizonResponse:
    """
    Run moving horizon optimization for time-series data.
    
    This endpoint performs day-by-day optimization that mimics real-world scenarios
    where price information becomes available incrementally (e.g., day-ahead markets).
    
    ## Parameters
    - **price_data**: Time series of prices with timestamps
    - **demand_data**: Time series of demand with timestamps
    - **daily_decision_hour**: Hour of day to make decisions (0-23)
    - **n_lookahead_hours**: Hours to look ahead (must be >= 24)
    - **load_shift**: Configuration for load shifting constraints
    
    ## Returns
    - **timestamps**: Timestamps for the optimization results
    - **original_demand**: Original demand profile
    - **optimal_demand**: Optimized demand profile
    - **shift**: Demand shift at each time step
    - **price**: Price at each time step
    - **original_cost**: Total cost without optimization
    - **optimized_cost**: Total cost after optimization
    - **cost_savings**: Total cost savings
    - **cost_savings_percent**: Percentage cost savings
    
    ## Example
    ```json
    {
        "price_data": {
            "timestamps": ["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
            "values": [30, 35]
        },
        "demand_data": {
            "timestamps": ["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
            "values": [10, 12]
        },
        "daily_decision_hour": 12,
        "n_lookahead_hours": 36,
        "load_shift": {
            "max_demand_advance": 2,
            "max_demand_delay": 3,
            "max_hourly_purchase": 20,
            "max_rate": 10
        }
    }
    ```
    """
    try:
        # Convert time series data to pandas DataFrames
        price_index = pd.to_datetime(request.price_data.timestamps)
        demand_index = pd.to_datetime(request.demand_data.timestamps)

        price_df = pd.DataFrame(
            {"price": request.price_data.values}, index=price_index
        )
        demand_df = pd.DataFrame(
            {"demand": request.demand_data.values}, index=demand_index
        )

        # Build configuration
        config = {
            "daily_decision_hour": request.daily_decision_hour,
            "n_lookahead_hours": request.n_lookahead_hours,
            "load_shift": {
                "max_demand_advance": request.load_shift.max_demand_advance,
                "max_demand_delay": request.load_shift.max_demand_delay,
                "max_hourly_purchase": request.load_shift.max_hourly_purchase,
                "max_rate": request.load_shift.max_rate,
                "enforce_charge_direction": request.load_shift.enforce_charge_direction,
                "solver": request.load_shift.solver,
            },
        }

        # Run moving horizon optimization
        result = moving_horizon(price_df, demand_df, config)
        results_df = result["results"]

        # Calculate costs
        original_cost = float((price_df["price"] * demand_df["demand"]).sum())
        optimized_cost = float((price_df["price"] * results_df["demand"]).sum())
        cost_savings = original_cost - optimized_cost
        cost_savings_percent = (
            (cost_savings / original_cost * 100) if original_cost > 0 else 0.0
        )

        # Build response
        return MovingHorizonResponse(
            timestamps=[ts.isoformat() for ts in results_df.index],
            original_demand=demand_df["demand"].tolist(),
            optimal_demand=results_df["demand"].tolist(),
            shift=results_df["shift"].tolist(),
            price=price_df["price"].tolist(),
            original_cost=original_cost,
            optimized_cost=optimized_cost,
            cost_savings=cost_savings,
            cost_savings_percent=cost_savings_percent,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request parameters: {str(e)}",
        ) from e
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required configuration key: {str(e)}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}",
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
