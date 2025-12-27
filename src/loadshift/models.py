"""Pydantic models for API request/response validation."""

from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class OptimizationRequest(BaseModel):
    """Request model for basic optimization."""

    price: List[float] = Field(
        ...,
        description="Array of energy prices per time interval (ct/kWh)",
        min_length=1,
    )
    demand: List[float] = Field(
        ..., description="Array of energy demand per time interval (kWh)", min_length=1
    )
    max_demand_advance: int = Field(
        ...,
        description="Maximum hours before demand energy can be purchased",
        ge=0,
    )
    max_demand_delay: int = Field(
        ..., description="Maximum hours after demand energy can be purchased", ge=0
    )
    max_hourly_purchase: float = Field(
        ..., description="Maximum energy that can be purchased per hour (kWh)", gt=0
    )
    max_rate: float = Field(
        ..., description="Maximum transfer rate (kW)", gt=0
    )
    enforce_charge_direction: bool = Field(
        default=False,
        description="If True, energy can only be added OR removed at each time step",
    )
    solver: str = Field(
        default="auto",
        description="Solver to use: 'auto', 'mip', or 'gurobi'",
    )
    n_control_hours: Optional[int] = Field(
        default=None,
        description="Number of hours in control period (optional)",
        ge=1,
    )

    @field_validator("price", "demand")
    @classmethod
    def validate_arrays_not_empty(cls, v: List[float]) -> List[float]:
        """Validate that arrays are not empty."""
        if len(v) == 0:
            raise ValueError("Array cannot be empty")
        return v

    @field_validator("demand")
    @classmethod
    def validate_demand_length(cls, v: List[float], info: Any) -> List[float]:
        """Validate that demand has the same length as price."""
        if "price" in info.data and len(v) != len(info.data["price"]):
            raise ValueError(
                f"Demand length ({len(v)}) must match price length "
                f"({len(info.data['price'])})"
            )
        return v


class OptimizationResponse(BaseModel):
    """Response model for basic optimization."""

    optimal_demand: List[float] = Field(
        ..., description="Optimized demand profile (kWh)"
    )
    optimal_shift: List[float] = Field(
        ..., description="Demand shift at each time step (kWh)"
    )
    original_cost: float = Field(
        ..., description="Original cost without optimization (ct)"
    )
    optimized_cost: float = Field(
        ..., description="Optimized cost after load shifting (ct)"
    )
    cost_savings: float = Field(
        ..., description="Cost savings from optimization (ct)"
    )
    cost_savings_percent: float = Field(
        ..., description="Cost savings as percentage (%)"
    )
    remove_spillover: Optional[List[float]] = Field(
        default=None,
        description="Energy removed from future periods (if n_control_hours specified)",
    )
    add_spillover: Optional[List[float]] = Field(
        default=None,
        description="Energy added to future periods (if n_control_hours specified)",
    )


class TimeSeriesData(BaseModel):
    """Time series data with timestamps and values."""

    timestamps: List[str] = Field(
        ...,
        description="ISO 8601 formatted timestamps (e.g., '2024-01-01T00:00:00')",
        min_length=1,
    )
    values: List[float] = Field(
        ..., description="Values corresponding to each timestamp", min_length=1
    )

    @field_validator("values")
    @classmethod
    def validate_values_length(cls, v: List[float], info: Any) -> List[float]:
        """Validate that values has the same length as timestamps."""
        if "timestamps" in info.data and len(v) != len(info.data["timestamps"]):
            raise ValueError(
                f"Values length ({len(v)}) must match timestamps length "
                f"({len(info.data['timestamps'])})"
            )
        return v


class LoadShiftConfig(BaseModel):
    """Load shift configuration for moving horizon optimization."""

    max_demand_advance: int = Field(
        ..., description="Maximum hours before demand energy can be purchased", ge=0
    )
    max_demand_delay: int = Field(
        ..., description="Maximum hours after demand energy can be purchased", ge=0
    )
    max_hourly_purchase: float = Field(
        ..., description="Maximum energy that can be purchased per hour (kWh)", gt=0
    )
    max_rate: float = Field(..., description="Maximum transfer rate (kW)", gt=0)
    enforce_charge_direction: bool = Field(
        default=False,
        description="If True, energy can only be added OR removed at each time step",
    )
    solver: str = Field(
        default="auto", description="Solver to use: 'auto', 'mip', or 'gurobi'"
    )


class MovingHorizonRequest(BaseModel):
    """Request model for moving horizon optimization."""

    price_data: TimeSeriesData = Field(
        ..., description="Price time series data (ct/kWh)"
    )
    demand_data: TimeSeriesData = Field(
        ..., description="Demand time series data (kWh)"
    )
    daily_decision_hour: int = Field(
        ..., description="Hour of day to make optimization decisions (0-23)", ge=0, le=23
    )
    n_lookahead_hours: int = Field(
        ..., description="Number of hours to look ahead in optimization", ge=24
    )
    load_shift: LoadShiftConfig = Field(
        ..., description="Load shift configuration parameters"
    )


class MovingHorizonResponse(BaseModel):
    """Response model for moving horizon optimization."""

    timestamps: List[str] = Field(..., description="Timestamps for the results")
    original_demand: List[float] = Field(..., description="Original demand profile")
    optimal_demand: List[float] = Field(..., description="Optimized demand profile")
    shift: List[float] = Field(..., description="Demand shift at each time step")
    price: List[float] = Field(..., description="Price at each time step")
    original_cost: float = Field(..., description="Original total cost")
    optimized_cost: float = Field(..., description="Optimized total cost")
    cost_savings: float = Field(..., description="Total cost savings")
    cost_savings_percent: float = Field(..., description="Cost savings as percentage")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    detail: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(default=None, description="Type of error")
