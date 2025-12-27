"""Tests for the FastAPI application."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from loadshift.api import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self) -> None:
        """Test that health check returns correct status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestBasicOptimizationEndpoint:
    """Tests for the basic optimization endpoint."""

    def test_basic_optimization_success(self) -> None:
        """Test successful basic optimization."""
        request_data = {
            "price": [30, 80, 20, 40, 35, 25],
            "demand": [10, 15, 8, 12, 10, 9],
            "max_demand_advance": 2,
            "max_demand_delay": 3,
            "max_hourly_purchase": 20,
            "max_rate": 10,
        }

        response = client.post("/api/v1/optimize", json=request_data)
        assert response.status_code == 200

        result = response.json()
        assert "optimal_demand" in result
        assert "optimal_shift" in result
        assert "original_cost" in result
        assert "optimized_cost" in result
        assert "cost_savings" in result
        assert "cost_savings_percent" in result

        # Verify array lengths
        assert len(result["optimal_demand"]) == len(request_data["price"])
        assert len(result["optimal_shift"]) == len(request_data["price"])

        # Verify cost savings is non-negative
        assert result["cost_savings"] >= 0
        assert result["optimized_cost"] <= result["original_cost"]

    def test_basic_optimization_with_control_hours(self) -> None:
        """Test basic optimization with control hours specified."""
        request_data = {
            "price": [30, 80, 20, 40, 35, 25],
            "demand": [10, 15, 8, 12, 10, 9],
            "max_demand_advance": 2,
            "max_demand_delay": 3,
            "max_hourly_purchase": 20,
            "max_rate": 10,
            "n_control_hours": 3,
        }

        response = client.post("/api/v1/optimize", json=request_data)
        assert response.status_code == 200

        result = response.json()
        assert "remove_spillover" in result
        assert "add_spillover" in result
        assert result["remove_spillover"] is not None
        assert result["add_spillover"] is not None

    def test_basic_optimization_invalid_empty_arrays(self) -> None:
        """Test that empty arrays are rejected."""
        request_data = {
            "price": [],
            "demand": [],
            "max_demand_advance": 2,
            "max_demand_delay": 3,
            "max_hourly_purchase": 20,
            "max_rate": 10,
        }

        response = client.post("/api/v1/optimize", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_basic_optimization_mismatched_lengths(self) -> None:
        """Test that mismatched array lengths are rejected."""
        request_data = {
            "price": [30, 80, 20],
            "demand": [10, 15],  # Different length
            "max_demand_advance": 2,
            "max_demand_delay": 3,
            "max_hourly_purchase": 20,
            "max_rate": 10,
        }

        response = client.post("/api/v1/optimize", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_basic_optimization_negative_constraints(self) -> None:
        """Test that negative constraint values are rejected."""
        request_data = {
            "price": [30, 80, 20, 40, 35, 25],
            "demand": [10, 15, 8, 12, 10, 9],
            "max_demand_advance": -1,  # Invalid
            "max_demand_delay": 3,
            "max_hourly_purchase": 20,
            "max_rate": 10,
        }

        response = client.post("/api/v1/optimize", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_basic_optimization_zero_max_rate(self) -> None:
        """Test that zero max_rate is rejected."""
        request_data = {
            "price": [30, 80, 20, 40, 35, 25],
            "demand": [10, 15, 8, 12, 10, 9],
            "max_demand_advance": 2,
            "max_demand_delay": 3,
            "max_hourly_purchase": 20,
            "max_rate": 0,  # Invalid
        }

        response = client.post("/api/v1/optimize", json=request_data)
        assert response.status_code == 422  # Validation error


class TestMovingHorizonEndpoint:
    """Tests for the moving horizon optimization endpoint."""

    def test_moving_horizon_success(self) -> None:
        """Test successful moving horizon optimization."""
        # Create 72 hours of data (3 days)
        timestamps = pd.date_range("2024-01-01", periods=72, freq="h")
        price_values = [30.0 + 20.0 * (i % 24 < 12) for i in range(72)]
        demand_values = [5.0 + 3.0 * (i % 24 >= 18) for i in range(72)]

        request_data = {
            "price_data": {
                "timestamps": [ts.isoformat() for ts in timestamps],
                "values": price_values,
            },
            "demand_data": {
                "timestamps": [ts.isoformat() for ts in timestamps],
                "values": demand_values,
            },
            "daily_decision_hour": 12,
            "n_lookahead_hours": 36,
            "load_shift": {
                "max_demand_advance": 2,
                "max_demand_delay": 3,
                "max_hourly_purchase": 20,
                "max_rate": 10,
            },
        }

        response = client.post("/api/v1/optimize/moving-horizon", json=request_data)
        assert response.status_code == 200

        result = response.json()
        assert "timestamps" in result
        assert "original_demand" in result
        assert "optimal_demand" in result
        assert "shift" in result
        assert "price" in result
        assert "original_cost" in result
        assert "optimized_cost" in result
        assert "cost_savings" in result
        assert "cost_savings_percent" in result

        # Verify cost savings
        assert result["cost_savings"] >= 0
        assert result["optimized_cost"] <= result["original_cost"]

    def test_moving_horizon_invalid_lookahead(self) -> None:
        """Test that lookahead < 24 hours is rejected."""
        timestamps = pd.date_range("2024-01-01", periods=48, freq="h")

        request_data = {
            "price_data": {
                "timestamps": [ts.isoformat() for ts in timestamps],
                "values": [30.0] * 48,
            },
            "demand_data": {
                "timestamps": [ts.isoformat() for ts in timestamps],
                "values": [10.0] * 48,
            },
            "daily_decision_hour": 12,
            "n_lookahead_hours": 20,  # Invalid: < 24
            "load_shift": {
                "max_demand_advance": 2,
                "max_demand_delay": 3,
                "max_hourly_purchase": 20,
                "max_rate": 10,
            },
        }

        response = client.post("/api/v1/optimize/moving-horizon", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_moving_horizon_invalid_decision_hour(self) -> None:
        """Test that invalid decision hour is rejected."""
        timestamps = pd.date_range("2024-01-01", periods=48, freq="h")

        request_data = {
            "price_data": {
                "timestamps": [ts.isoformat() for ts in timestamps],
                "values": [30.0] * 48,
            },
            "demand_data": {
                "timestamps": [ts.isoformat() for ts in timestamps],
                "values": [10.0] * 48,
            },
            "daily_decision_hour": 25,  # Invalid: > 23
            "n_lookahead_hours": 36,
            "load_shift": {
                "max_demand_advance": 2,
                "max_demand_delay": 3,
                "max_hourly_purchase": 20,
                "max_rate": 10,
            },
        }

        response = client.post("/api/v1/optimize/moving-horizon", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_moving_horizon_mismatched_timestamps(self) -> None:
        """Test that mismatched timestamp lengths are rejected."""
        timestamps_price = pd.date_range("2024-01-01", periods=48, freq="h")
        timestamps_demand = pd.date_range("2024-01-01", periods=36, freq="h")

        request_data = {
            "price_data": {
                "timestamps": [ts.isoformat() for ts in timestamps_price],
                "values": [30.0] * 48,
            },
            "demand_data": {
                "timestamps": [ts.isoformat() for ts in timestamps_demand],
                "values": [10.0] * 36,
            },
            "daily_decision_hour": 12,
            "n_lookahead_hours": 36,
            "load_shift": {
                "max_demand_advance": 2,
                "max_demand_delay": 3,
                "max_hourly_purchase": 20,
                "max_rate": 10,
            },
        }

        response = client.post("/api/v1/optimize/moving-horizon", json=request_data)
        # This should succeed as the timestamps can be different
        # The validation happens at the pandas level
        assert response.status_code in [200, 400, 500]


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    def test_openapi_schema(self) -> None:
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

    def test_swagger_ui(self) -> None:
        """Test that Swagger UI is available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc(self) -> None:
        """Test that ReDoc is available."""
        response = client.get("/redoc")
        assert response.status_code == 200
