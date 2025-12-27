"""Example script demonstrating how to interact with the Load Shift Optimizer API.

This script shows how to use the API endpoints with Python's requests library.
Make sure the API server is running before executing this script:
    python examples/run_api.py
"""

import requests

# API base URL
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint."""
    print("\n" + "=" * 70)
    print("Testing Health Check Endpoint")
    print("=" * 70)

    response = requests.get(f"{BASE_URL}/health", timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    return response.status_code == 200


def test_basic_optimization():
    """Test the basic optimization endpoint."""
    print("\n" + "=" * 70)
    print("Testing Basic Optimization Endpoint")
    print("=" * 70)

    # Example request data
    request_data = {
        "price": [30, 80, 20, 40, 35, 25],  # ct/kWh
        "demand": [10, 15, 8, 12, 10, 9],  # kWh
        "max_demand_advance": 2,  # Can shift 2 hours earlier
        "max_demand_delay": 3,  # Can shift 3 hours later
        "max_hourly_purchase": 20,  # Max 20 kWh per hour
        "max_rate": 10,  # Max 10 kW transfer rate
    }

    print("\nRequest Data:")
    print(f"  Price:               {request_data['price']}")
    print(f"  Demand:              {request_data['demand']}")
    print(f"  Max Advance:         {request_data['max_demand_advance']} hours")
    print(f"  Max Delay:           {request_data['max_demand_delay']} hours")
    print(f"  Max Hourly Purchase: {request_data['max_hourly_purchase']} kWh")
    print(f"  Max Rate:            {request_data['max_rate']} kW")

    response = requests.post(
        f"{BASE_URL}/api/v1/optimize", json=request_data, timeout=30
    )

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print("\nOptimization Results:")
        print(f"  Original Demand:     {result['optimal_demand']}")
        print(f"  Optimal Shift:       {result['optimal_shift']}")
        print(f"  Original Cost:       {result['original_cost']:.2f} ct")
        print(f"  Optimized Cost:      {result['optimized_cost']:.2f} ct")
        print(f"  Cost Savings:        {result['cost_savings']:.2f} ct")
        print(f"  Savings Percentage:  {result['cost_savings_percent']:.2f}%")
        return True
    else:
        print(f"Error: {response.json()}")
        return False


def test_moving_horizon_optimization():
    """Test the moving horizon optimization endpoint."""
    print("\n" + "=" * 70)
    print("Testing Moving Horizon Optimization Endpoint")
    print("=" * 70)

    # Example request data with 3 days of hourly data
    import pandas as pd

    # Create sample data
    timestamps = pd.date_range("2024-01-01", periods=72, freq="h")
    price_values = [30 + 20 * (i % 24 < 12) for i in range(72)]  # Day/night pattern
    demand_values = [5 + 3 * (i % 24 >= 18) for i in range(72)]  # Evening peak

    request_data = {
        "price_data": {
            "timestamps": [ts.isoformat() for ts in timestamps],
            "values": price_values,
        },
        "demand_data": {
            "timestamps": [ts.isoformat() for ts in timestamps],
            "values": demand_values,
        },
        "daily_decision_hour": 12,  # Make decisions at noon
        "n_lookahead_hours": 36,  # Look ahead 36 hours
        "load_shift": {
            "max_demand_advance": 2,
            "max_demand_delay": 3,
            "max_hourly_purchase": 20,
            "max_rate": 10,
        },
    }

    print("\nRequest Data:")
    print(f"  Time Period:         {timestamps[0]} to {timestamps[-1]}")
    print(f"  Total Hours:         {len(timestamps)}")
    print(f"  Decision Hour:       {request_data['daily_decision_hour']}:00")
    print(f"  Lookahead Hours:     {request_data['n_lookahead_hours']}")

    response = requests.post(
        f"{BASE_URL}/api/v1/optimize/moving-horizon", json=request_data, timeout=60
    )

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print("\nOptimization Results:")
        print(f"  Time Points:         {len(result['timestamps'])}")
        print(f"  Original Cost:       {result['original_cost']:.2f} ct")
        print(f"  Optimized Cost:      {result['optimized_cost']:.2f} ct")
        print(f"  Cost Savings:        {result['cost_savings']:.2f} ct")
        print(f"  Savings Percentage:  {result['cost_savings_percent']:.2f}%")
        print(f"\n  First 5 timestamps:")
        for i in range(min(5, len(result["timestamps"]))):
            print(f"    {result['timestamps'][i]}")
        return True
    else:
        print(f"Error: {response.json()}")
        return False


def main():
    """Run all API tests."""
    print("\n" + "=" * 70)
    print("Load Shift Optimizer API - Example Client")
    print("=" * 70)
    print("\nMake sure the API server is running:")
    print("  python examples/run_api.py")
    print("\nOr start it manually:")
    print("  uv run uvicorn loadshift.api:app --reload")

    try:
        # Test health check
        if not test_health_check():
            print("\n❌ Health check failed. Is the server running?")
            return

        # Test basic optimization
        if test_basic_optimization():
            print("\n✅ Basic optimization test passed!")
        else:
            print("\n❌ Basic optimization test failed!")

        # Test moving horizon optimization
        if test_moving_horizon_optimization():
            print("\n✅ Moving horizon optimization test passed!")
        else:
            print("\n❌ Moving horizon optimization test failed!")

        print("\n" + "=" * 70)
        print("All tests completed!")
        print("=" * 70)
        print("\nNext steps:")
        print("  • Visit http://localhost:8000/docs for interactive API documentation")
        print("  • Try modifying the request parameters in this script")
        print("  • Deploy the API to a production environment")

    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to the API server.")
        print("Please start the server first:")
        print("  python examples/run_api.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
