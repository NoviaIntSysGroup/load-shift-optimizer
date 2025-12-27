import numpy as np
import pandas as pd
from loadshift.virtual_storage import VirtualStorage
from loadshift.models import OptimizationRequest

def test_repro():
    try:
        price = np.array([30, 80, 20, 40, 35, 25])
        demand = np.array([10, 15, 8, 12, 10, 9])
        
        optimizer = VirtualStorage(
            max_demand_advance=2,
            max_demand_delay=3,
            max_hourly_purchase=20.0,
            max_rate=10.0,
            enforce_charge_direction=False,
            solver="mip",
        )
        
        print("Starting optimization...")
        result = optimizer.optimize_demand(
            price=price, demand=demand, n_control_hours=None
        )
        print("Optimization successful!")
        print(result["optimal_demand"])
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_repro()
