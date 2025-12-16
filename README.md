<p align="center">
    <img src="images/logo.png" alt="load-shift-optimizer logo" width="256px" >   
</p>

<h2 align="center"> Shift loads optimally to minimize electricity costs </h2>

<div align="center">


[![Ruff Lint](https://github.com/NoviaIntSysGroup/load-shift-optimizer/actions/workflows/lint.yml/badge.svg)](https://github.com/NoviaIntSysGroup/load-shift-optimizer/actions/workflows/lint.yml)
[![Unit Tests](https://github.com/NoviaIntSysGroup/load-shift-optimizer/actions/workflows/tests.yml/badge.svg)](https://github.com/NoviaIntSysGroup/load-shift-optimizer/actions/workflows/tests.yml)


![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

</div>

<div align="center">

`loadshift` helps you optimally shift loads based on known prices and a given flexibility level, where the flexibility level indicates how many hours earlier or later a load can run. The package can be used to determine optimal load shifts based on day-ahead electricity prices, or to evaluate potential savings from various load-shifting scenarios.

</div>

## Features

* ⚡ **Cost Optimization** - Shift loads optimally to minimize costs based on known electricity prices.  
* 🎛️ **Flexible Constraints** - Define how many hours loads can shift earlier or later, transfer rate limits, and power capacity to match your use case.  
* 📅 **Moving Horizon** - Daily optimization approach that replicates real-world day-ahead market scenarios.

## Example
The example below shows how electricity prices and residential loads change over a typical day: prices peak in the morning and evening, while residential loads peak in the evening when people get home. The two rightmost panels illustrate the results of shifting loads optimally for two flexibility levels: ±2 hours and ±4 hours. Observe how more and more consumption shifts towards nighttime and the afternoon as the flexibility level increases.


<p align="center">
<img src="examples/load_shift_example.png" style="max-width: 600px; width: 100%; height: auto;" /> 
</p>

## Installation

### From PyPI (recommended)

To use `load-shift-optimizer` in your project, install it from PyPI:

```bash
pip install loadshift
```

By default, this installs the free open-source MIP solver (CBC backend). For better performance on large problems, you can install with Gurobi support if you have a Gurobi license:

```bash
pip install loadshift[gurobi]
```

### From source (for examples and development)

If you want to explore the examples or contribute to the project, follow these steps to install from source:

```bash
# clone repository
$ git clone https://github.com/NoviaIntSysGroup/load-shift-optimizer.git
$ cd load-shift-optimizer

# install package and development dependencies (with MIP solver)
$ uv sync

# OR install with Gurobi support (requires a license)
$ uv sync --extra gurobi
```

## Usage

### Basic Optimization

```python
import numpy as np
from loadshift import LoadShifter

# Define your price and demand data
price = np.array([30, 80, 20, 40, 35, 25])  # ct/kWh
demand = np.array([10, 15, 8, 12, 10, 9])   # kWh

# Create optimizer with flexibility constraints
optimizer = LoadShifter(
    max_demand_advance=2,      # Can shift loads up to 2 hours earlier
    max_demand_delay=3,        # Can shift loads up to 3 hours later
    max_hourly_purchase=20,    # Maximum 20 kWh per hour
    max_rate=10                # Maximum 10 kW transfer rate
)

# Optimize demand
result = optimizer.optimize_demand(price, demand)

print("Optimal demand:", result["optimal_demand"])
print("Demand shift:", result["optimal_shift"])
```

### Moving Horizon Optimization

```python
import pandas as pd
from loadshift import moving_horizon

# Create DataFrames with a datetime index
index = pd.date_range("2024-01-01", periods=72, freq="h")
price_data = pd.DataFrame({"price": price_values}, index=index)
demand_data = pd.DataFrame({"demand": demand_values}, index=index)

# Configuration
config = {
    "daily_decision_hour": 12,     # Make decisions at noon each day
    "n_lookahead_hours": 36,       # Look ahead 36 hours
    "load_shift": {
        "max_demand_advance": 2,
        "max_demand_delay": 3,
        "max_hourly_purchase": 20,
        "max_rate": 10
    }
}

# Run moving horizon optimization
result = moving_horizon(price_data, demand_data, config)

# Access optimized demand and shifts
optimized = result["results"]
print(optimized.head())
```

## Development

Install development requirements and set up the hooks:

```bash
uv sync
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

Before committing or pushing run:

```bash
uv run ruff check .
uv run pytest
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the test suite (`uv run pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Please ensure your code follows our style guidelines:
- Use Ruff for code formatting and linting
- Follow Google's Python style guide for docstrings
- Include type annotations for all functions
- Add tests for new functionality

## Acknowledgements
This tool was developed within the "Demand response - Promoting electricity demand response management in
Ostrobothnia" project co-funded by the European Union through the "Just Transition Fund" under the "A Renewing and
Skilled Finland 2021–2027" programme.

## License

This project is released under the [MIT License](LICENSE).
