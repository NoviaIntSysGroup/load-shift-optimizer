import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm

from .utils import get_logger
from .virtual_storage import VirtualStorage

logger = get_logger(__name__)


def _compute_control_hours(
    horizons: List[Tuple[pd.Timestamp, pd.Timestamp]], horizon_idx: int
) -> int:
    """Compute the number of control hours for a given horizon, accounting for DST.

    Args:
        horizons: List of (horizon_start, horizon_end) tuples
        horizon_idx: Index of the current horizon

    Returns:
        Number of hours to control (until next horizon start or current horizon end)
    """
    horizon_start, horizon_end = horizons[horizon_idx]

    if horizon_idx < len(horizons) - 1:
        # Time until next horizon starts (accounting for DST transitions)
        next_horizon_start = horizons[horizon_idx + 1][0]
        return len(
            pd.date_range(
                start=horizon_start, end=next_horizon_start, freq="h", inclusive="left"
            )
        )
    else:
        # Last horizon: control until the end of current horizon
        return len(
            pd.date_range(
                start=horizon_start, end=horizon_end, freq="h", inclusive="both"
            )
        )


def _create_horizons(
    datetime_index: pd.DatetimeIndex, daily_decision_hour: int, horizon_length: int
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Create optimization horizon start and end times.

    Each horizon extends from a decision time to
    (next_decision + horizon_length - 24 - 1) hours.
    The -1 accounts for pandas .loc including both endpoints.
    Horizons must be ≥24 hours.

    Args:
        datetime_index: DatetimeIndex from price/demand data
        daily_decision_hour: Hour of day (0-23) when decisions are made
        horizon_length: Length of each horizon in hours (must be ≥24)

    Returns:
        List of tuples: (horizon_start, horizon_end)

    Raises:
        ValueError: If horizon_length < 24

    Example:
        index = ['2023-01-01 00:00', ..., '2023-01-03 23:00']
        _create_horizons(index, daily_decision_hour=6, horizon_length=48)

        With 24h decision intervals and 48h horizons:
        -> [('2023-01-01 00:00', '2023-01-01 05:00'),      # partial first horizon (6h)
            ('2023-01-01 06:00', '2023-01-03 05:00'),      # 48h lookahead
            ('2023-01-02 06:00', '2023-01-03 23:00'),      # shortened: data ends
            ('2023-01-03 06:00', '2023-01-03 23:00')]      # final horizon to end
    """
    if horizon_length < 24:
        raise ValueError(
            f"horizon_length must be ≥24 hours, got {horizon_length}. "
            "The formula (next_decision + horizon_length - 24 - 1) requires this."
        )
    # Get all decision times
    decision_times = datetime_index[datetime_index.hour == daily_decision_hour].tolist()

    # Insert the first row's datetime to capture all data from the beginning
    # only if it's not already a decision time
    if datetime_index[0] not in decision_times:
        decision_times.insert(0, datetime_index[0])

    horizons = []

    for i, decision_time in enumerate(decision_times):
        # Calculate horizon end, but don't exceed the end of available data
        if i < len(decision_times) - 1:
            # Horizon extends (horizon_length - 24) hours beyond next decision.
            # Subtract 1 to avoid overlap (pandas .loc includes both endpoints)
            calculated_end = decision_times[i + 1] + pd.DateOffset(
                hours=horizon_length - 24 - 1
            )
        else:
            calculated_end = datetime_index[-1]

        horizon_end = min(calculated_end, datetime_index[-1])

        horizons.append((decision_time, horizon_end))

    logger.info("Created %d horizons", len(horizons))
    return horizons


def moving_horizon(
    price_data: pd.DataFrame,
    demand_data: pd.DataFrame,
    config: dict,
    debug: bool = False,
) -> dict:
    """Moving horizon optimization for demand response.

    Args:
        price_data: DataFrame with datetime index and price column.
            Required column: 'price' (ct/kWh) - spot market price
        demand_data: DataFrame with datetime index and demand column.
            Required column: 'demand' (kWh)
        config: Configuration dictionary containing optimization parameters.
            Required keys:
                'daily_decision_hour' (int): Hour of day (0-23) for daily decisions
                'n_lookahead_hours' (int): Horizon length in hours (must be ≥24)
                'virtual_storage' or 'load_shift' (dict): VirtualStorage
                    initialization parameters
        debug: If True, include detailed debug information in return dict.
            Defaults to False.

    Returns:
        Dictionary with the following keys:
        - "results": DataFrame with datetime index containing:
            - 'demand' (kWh): Optimized demand values
            - 'shift' (kWh): Demand shift values (add_to - remove_from)
        - "debug_info": List of debug dictionaries (only present if debug=True)

    Raises:
        ValueError: If price_data and demand_data have different datetime indices
        ValueError: If required columns are missing from DataFrames
        ValueError: If required config keys are missing
        ValueError: If n_lookahead_hours < 24
    """
    # Verify that price_data and demand_data have the same datetime index
    if not price_data.index.equals(demand_data.index):
        raise ValueError("price_data and demand_data must have the same datetime index")

    # Verify required columns exist
    if "price" not in price_data.columns:
        raise ValueError("price_data must contain a 'price' column")
    if "demand" not in demand_data.columns:
        raise ValueError("demand_data must contain a 'demand' column")

    if "daily_decision_hour" not in config:
        raise ValueError("daily_decision_hour is required in config")

    if "n_lookahead_hours" not in config:
        raise ValueError("n_lookahead_hours is required in config")

    if config["n_lookahead_hours"] < 24:
        raise ValueError(
            f"n_lookahead_hours must be ≥24, got {config['n_lookahead_hours']}"
        )

    # Check for VirtualStorage config (support both key names)
    vs_config = config.get("load_shift") or config.get("virtual_storage")
    if vs_config is None:
        raise ValueError(
            "config must contain either 'virtual_storage' or 'load_shift' key"
        )

    # Create all horizons
    horizons = _create_horizons(
        price_data.index, config["daily_decision_hour"], config["n_lookahead_hours"]
    )

    optimal_demands = []
    optimal_shifts = []
    debug_info_list = [] if debug else None

    # Process each horizon with optimization
    remove_from_spillover = None
    add_to_spillover = None
    for horizon_start, horizon_end in tqdm(horizons, desc="Optimizing horizons"):
        # Extract the price and demand horizons
        price_horizon = price_data.loc[horizon_start:horizon_end]
        demand_horizon = demand_data.loc[horizon_start:horizon_end]

        # n_control_hours is the time until the next decision point
        # (start of next horizon)
        # Alternatively the end of the current horizon if this is the last horizon
        current_horizon_idx = horizons.index((horizon_start, horizon_end))
        n_control_hours = _compute_control_hours(horizons, current_horizon_idx)

        # Support both 'virtual_storage' and 'load_shift' config keys
        vs_config = config.get("load_shift") or config.get("virtual_storage")

        virtual_storage = VirtualStorage(**vs_config)
        result = virtual_storage.optimize_demand(
            price_horizon["price"].values,
            demand_horizon["demand"].values,
            remove_from_history=remove_from_spillover,
            add_to_history=add_to_spillover,
            n_control_hours=n_control_hours,
            debug=debug,
        )

        optimal_demands.append(result["optimal_demand"])
        optimal_shifts.append(result["optimal_shift"])
        remove_from_spillover = result["remove_spillover"]
        add_to_spillover = result["add_spillover"]

        if debug:
            debug_info_list.append(result["debug_info"])

    logger.info("Total horizons processed: %d", len(horizons))

    # Concatenate all results into a single DataFrame
    all_demands = np.concatenate(optimal_demands)
    all_shifts = np.concatenate(optimal_shifts)

    results_df = pd.DataFrame(
        {
            "demand": all_demands,
            "shift": all_shifts,
        },
        index=price_data.index,
    )

    # Build return dictionary
    return_dict = {"results": results_df}

    if debug:
        return_dict["debug_info"] = debug_info_list

    return return_dict
