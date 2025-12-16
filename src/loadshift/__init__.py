"""Flexibility module for demand response optimization."""

from .moving_horizon import moving_horizon
from .virtual_storage import VirtualStorage

# User-friendly alias
LoadShifter = VirtualStorage

__all__ = ["LoadShifter", "VirtualStorage", "moving_horizon"]
