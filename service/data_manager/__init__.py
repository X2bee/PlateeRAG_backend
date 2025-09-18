"""
Data Manager Service
"""
from .data_manager import DataManager
from .data_manager_register import DataManagerRegistry

__all__ = [
    'DataManager',
    'DataManagerRegistry',
]
