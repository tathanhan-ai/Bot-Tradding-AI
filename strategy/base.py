"""
Base Strategy Abstract Class
"""
from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd


class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(self, data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate trading signals from multi-timeframe data.
        data_map: dictionary mapping timeframe (e.g., '1h', '15m') to OHLCV DataFrame.
        Returns a DataFrame on the entry timeframe with columns:
          - signal: 1 (Long entry), -1 (Short entry), 0 (Hold/No trade)
          - stop_loss: target stop loss price
          - take_profit: target take profit price
        """
        pass
