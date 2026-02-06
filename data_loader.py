import pandas as pd
import numpy as np
from typing import Optional, Tuple

class DataLoader:
    """
    Handles loading, cleaning, and windowing of OHLCV data.
    """
    def __init__(self, filepath: Optional[str] = None, data: Optional[pd.DataFrame] = None):
        if filepath:
            self.df = pd.read_csv(filepath)
        elif data is not None:
            self.df = data.copy()
        else:
            raise ValueError("Either filepath or data must be provided")
        
        self._preprocess()

    def _preprocess(self):
        """Standardizes column names and converts timestamp."""
        # Standardize columns to lowercase
        self.df.columns = [col.lower() for col in self.df.columns]
        
        # Ensure required columns exist
        required = ['timestamp', 'open', 'high', 'low', 'close']
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert timestamp to datetime if numeric
        if pd.api.types.is_numeric_dtype(self.df['timestamp']):
            # Assume unix timestamp in seconds if not specified
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='s')
        else:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)

    def get_window(self, end_idx: int, window_size: int) -> pd.DataFrame:
        """Returns a sliding window of data ending at end_idx."""
        start_idx = max(0, end_idx - window_size + 1)
        return self.df.iloc[start_idx:end_idx + 1].copy()

    def resample(self, timeframe: str) -> pd.DataFrame:
        """
        Resamples data to a higher timeframe (e.g., '5min').
        """
        resampled = self.df.set_index('timestamp').resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum' if 'volume' in self.df.columns else 'first' # volume might be optional
        }).dropna().reset_index()
        return resampled

if __name__ == "__main__":
    # Example usage/test
    print("DataLoader module loaded.")
