import torch
from torch.utils.data import IterableDataset
from pathlib import Path
import pandas as pd
import numpy as np
import random
import re


class TurbineStreamingDataset(IterableDataset):

    def __init__(
        self,
        data_dir,
        target_col="power_output",
        num_lags=24,
        static_cols=("capacity", "age"),
        shuffle=True,
        forecast_horizon=1,
    ):
        self.paths = sorted(Path(data_dir).glob("GSRN=*"))
        self.target_col = target_col
        self.num_lags = num_lags
        self.static_cols = list(static_cols)
        self.shuffle = shuffle
        self.forecast_horizon = forecast_horizon

    def _parse_dynamic_features(self, columns):
        """
        从 lag 列中推断 dynamic feature 名称
        e.g. wind_speed_lag_1 → wind_speed
        """

        pattern = re.compile(r"(.+)_lag_\d+")
        dynamic = set()

        for col in columns:
            match = pattern.match(col)
            if match:
                dynamic.add(match.group(1))

        return sorted(dynamic)

    def __iter__(self):

        paths = self.paths.copy()

        if self.shuffle:
            random.shuffle(paths)

        for path in paths:

            df = pd.read_parquet(path)

            columns = df.columns.tolist()

            dynamic_features = self._parse_dynamic_features(columns)

            # sliding window: stop forecast_horizon rows before the end
            # so we can look ahead for multi-step targets
            for i in range(len(df) - self.forecast_horizon):

                row = df.iloc[i]

                # --------------------
                # build time series
                # --------------------

                seq = []

                for lag in range(self.num_lags, 0, -1):

                    timestep = []

                    # dynamic lag features
                    for feat in dynamic_features:
                        timestep.append(row[f"{feat}_lag_{lag}"])

                    # static features (broadcast to every timestep)
                    for s in self.static_cols:
                        timestep.append(row[s])

                    seq.append(timestep)

                X = np.array(seq, dtype=np.float32)

                # multi-step targets: t+1, t+2, ..., t+forecast_horizon
                y_vals = [
                    df.iloc[i + step][self.target_col]
                    for step in range(1, self.forecast_horizon + 1)
                ]
                y = np.array(y_vals, dtype=np.float32)

                yield torch.from_numpy(X), torch.from_numpy(y)