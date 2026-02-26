import pandas as pd
import numpy as np

class DataLoader:


    def __init__(self, filepath: str):
        self.filepath = filepath
        self.dates = None
        self.eff = None
    
    def load(self):
        df = pd.read_csv(self.filepath)
        self.dates = pd.to_datetime(df['date']).to_numpy()
        self.eff = df['avg_efficiency'].to_numpy()
        return self
    
class Pipeline:


    def __init__(self, dates: np.ndarray, eff: np.ndarray):
        self.dates = dates
        self.efficiency_values = eff
        self.timeseries = None
        self.efficiency = None

    def coupler(self, x, a, b, a_val, b_val):
        range_val = b_val - a_val
        norm = (x - a) / (b - a)
        t = np.clip(norm, 0, 1)
        s = 3*t**2 - 2*t**3
        return a_val + s*range_val
    
    def run(self, end_date):
        self.timeseries = pd.date_range(min(self.dates), end_date, freq='D')
        self.efficiency = np.zeros(len(self.timeseries))

        
        for i in range(len(self.dates) - 1):
            mask = (self.timeseries >= self.dates[i]) & (self.timeseries <= self.dates[i+1])
            self.efficiency[mask] = self.coupler(self.timeseries[mask], self.dates[i], self.dates[i+1], self.efficiency_values[i], self.efficiency_values[i+1])
        mask = (self.timeseries >= self.dates[-1]) & (self.timeseries <= end_date)
        self.efficiency[mask] = self.efficiency_values[-1]

        return self
    
    
def environment_pipeline():
    filepath = 'BTC_model/data/efficiency.csv'

    loader = DataLoader(filepath).load()
    pipeline = Pipeline(loader.dates, loader.eff).run('2028-01-01')

    df = pd.DataFrame({'date': pipeline.timeseries, 'env_efficiency': pipeline.efficiency})


    return df
