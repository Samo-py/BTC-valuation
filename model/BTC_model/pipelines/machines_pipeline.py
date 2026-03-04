import pandas as pd
import numpy as np

class DataLoader():

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.dates = None
        self.hash_rate_values = None
        self.power_values = None
        self.efficiency_values = None

    def load(self):
        df = pd.read_csv(self.filepath)
        self.dates = pd.to_datetime(df['release_date']).to_numpy()
        self.hash_rate_values = df['hashrate_th'].to_numpy()
        self.power_values = df['power_kw'].to_numpy()
        self.efficiency_values = df['efficiency_jth'].to_numpy()
        return self
    
class Pipeline:

    def __init__(self, dates: np.ndarray, hashrate: np.ndarray, power: np.ndarray, efficiency: np.ndarray):
        self.dates = dates
        self.hash_rate_values = hashrate
        self.power_values = power
        self.efficiency_values = efficiency
        self.timeseries = None
        self.hash_rate = None
        self.power = None
        self.efficiency = None

    def coupler(self, x, a, b, min, max):
        range_val = max - min
        norm = (x - a) / (b - a)
        t = np.clip(norm, 0, 1)
        s = 3*t**2 - 2*t**3
        return min + s*range_val
    
    def run(self, end_date, transition_days):
        self.timeseries = pd.date_range(min(self.dates), end_date, freq='D')
        data_l = [
            {'data': np.zeros(len(self.timeseries)), 'values' : self.hash_rate_values},
            {'data': np.zeros(len(self.timeseries)), 'values' : self.power_values},
            {'data': np.zeros(len(self.timeseries)), 'values' : self.efficiency_values}
            ]

        transition_starts = self.dates[1:] - np.timedelta64(transition_days, 'D')
        for block in data_l:
            data = block['data']
            values = block['values']
            for i in range(len(self.dates)-1):
                mask = (self.timeseries >= self.dates[i]) & (self.timeseries[i] <= transition_starts[i])
                data[mask] = values[i]

                mask = (self.timeseries >= transition_starts[i]) & (self.timeseries <= self.dates[i + 1])
                data[mask] = self.coupler(self.timeseries[mask], transition_starts[i], self.dates[i + 1], values[i], values[i+1])
            mask = self.timeseries >= self.dates[-1]
            data[mask] = values[-1]
            
        self.hash_rate = data_l[0]['data']
        self.power = data_l[1]['data']
        self.efficiency = data_l[2]['data']

        return self
    
def machines_pipeline():
    path = 'BTC_model/data/antminers.csv'
    transition_D = 120
    end_date = '2028-01-01'

    loader = DataLoader(path).load()
    pipeline = Pipeline(loader.dates, loader.hash_rate_values, loader.power_values, loader.efficiency_values).run(end_date, transition_D)
    

    df = pd.DataFrame({'date': pipeline.timeseries, 'ant_hash_rate': pipeline.hash_rate, 'ant_power': pipeline.power, 'ant_efficiency': pipeline.efficiency})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    
    return df