import pandas as pd
import numpy as np

class Dataloader:

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.dates = None
        self.values = None
        self.df = None
    
    def load_emission(self):
        df = pd.read_csv(self.filepath)
        self.dates = pd.to_datetime(df['date']).to_numpy()
        self.values = df['emission'].to_numpy()
        return self
    
    def load_fees(self):
        self.df = pd.read_csv(self.filepath)
        self.df['date'] = pd.to_datetime(self.df['date'])
        return self
    
class Pipeline:

    def __init__(self, dates: np.ndarray, values: np.ndarray):
        self.dates = dates
        self.values = values
        self.timeseries = None
        self.emission = None

    def coupler(self, x, a, b, min, max):
        range_val = max - min
        norm = (x - a) / (b - a)
        t = np.clip(norm, 0, 1)
        s = 3*t**2 - 2*t**3
        return min + s*range_val 

    def run(self, transition_days: int = 574):
        self.timeseries = pd.date_range(min(self.dates), max(self.dates), freq='D').to_numpy()
        self.emission = np.zeros(len(self.timeseries))

        transition_starts = self.dates[1:] - np.timedelta64(transition_days, 'D')

        for i in range(len(self.dates)-1):
            mask = (self.timeseries >= self.dates[i]) & (self.timeseries[i] <= transition_starts[i])
            self.emission[mask] = self.values[i]

            mask = (self.timeseries >= transition_starts[i]) & (self.timeseries <= self.dates[i + 1])
            self.emission[mask] = self.coupler(self.timeseries[mask], transition_starts[i], self.dates[i + 1], self.values[i], self.values[i+1])
        return self
    
def emission_pipeline():
    emission_path = 'BTC_model/data/emission.csv'
    transition_D = 500
    
    emission_loader = Dataloader(emission_path).load_emission()
    pipeline = Pipeline(emission_loader.dates, emission_loader.values).run(transition_D)
    
    df =  pd.DataFrame({'date': pipeline.timeseries, 'emission': pipeline.emission})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    return df, emission_loader.dates