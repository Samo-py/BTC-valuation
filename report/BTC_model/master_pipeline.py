import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import asyncio

from pipelines.live_hash import live_hash
from pipelines.live_price import live_price

from pipelines.environment_pipeline import environment_pipeline
from pipelines.emission_pipeline import emission_pipeline
from pipelines.machines_pipeline import machines_pipeline



class Pipeline:
    def __init__(self, rate_df: pd.DataFrame, antminers_df: pd.DataFrame, environment_df: pd.DataFrame, emission_df: pd.DataFrame, halving_dates: np.ndarray, coef):
        self.df = pd.merge(pd.merge(pd.merge(rate_df, antminers_df, how='inner', on='date'), environment_df, how='inner', on='date'), emission_df, how='inner', on='date')
        self.halving_dates = halving_dates
        self.result_est = np.zeros(len(self.df.index))
        self.timeseries = self.df.index
        self.coef = coef

    def run(self):
        self.result_est = ((((self.df['hash_rate'] / self.df['ant_hash_rate']) * self.df['ant_power']) * 24 * 0.05) / (self.df['emission'] * 144) * 2.5) * (self.df['env_efficiency'] / self.df['ant_efficiency'])
        self.result_max = self.result_est * self.coef
        self.result_min = self.result_est * (1/self.coef)

        self.log_ratios = np.log2(self.df['market_price'] / self.result_est)

        return self


class Visualizer:
    def __init__(self, timeseries, result_est, result_min, result_max, halving_dates, market_price):
        self.timeseries = timeseries
        self.result_est = result_est
        self.result_min = result_min
        self.result_max = result_max
        self.market_price = market_price
        self.halving_dates = halving_dates

        self.xticks = []
        self.xticks_labels = []

    def compute_xticks(self):
        self.xticks = np.array(self.halving_dates[2:], dtype="datetime64[ns]")
        self.xticks_labels = self.xticks.astype('datetime64[Y]')
        return self

    def visualize(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.timeseries, self.result_est, color='blue',label='est value')
        plt.plot(self.timeseries, self.result_min, color='gray', label='oversold')
        plt.plot(self.timeseries, self.result_max, color='gray', label='overbought')
        plt.plot(self.timeseries, self.market_price, color='orange', label='market price')
        plt.yscale('log')
        plt.xticks(self.xticks, self.xticks_labels)
        for x in self.xticks:
            plt.axvline(x=x, color='red', linestyle='--', linewidth = 1)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.tight_layout()
        plt.legend()
        plt.grid(True)
        # plt.savefig("images/fig.png")
        plt.show()

def master_pipeline():
    coef = 1.6

    price_df = asyncio.run(live_price())
    hash_df = asyncio.run(live_hash())
    df = pd.merge(hash_df, price_df, on="date", how="inner")
    pipeline = Pipeline(df, machines_pipeline(), environment_pipeline(), *emission_pipeline(), coef).run()
    visualizer = Visualizer(pipeline.df["date"], pipeline.result_est, pipeline.result_min, pipeline.result_max, pipeline.halving_dates, pipeline.df['market_price']).compute_xticks()
    visualizer.visualize()

if __name__ == "__main__":
    master_pipeline()