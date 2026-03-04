import asyncio
import aiohttp
import pandas as pd
import matplotlib.pyplot as plt

async def live_hash():
    url = "https://api.blockchain.info/charts/hash-rate?timespan=all&format=json"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            dataframe = pd.DataFrame(data["values"]).rename(columns={"x": "date", "y": "hash_rate"})
            dataframe["date"] = pd.to_datetime(dataframe['date'], unit="s")
            dataframe = dataframe.set_index("date").resample("D").interpolate(method="linear")


            return dataframe
        
if __name__ == "__main__":
    dataframe = asyncio.run(live_hash())
    print(dataframe)