import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np


def visualizeBars(df, highlight_indices):
    df['timestamp_end'] = pd.to_datetime(df['timestamp_end'], unit='ms')
    print(df)

    # Convert the data to the format expected by mplfinance
    ohlc_data = df[["timestamp_end", "open", "high", "low",
                    "close", "volume"]].set_index("timestamp_end")

    highlight_ts = np.array(df['timestamp_end'])[highlight_indices]
    highlight_ts = [pd.to_datetime(str(date))
                    for date in highlight_ts]

    # Plot the data using mplfinance
    mpf.plot(ohlc_data, type='candle', volume=True, vlines={
             "vlines": highlight_ts, "linestyle": "dotted", "linewidths": 0.7})
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("data_small/dollar_bars.csv")
    visualizeBars(df)
