import pandas as pd
import numpy as np
import pickle


def cusum(df, thresh):
    S_plus = 0
    S_minus = 0
    samples_indices = []
    for i in range(1, len(df)):
        S_plus = max(S_plus + df.iloc[i]['close'] - df.iloc[i-1]['close'], 0)
        S_minus = min(S_minus + df.iloc[i]['close'] - df.iloc[i-1]['close'], 0)

        S_t = max(S_plus, -S_minus)
        if S_t > thresh:
            samples_indices.append(i)
            S_plus = 0
            S_minus = 0

    return samples_indices


if __name__ == '__main__':
    df1 = pd.read_csv(
        "drive_data/dollar_bars/1613399976187-1615899928558-1000000.csv")
    df2 = pd.read_csv(
        "drive_data/dollar_bars/1615900014959-1618399971101-1000000.csv")
    df = pd.concat([df1, df2], ignore_index=True)
    # change this for every security. Think if it should sepend on volatility.
    threshold = 4
    sample_indices = cusum(df, threshold)
    print("Num filter samples", len(sample_indices))

    version = "large/v2"
    data_path_output = "training/{0}/cusum_filter_indices.pickle".format(
        version)

    f = open(data_path_output, "wb")
    pickle.dump(sample_indices, f)
