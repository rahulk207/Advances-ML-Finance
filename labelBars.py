import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm


def labelSize(df, filter_indices_set):
    size_labels = []
    size_labels_range = []
    for i in tqdm(range(len(df))):
        side = df.iloc[i]['side_label']
        if i not in filter_indices_set or side == 0:
            size_labels.append(None)
            size_labels_range.append(None)
            continue

        flag = False
        j = i + 1
        while j < min(len(df), i + 1 + df.iloc[i]['expiry']):
            if (side == 1 and df.iloc[j]['close'] >= df.iloc[i]['close'] + df.iloc[i]['profit_taking']) or (side == -1 and df.iloc[j]['close'] <= df.iloc[i]['close'] - df.iloc[i]['profit_taking']):
                size_labels.append(1)
                size_labels_range.append((i+1, j))
                flag = True
                break
            elif (side == 1 and df.iloc[j]['close'] <= df.iloc[i]['close'] - df.iloc[i]['stop_loss']) or (side == -1 and df.iloc[j]['close'] >= df.iloc[i]['close'] + df.iloc[i]['stop_loss']):
                size_labels.append(0)
                size_labels_range.append((i+1, j))
                flag = True
                break
            j += 1

        if flag == False:
            size_labels.append(0)
            size_labels_range.append((i+1, j-1))

    df['size_label'] = size_labels
    df['size_label_range'] = size_labels_range


def labelSide(df, filter_indices_set):
    side_label = []
    side_labels_range = []
    for i in tqdm(range(len(df))):
        if i not in filter_indices_set:
            side_label.append(None)
            side_labels_range.append(None)
            continue

        flag = False
        j = i + 1
        while j < min(len(df), i + 1 + df.iloc[i]['expiry']):
            if df.iloc[j]['close'] >= df.iloc[i]['close'] + df.iloc[i]['side_threshold']:
                side_label.append(1)
                side_labels_range.append((i+1, j))
                flag = True
                break
            elif df.iloc[j]['close'] <= df.iloc[i]['close'] - df.iloc[i]['side_threshold']:
                side_label.append(-1)
                side_labels_range.append((i+1, j))
                flag = True
                break
            j += 1

        if flag == False:
            side_label.append(0)
            side_labels_range.append((i+1, j-1))

    df['side_label'] = side_label
    df['side_label_range'] = side_labels_range


def calVolatility(df):
    # Should I calculate volatility based on time only. Like let's say volatility in the last day
    window_size = 40
    df['volatility'] = df['close'].ewm(span=window_size).std()


def calStopLoss(df):
    risk_factor = 2
    df['stop_loss'] = risk_factor*df['volatility']


def calProfitTaking(df):
    profit_factor = 4
    df['profit_taking'] = profit_factor*df['volatility']


def calExpiry(df):
    df['expiry'] = 40


def calSideThreshold(df):
    factor = 5
    df['side_threshold'] = factor*df['volatility']


if __name__ == '__main__':
    version = "large/v2"

    df1 = pd.read_csv(
        "drive_data/dollar_bars/1613399976187-1615899928558-1000000.csv")
    df2 = pd.read_csv(
        "drive_data/dollar_bars/1615900014959-1618399971101-1000000.csv")
    df = pd.concat([df1, df2], ignore_index=True)

    filter_indices_path = "training/{0}/cusum_filter_indices.pickle".format(
        version)
    data_path_output = "training/{0}/dollar_bars_labeled.csv".format(version)
    # data_path_output_test = "training/test/dollar_bars_labeled.csv"

    f = open(filter_indices_path, "rb")
    filter_indices = pickle.load(f)

    calVolatility(df)
    df = df.dropna(subset=["volatility"])

    calSideThreshold(df)
    calExpiry(df)
    calProfitTaking(df)
    calStopLoss(df)
    labelSide(df, set(filter_indices))
    labelSize(df, set(filter_indices))

    print("Number of bars", len(df))

    print("Side labels Count")
    print(df['side_label'].value_counts())

    print("Size labels Count")
    print(df['size_label'].value_counts())

    # num_train = int(0.7*len(df))
    # df_train = df[:num_train]
    # df_test = df[num_train:]

    # df_train.to_csv(data_path_output_train)
    # df_test.to_csv(data_path_output_test)

    df.to_csv(data_path_output)
