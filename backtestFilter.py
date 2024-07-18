from createPurgedCVFolds import purgeDataframe
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import pickle
from backtesting import *
import random


def fetch_filter_indices(data_path, df):
    f = open(data_path, "rb")
    train_indices = pickle.load(f)
    all_indices = set(range(len(df)))
    test_indices = list(all_indices.difference(set(train_indices)))
    # random.shuffle(train_indices)
    print(df.iloc[train_indices]['side_label'].value_counts())
    print(df.iloc[train_indices]['size_label'].value_counts())
    print(len(test_indices))
    return train_indices, test_indices


def backtest_filter(data_path, filter_index_path, feature_columns):
    backtest_df = load_stuff(data_path)
    print(len(backtest_df))
    # Define the parameters for the "side" Random Forest Classifier
    n_estimators = 100
    max_depth = 20
    random_state = 42
    side_clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    # Define the parameters for the "size" Random Forest Classifier
    n_estimators = 100
    max_depth = 5
    random_state = 42
    size_clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    start_indices, close_indices, sides = [], [], []
    percentage_returns = []
    c_expiry = 0
    c_sl = 0
    c_pt = 0

    train_index, test_index = fetch_filter_indices(
        filter_index_path, backtest_df)
    side_pred, _ = purgeAndTrainAndInfer(
        backtest_df, train_index, test_index, "side_label_range", side_clf, feature_columns, ["side_label"])
    size_pred, size_pred_prob = purgeAndTrainAndInfer(
        backtest_df, train_index, test_index, "size_label_range", size_clf, feature_columns, ["size_label"])
    print(len(np.where(side_pred == 2)[0]))
    print(len(np.where(side_pred == 0)[0]))
    # print(side_pred)
    for i in range(len(test_index)):
        side = side_pred[i] - 1
        size = size_pred[i]
        # size = calBetSize(size_pred[i], size_pred_prob[i])
        if (side != 0) and (size != 0):
            close_index, percentage_return, expiry_flag = simulateBet(
                backtest_df, side, test_index[i])
            percentage_returns.append(percentage_return)
            start_indices.append(test_index[i])
            close_indices.append(close_index)
            sides.append(side)
            if expiry_flag == True:
                if percentage_return >= 0:
                    c_pt += 1
                else:
                    c_sl += 1
            else:
                c_expiry += 1

    print("Num Expiry", c_expiry, "Num SL", c_sl, "Num PT", c_pt)

    # The higher the better
    return calSharpeRatio(percentage_returns), percentage_returns, start_indices, close_indices, sides


if __name__ == '__main__':
    version = "large/v1"
    # data_path = "training/{0}/dollar_bars_feature_differenced.csv".format(
    #     version)
    data_path = "training/{0}/dollar_bars_labeled.csv".format(
        version)
    filter_indices_path = "training/{0}/cusum_filter_indices.pickle".format(
        version)
    feature_columns = ["open", "high", "low", "close", "volume", "volatility"]
    sharpe_ratio, percentage_returns, start_indices, close_indices, sides = backtest_filter(
        data_path, filter_indices_path, feature_columns)
    # print(percentage_returns)
    c_positive_return = 0
    for p in percentage_returns:
        if p > 0:
            c_positive_return += 1

    print("Number of positive returns", c_positive_return)
    print("Number of negative returns", len(
        percentage_returns) - c_positive_return)

    c_short_negative = 0
    c_long_negative = 0
    for i in range(len(sides)):
        if sides[i] == -1 and percentage_returns[i] < 0:
            c_short_negative += 1
        elif sides[i] == 1 and percentage_returns[i] < 0:
            c_long_negative += 1

    print("Number of short trades with negative returns", c_short_negative)
    print("Number of long trades with negative returns", c_long_negative)

    print(sharpe_ratio, np.mean(percentage_returns), np.sum(percentage_returns))

    # f = open("analysis/{0}/percentage_returns".format(version), "wb")
    # pickle.dump(percentage_returns, f)

    # f = open("analysis/{0}/backtest_start_indices".format(version), "wb")
    # pickle.dump(start_indices, f)

    # f = open("analysis/{0}/backtest_close_indices".format(version), "wb")
    # pickle.dump(close_indices, f)

    # f = open("analysis/{0}/backtest_sides".format(version), "wb")
    # pickle.dump(sides, f)
