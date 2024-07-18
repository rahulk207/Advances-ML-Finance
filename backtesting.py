from createPurgedCVFolds import purgeDataframe
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import pickle
import random

# Here, use a different data than the one used in training. Use 'size' as the label_range_column


def load_stuff(data_path):

    return pd.read_csv(data_path)


def calBetSize(size_pred, size_pred_prob):

    return


# Verify this function. Seems wrong.
def calRelativeReturn(open_price, close_price, side):

    return side*(close_price - open_price)/open_price


def calAbsoluteReturn(open_price, close_price, side, size):

    return side*(close_price - open_price)*size


def calSharpeRatio(percentage_returns):

    return np.mean(percentage_returns)/np.std(percentage_returns)


def simulateBet(backtest_df, side, i):  # Add support for size too
    close_index = i
    flag = False
    j = i + 1
    while j < min(len(backtest_df), i + 1 + backtest_df.iloc[i]['expiry']):
        if (side == 1 and backtest_df.iloc[j]['close'] >= backtest_df.iloc[i]['close'] + backtest_df.iloc[i]['profit_taking']) or (side == -1 and backtest_df.iloc[j]['close'] <= backtest_df.iloc[i]['close'] - backtest_df.iloc[i]['profit_taking']):
            close_index = j
            flag = True
            break
        elif (side == 1 and backtest_df.iloc[j]['close'] <= backtest_df.iloc[i]['close'] - backtest_df.iloc[i]['stop_loss']) or (side == -1 and backtest_df.iloc[j]['close'] >= backtest_df.iloc[i]['close'] + backtest_df.iloc[i]['stop_loss']):
            close_index = j
            flag = True
            break
        j += 1

    if flag == False:
        close_index = j - 1

    open_price = backtest_df.iloc[i]['close']
    close_price = backtest_df.iloc[close_index]['close']

    # return calRelativeReturn(open_price, close_price), calAbsoluteReturn(open_price, close_price, size)
    # Use the above return when added support for bet_size
    return close_index, calRelativeReturn(open_price, close_price, side)*100, flag


def printFeatureImportances(clf, feature_columns):
    importances = clf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    print("Feature mportances")
    for i in range(len(feature_columns)):
        print("%2d) %-*s %f" % (i + 1, 30,
              feature_columns[sorted_indices[i]], importances[sorted_indices[i]]))


def purgeAndTrainAndInfer(backtest_df, train_index, test_index, label_range_column, clf, feature_columns, label_column):
    train_index_purged, test_index_purged = purgeDataframe(
        backtest_df, list(train_index), list(test_index), label_range_column)

    df_train, df_test = backtest_df.iloc[train_index_purged], backtest_df.iloc[test_index_purged]
    df_train = df_train.dropna(subset=[label_column[0]])
    # Don't use label_column here so we get equal number of predictions for both side and size
    # df_test = df_test.dropna(subset=["volatility"])

    X_train, y_train = np.array(
        df_train[feature_columns]), np.array(df_train[label_column])
    X_test, y_test = np.array(
        df_test[feature_columns]), np.array(df_test[label_column])

    if label_column[0] == "side_label":
        y_train += 1
        y_test += 1

    clf.fit(X_train, y_train)

    printFeatureImportances(clf, feature_columns)

    y_pred = clf.predict(X_test)
    # probability estimates of positive class
    y_pred_prob = clf.predict_proba(X_test)

    return y_pred, y_pred_prob


def filterTrainSamples(train_indices, filter_indices):
    filter_indices_set = set(filter_indices)
    train_filter_indices = []
    for i in train_indices:
        if i in filter_indices_set:
            train_filter_indices.append(i)

    return train_filter_indices


def backtest(data_path, feature_columns, filter_indices=None):

    backtest_df = load_stuff(data_path)

    n_folds = 5
    kfolds = KFold(n_splits=n_folds, shuffle=False)

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
    for train_index, test_index in kfolds.split(backtest_df):

        if filter_indices:
            train_index = filterTrainSamples(train_index, filter_indices)
        print("Num training samples", len(train_index))

        side_pred, _ = purgeAndTrainAndInfer(
            backtest_df, train_index, test_index, "side_label_range", side_clf, feature_columns, ["side_label"])
        size_pred, size_pred_prob = purgeAndTrainAndInfer(
            backtest_df, train_index, test_index, "size_label_range", size_clf, feature_columns, ["size_label"])

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

    # Use the same training code you'll write in the other two files
    # Combine the performance for each fold. For measuring performance, refer the above functions. We'll have to simulate the bet.
    # Follow the below steps -
    # 1. Once you get side and size prediction - only if size_pred==1, simulate the bet. Need to see what barrier does it hit - pt, sl, expiry?
    # 2. Take side_pred into consideration to decide which pt, sl to use. Take help from earlier labelling code to write this.
    # 3. Use size_prob to calculate the bet size. Then, based on which barrier hits - calculate the return and Sharpe Ratio.


if __name__ == '__main__':
    version = "large/v2"
    data_path = "training/{0}/dollar_bars_labeled.csv".format(version)

    filter_indices_path = "training/{0}/cusum_filter_indices.pickle".format(
        version)
    f = open(filter_indices_path, "rb")
    filter_indices = pickle.load(f)

    feature_columns = ["open", "high", "low", "close", "volume", "volatility"]
    sharpe_ratio, percentage_returns, start_indices, close_indices, sides = backtest(
        data_path, feature_columns, filter_indices)

    print(sharpe_ratio, np.mean(percentage_returns), np.sum(percentage_returns))

    # f = open("analysis/{0}/percentage_returns".format(version), "wb")
    # pickle.dump(percentage_returns, f)

    # f = open("analysis/{0}/backtest_start_indices".format(version), "wb")
    # pickle.dump(start_indices, f)

    # f = open("analysis/{0}/backtest_close_indices".format(version), "wb")
    # pickle.dump(close_indices, f)

    # f = open("analysis/{0}/backtest_sides".format(version), "wb")
    # pickle.dump(sides, f)
