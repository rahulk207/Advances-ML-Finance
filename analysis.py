from visualizeBars import visualizeBars
import pandas as pd
import pickle


def plotAnalysis(df, bet_indices, close_bet_indices, backtest_start_indices, backtest_close_indices, backtest_sides, percentage_returns):

    # visualizeBars(df, bet_indices)
    # visualizeBars(df, close_bet_indices)
    # visualizeBars(df, backtest_start_indices)
    # visualizeBars(df, backtest_close_indices)
    for i in range(len(backtest_start_indices)):
        if backtest_start_indices[i] < 20:
            print(
                backtest_start_indices[i], backtest_close_indices[i], backtest_sides[i], percentage_returns[i])


if __name__ == '__main__':
    version = "v0"
    data_path = "training/{0}/dollar_bars_labeled.csv".format(version)
    df = pd.read_csv(data_path)

    bet_indices = []
    close_bet_indices = []
    for i in range(len(df)):
        if df.iloc[i]['size_label'] == 1:
            bet_indices.append(i)
            close_range = df.iloc[i]['size_label_range'][1:-1]
            close_bet_indices.append(
                tuple(map(int, close_range.split(', ')))[1])

    f = open("analysis/{0}/backtest_start_indices".format(version), "rb")
    backtest_start_indices = pickle.load(f)

    f = open("analysis/{0}/backtest_close_indices".format(version), "rb")
    backtest_close_indices = pickle.load(f)

    f = open("analysis/{0}/backtest_sides".format(version), "rb")
    backtest_sides = pickle.load(f)

    f = open("analysis/{0}/percentage_returns".format(version), "rb")
    percentage_returns = pickle.load(f)

    plotAnalysis(df, bet_indices, close_bet_indices,
                 backtest_start_indices, backtest_close_indices, backtest_sides, percentage_returns)
