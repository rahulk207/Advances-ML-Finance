from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import math


def loadStuff(path):

    return pd.read_csv(path)


def purgeDataframe(df, train_indices, test_indices, label_range_column):
    # This code will currently only work for constant expiry across dataset

    # Left-purging
    start_train_index = max(
        0, test_indices[0] - df.iloc[test_indices[0]]['expiry'])
    for i in range(start_train_index, test_indices[0]):
        for j in test_indices:  # Can optimise this loop considering expiry
            if type(df.iloc[i][label_range_column]) == str and type(df.iloc[j][label_range_column]) == str:
                range1 = df.iloc[i][label_range_column][1:-1]
                range2 = df.iloc[j][label_range_column][1:-1]
                range1 = tuple(map(int, range1.split(', ')))
                range2 = tuple(map(int, range2.split(', ')))
                if range1[1] >= range2[0] and i in train_indices:
                    train_indices.remove(i)
                    break

    # Right-purging
    end_train_index = min(
        len(df)-1, test_indices[-1] + df.iloc[test_indices[-1]]['expiry'])
    for i in range(test_indices[-1]+1, end_train_index):
        for j in test_indices:
            if type(df.iloc[i][label_range_column]) == str and type(df.iloc[j][label_range_column]) == str:
                range1 = df.iloc[i][label_range_column][1:-1]
                range2 = df.iloc[j][label_range_column][1:-1]
                range1 = tuple(map(int, range1.split(', ')))
                range2 = tuple(map(int, range2.split(', ')))
                if range1[0] <= range2[1] and i in train_indices:
                    train_indices.remove(i)
                    break

    return train_indices, test_indices


def purgeCV(input_path, output_path, label_range_column):
    n_folds = 5
    kfolds = KFold(n_splits=n_folds, shuffle=False)

    df = loadStuff(input_path)

    folds_index = []
    for train_index, test_index in tqdm(kfolds.split(df)):
        train_index_purged, test_index_purged = purgeDataframe(
            df, list(train_index), list(test_index), label_range_column)
        folds_index.append((train_index_purged, test_index_purged))

    # Write the fold index to some location
    f = open(output_path, "wb")
    pickle.dump(folds_index, f)


if __name__ == '__main__':
    version = "v5_cusum"
    purgeCV("training/{0}/dollar_bars_labeled.csv".format(version),
            "training/{0}/purged_cv_folds_side".format(version), "side_label_range")

    purgeCV("training/{0}/dollar_bars_labeled.csv".format(version),
            "training/{0}/purged_cv_folds_size".format(version), "size_label_range")
