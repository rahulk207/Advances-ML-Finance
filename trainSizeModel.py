from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import pickle


def loadStuff(data_path, folds_path):
    df = pd.read_csv(data_path)
    f = open(folds_path, "rb")
    folds_index = pickle.load(f)

    return df, folds_index


def trainAndInferModel(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    # Calculate the accuracy and log loss for the current fold
    return precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)


def trainSizeModel(data_path, folds_path, feature_columns, label_column):
    df, folds_index = loadStuff(data_path, folds_path)

    # Define the parameters for the Random Forest Classifier
    n_estimators = 100
    max_depth = 5
    random_state = 42
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    precision_list, recall_list, f1_list = [], [], []
    for train_index, test_index in folds_index:
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]
        df_train = df_train.dropna(subset=[label_column[0]])
        df_test = df_test.dropna(subset=[label_column[0]])

        X_train, y_train = np.array(
            df_train[feature_columns]), np.array(df_train[label_column])
        X_test, y_test = np.array(
            df_test[feature_columns]), np.array(df_test[label_column])

        # Train model and use F1-score to select the best one
        precision, recall, f1 = trainAndInferModel(
            clf, X_train, y_train, X_test, y_test)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    print("Result lists", precision_list, recall_list, f1_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)

    print("Mean Results", avg_precision, avg_recall, avg_f1)


if __name__ == '__main__':
    version = "v5_cusum"
    data_path = "training/{0}/dollar_bars_labeled.csv".format(version)
    side_folds_path = "training/{0}/purged_cv_folds_size".format(version)

    feature_columns = ["open", "high", "low", "close", "volume", "volatility"]
    label_column = ['size_label']

    trainSizeModel(data_path, side_folds_path, feature_columns, label_column)
