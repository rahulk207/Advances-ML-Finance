from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
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

    # probability estimates of positive class
    y_pred_prob = clf.predict_proba(X_test)

    # Calculate the accuracy and log loss for the current fold
    return accuracy_score(y_test, y_pred), log_loss(y_test, y_pred_prob)


def trainSideModel(data_path, folds_path, feature_columns, label_column):
    df, folds_index = loadStuff(data_path, folds_path)

    # Define the parameters for the Random Forest Classifier
    n_estimators = 100
    max_depth = 5
    random_state = 42
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    accuracy_list, log_loss_list = [], []
    for train_index, test_index in folds_index:
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]
        df_train = df_train.dropna(subset=[label_column[0]])
        df_test = df_test.dropna(subset=[label_column[0]])

        X_train, y_train = np.array(
            df_train[feature_columns]), np.array(df_train[label_column]) + 1
        X_test, y_test = np.array(
            df_test[feature_columns]), np.array(df_test[label_column]) + 1

        print(np.unique(y_train))
        print(np.unique(y_test))
        # Train model and compute average acc and neg-log-loss
        accuracy, log_loss_ = trainAndInferModel(
            clf, X_train, y_train, X_test, y_test)
        accuracy_list.append(accuracy)
        log_loss_list.append(log_loss_)

    print("Result lists", accuracy_list, log_loss_list)
    avg_accuracy = np.mean(accuracy_list)
    avg_log_loss = np.mean(log_loss_list)

    print("Mean Results", avg_accuracy, avg_log_loss)


if __name__ == '__main__':
    version = "v5_cusum"
    data_path = "training/{0}/dollar_bars_labeled.csv".format(version)
    side_folds_path = "training/{0}/purged_cv_folds_side".format(version)

    feature_columns = ["open", "high", "low", "close", "volume", "volatility"]
    label_column = ['side_label']

    trainSideModel(data_path, side_folds_path, feature_columns, label_column)
