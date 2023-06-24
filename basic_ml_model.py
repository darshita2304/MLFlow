import pandas as pd
import numpy as np
import os

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import argparse


def get_data():
    df = pd.read_csv("wine-data.csv", sep=";")

    return df
    # print(df)


def evaluate(y_true, y_pred):

    # mae = mean_absolute_error(y_true, y_pred)
    # mse = mean_squared_error(y_true, y_pred)
    # rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # r2 = r2_score(y_true, y_pred)

    ascore = accuracy_score(y_true, y_pred)

    # return mae, mse, rmse, r2
    return ascore


# def main():
def main(n_estimator, max_depth):
    df = get_data()
    # print("hello from main...")
    # train, test = train_test_split(df)

    # X_train = train.drop(["quality"], axis=1)
    # X_test = test.drop(["quality"], axis=1)

    # y_train = train[["quality"]]
    # y_test = test[["quality"]]

    # all rows, column last ignore -1 ---- creating independent features dataset
    X = df.iloc[:, df.columns != 'quality']
    y = df.iloc[:, df.columns == 'quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # lr = ElasticNet(n_estimator, max_depth)
    lr = ElasticNet()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    # mae, mse, rmse, r2 = evaluate(y_test, y_pred)
    ascore = evaluate(y_test, y_pred)

    # print(f"mae={mae}, mse={mse}, rmse={rmse}, r2={r2}")
    print(f"acore={ascore}")


# if __name__ == '__main__':
#     try:
#         main()
#     except Exception as e:
#         raise e

if __name__ == '__main__':
    try:
        args = argparse.ArgumentParser()
        args.add_argument("--n_esitmators", '-n', default=50, type=int)
        args.add_argument("--max_depth", "-m", default=25.0, type=int)

        parse_args = args.parse_args()

        # main()
        main(n_estimator=parse_args.n_esitmators,
             max_depth=parse_args.max_depth)
    except Exception as e:
        raise e
