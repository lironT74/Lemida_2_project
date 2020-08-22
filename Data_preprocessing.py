from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from auxillary_functions import get_x_any_y


def show_hist():
    df = pd.read_csv(r'london_merged.csv')
    df["cnt"].hist(figsize=(5, 5), grid=False, bins=100)
    buckets = 3
    colors = ["red", "orange", "green"]
    for i in range(1, buckets+1):
        print(df["cnt"].quantile(q=i*1/buckets))
        plt.vlines(x=df["cnt"].quantile(q=i/buckets), ymin=0, ymax=1750, colors=colors[i-1],
                   label=f"{np.round(i/buckets, 2)} quantile")

    plt.title("histogram of bicycles count")
    plt.legend()
    plt.xlabel("bike counts")
    plt.show()


def prepare_dataset():
    df = pd.read_csv(r'london_merged.csv')
    df["cnt_categories"] = df["cnt"].apply(lambda x: 0 if x < 450 else (1 if x < 1400 else 2))
    df["hour"] = df["timestamp"].apply(lambda x: int(x[11:13]))
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["date"] = df["date"].apply(lambda x: int(x.strftime('%d%m%Y')))

    return df


def prepare_train_test(scale=True):
    df = prepare_dataset()
    dates_in_data = df['date'].unique()
    test_size = 0.25
    train_days, test_days = train_test_split(dates_in_data,
                                             test_size=test_size,
                                             random_state=57)

    test_set = df[df['date'].isin(test_days)]
    train_set = df[df['date'].isin(train_days)]

    test_set_x = test_set.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
    test_set_y = test_set.iloc[:, 10]

    train_set_x = train_set.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
    train_set_y = train_set.iloc[:, 10]

    if scale:
        scalar = StandardScaler()
        scalar.fit(train_set_x)

        scaled_train_x = scalar.transform(train_set_x)
        scaled_test_x = scalar.transform(test_set_x)
        return scaled_test_x, test_set_y, scaled_train_x, train_set_y

    return test_set_x, test_set_y, train_set_x, train_set_y


def prepare_grouped_data(scale=True):
    df = prepare_dataset()
    dates_in_data = df['date'].unique()
    test_size = 0.25
    train_days, test_days = train_test_split(dates_in_data,
                                             test_size=test_size,
                                             random_state=57)
    X_train, y_train = get_x_any_y(df, train_days)
    X_test, y_test = get_x_any_y(df, test_days)

    if scale:
        train_set = df[df['date'].isin(train_days)]
        train_set_x = train_set.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
        scalar = StandardScaler()
        scalar.fit(train_set_x)

        scaled_X_train = [scalar.transform(day) for day in X_train]
        scaled_X_test = [scalar.transform(day) for day in X_test]
        return scaled_X_train, y_train, scaled_X_test, y_test

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    # prepare_train_test()
    # _, _, _, _, _, _, train_days, test_days, df = prepare_train_test()
    # print(train_set_x[train_set_x['date'] == True].index.tolist())
    # prepare_grouped_data()
    data = prepare_dataset()
    print(data.head())
    pass
