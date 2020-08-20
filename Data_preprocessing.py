from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def show_hist():
    df = pd.read_csv(r'london_merged.csv')
    hist = df["cnt"].hist(figsize=(5,5), grid=False, bins=100)
    buckets = 3
    colors = ["red", "orange", "green"]
    for i in range(1, buckets+1):
        print(df["cnt"].quantile(q=i*1/buckets))
        plt.vlines(x=df["cnt"].quantile(q=i/buckets), ymin=0, ymax=1750, colors=colors[i-1], label=f"{np.round(i/buckets, 2)} quantile")

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

def prepare_train_test():

    df = prepare_dataset()
    dates_in_data = df['date'].unique()
    test_size = 0.25
    train_days, test_days = train_test_split(dates_in_data,
                                             test_size=test_size,
                                             random_state=57)

    test_set = df[df['date'].isin(test_days)]
    train_set = df[df['date'].isin(train_days)]

    test_set_x = test_set.iloc[:,[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    test_set_y = test_set.iloc[:,1]

    train_set_x = train_set.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    train_set_y = train_set.iloc[:, 1]

    return test_set_x, test_set_y, train_set_x, train_set_y


if __name__ == '__main__':
    # show_hist()
    print(prepare_train_test()[0].dtypes)
