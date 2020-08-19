from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    df["date"] = df["timestamp"].apply(lambda x: x[:10])
    df["hour"] = df["timestamp"].apply(lambda x: int(x[11:13]))

    return df

def prepare_train_test_validation():

    df = prepare_dataset()
    dates_in_data = df['date'].unique()
    test_size = 0.25
    train_days, test_days = train_test_split(dates_in_data,
                                             test_size=test_size,
                                             random_state=57)

    test_set = df[df['date'].isin(test_days)]
    train_set = df[df['date'].isin(train_days)]

    return train_set, test_set

if __name__ == '__main__':
    show_hist()
    df = prepare_dataset()
    print(df)
    print(prepare_train_test_validation())
