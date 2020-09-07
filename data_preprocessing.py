from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from auxiliary_functions import get_x_any_y

X_COLUMNS = ['t1', 't2', 'hum', 'wind_speed', 'weather_code', 'is_holiday', 'is_weekend', 'season', 'hour', 'month',
             'day']
Y_COLUMN = 'cnt_categories'


def show_hist():
    df = pd.read_csv(r'london_merged.csv')
    df["cnt"].hist(figsize=(5, 5), grid=False, bins=100)
    buckets = 3
    colors = ["red", "orange", "green"]
    for i in range(1, buckets + 1):
        print(df["cnt"].quantile(q=i / buckets))
        plt.vlines(x=df["cnt"].quantile(q=i / buckets), ymin=0, ymax=1750, colors=colors[i - 1],
                   label=f"{int(df['cnt'].quantile(q=i / buckets))} - {round(i / buckets, 2)} quantile")

    plt.title("histogram of bicycles count")
    plt.legend()
    plt.xlabel("bike counts")
    plt.show()


def prepare_dataset():
    # TODO consider using the function pd.set_index to set the column date as the index of the df
    df = pd.read_csv(r'london_merged.csv')
    df["cnt_categories"] = df["cnt"].apply(lambda x: 0 if x < 450 else (1 if x < 1400 else 2))
    df["hour"] = df["timestamp"].apply(lambda x: int(x[11:13]))

    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date
    df["date"] = df['timestamp'].apply(lambda x: int(x.strftime('%d%m%Y')))
    df["month"] = df['timestamp'].apply(lambda x: int(x.strftime('%m')))
    df["day"] = df['timestamp'].apply(lambda x: int(x.strftime('%d')))

    df.drop(['timestamp', 'cnt'], axis=1, inplace=True)
    return df


def prepare_categorized_dataset():
    df = prepare_dataset()
    for col in df:
        if col == 'date':
            continue
        if len(df[col].unique()) > 24:
            lower_barrier = df[col].quantile(q=1 / 3)
            higher_barrier = df[col].quantile(q=2 / 3)
            df[f'{col}_categorized'] = df[col].\
                apply(lambda x: 'low' if x < lower_barrier else 'medium' if x < higher_barrier else 'high')
            df.drop(col, axis=1, inplace=True)
    return df


def prepare_train_test(categorized=False, scale=True):
    df = prepare_categorized_dataset() if categorized else prepare_dataset()
    dates_in_data = df['date'].unique()
    test_size = 0.25
    train_days, test_days = train_test_split(dates_in_data,
                                             test_size=test_size,
                                             random_state=57)

    test_set = df[df['date'].isin(test_days)]
    train_set = df[df['date'].isin(train_days)]

    test_x = test_set.drop([Y_COLUMN, 'date'], axis=1)
    test_y = test_set[Y_COLUMN]

    train_x = train_set.drop([Y_COLUMN, 'date'], axis=1)
    train_y = train_set[Y_COLUMN]

    if scale:
        scalar = StandardScaler()
        scalar.fit(train_x)

        scaled_train_x = scalar.transform(train_x)
        scaled_test_x = scalar.transform(test_x)
        return scaled_test_x, test_y, scaled_train_x, train_y

    return test_x, test_y, train_x, train_y


def prepare_grouped_data(categorized=False, scale=True):
    df = prepare_categorized_dataset() if categorized else prepare_dataset()
    dates_in_data = df['date'].unique()
    test_size = 0.25
    train_days, test_days = train_test_split(dates_in_data,
                                             test_size=test_size,
                                             random_state=57)
    train_x, train_y = get_x_any_y(df, train_days, Y_COLUMN)
    test_x, test_y = get_x_any_y(df, test_days, Y_COLUMN)

    if scale:
        train_set = df[df['date'].isin(train_days)]
        train_set_x = train_set.drop([Y_COLUMN, 'date'], axis=1)
        scalar = StandardScaler()
        scalar.fit(train_set_x)

        scaled_train_x = [scalar.transform(day) for day in train_x]
        scaled_test_x = [scalar.transform(day) for day in test_x]
        return scaled_train_x, train_y, scaled_test_x, test_y

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    pass
