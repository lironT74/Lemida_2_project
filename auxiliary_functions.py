import numpy
import numpy as np
import itertools
import scipy.sparse as sp
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

BEGIN = '*B'
STOP = '*S'
CONTAINS_DIGIT = '*CD'
CONTAINS_UPPER = '*CU'
CONTAINS_HYPHEN = '*CH'



def get_x_any_y(df, dates, y_column):
    x, y = [], []
    for date in dates:
        day_df = df[df['date'] == date]
        x.append(day_df.drop([y_column, 'date', 'year'], axis=1).to_numpy())
        y.append(day_df[y_column].to_numpy())
    return x, y


def get_x_any_y_years(df, years, y_column):
    x, y = [], []
    for year in years:
        years_df = df[df['year'] == year]
        x.append(years_df.drop([y_column, 'date', 'year'], axis=1).to_numpy())
        y.append(years_df[y_column].to_numpy())
    return x, y


def index_to_value(x, num_of_hours):

    values = []

    while x > 0:
        values.append(x % 3)
        x //= 3

    while len(values) < num_of_hours:
        values.append(0)

    values.reverse()

    return np.array(values)


def value_to_index(lst):

    res = 0
    for index, value in enumerate(reversed(lst)):
        res += value * 3 ** index

    return res


def get_x_any_y_advanced_creative(df, weeks, y_column, k):

    x, y = {}, {}

    assert 24 % k == 0

    for week_index, week in enumerate(weeks):

        x[week_index] = []
        y[week_index] = []

        x_week = {i: [] for i in range(24//k)}
        y_week = {i: [] for i in range(24//k)}

        week_df = df[df['year_week'] == week]
        for week_day in sorted(week_df['week_day'].unique()):

            x_day_array = week_df[week_df['week_day'] == week_day].drop([y_column, 'year_week'], axis=1).to_numpy()

            y_hours_labels = {week_df[week_df['week_day'] == week_day]['hour'].iloc[i]: week_df[week_df['week_day'] == week_day][y_column].iloc[0] for i in range(len(week_df[week_df['week_day'] == week_day].index))}
            x_hours_vectors = {week_df[week_df['week_day'] == week_day]['hour'].iloc[i]: week_df[week_df['week_day'] == week_day].drop([y_column, 'year_week'], axis=1).iloc[0].to_numpy() for i in range(len(week_df[week_df['week_day'] == week_day].index))}


            hour_index = list(week_df[week_df['week_day'] == week_day].drop([y_column, 'year_week'], axis=1).columns).index('hour')

            # No missing hours:
            if len(x_day_array) == 24:
                for block_index in range(24 // k):
                    hours_vectors = [x_hours_vectors[hour] for hour in range(k * block_index, k * (block_index + 1), 1)]
                    x_week[block_index].append(list(itertools.chain(*hours_vectors)))

                    y_week[block_index].append(value_to_index([y_hours_labels[hour] for hour in range(k * block_index, k * (block_index + 1), 1)]))

            # missing hours. should discard windows with missing hours.
            else:
                missing_hours = [hour for hour in range(24) if hour not in x_day_array[:, hour_index]]
                missing_blocks = set()

                for block_index in range(24 // k):
                    for block_hour in range(k * block_index, k * (block_index + 1), 1):
                        if block_hour in missing_hours:
                            missing_blocks.add(block_index)

                for block_index in range(24 // k):
                    if block_index not in missing_blocks:

                        hours_vectors = [x_hours_vectors[hour] for hour in range(k * block_index, k * (block_index + 1), 1)]
                        x_week[block_index].append(list(itertools.chain(*hours_vectors)))

                        y_week[block_index].append(value_to_index([
                            y_hours_labels[hour] for hour in range(k * block_index, k * (block_index + 1), 1)]))


        for block_index in range(24 // k):
            x[week_index].append(x_week[block_index])
            y[week_index].append(np.array(y_week[block_index]))


    return x, y



def create_month_dict(data):
    months = []
    for x in data:
        month = x[0, -2]
        if month not in months:
            months.append(month)
    month_dict = {val: i for i, val in enumerate(sorted(months))}
    return month_dict


def draw_confusion_matrix(confusion_matrix, xtick_labels=None, ytick_labels=None, title=None, fontsize=10, save=False,
                          show=True):
    fig, ax = plt.subplots()

    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=np.max(confusion_matrix)))
    im = ax.imshow(confusion_matrix, cmap='jet', norm=plt.Normalize(vmin=0, vmax=np.max(confusion_matrix)))
    im = ax.imshow(confusion_matrix, cmap='jet')

    divider1 = make_axes_locatable(ax)
    cax = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(sm, ax=ax, cax=cax).ax.tick_params(labelsize=fontsize)

    ax.set_xticks(np.arange(confusion_matrix.shape[0]))
    ax.set_yticks(np.arange(confusion_matrix.shape[1]))

    if xtick_labels:
        ax.set_xticklabels(xtick_labels, fontsize=fontsize)
    if ytick_labels:
        ax.set_yticklabels(ytick_labels, fontsize=fontsize)

    ax.set_xlabel("predictions", fontsize=fontsize)
    ax.set_ylabel("labels", fontsize=fontsize)

    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", color='w', fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize + 4)

    if save:
        plt.savefig(f"./cms/{title}.png")
    if show:
        plt.show()