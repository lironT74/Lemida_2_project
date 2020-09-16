import numpy
import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

BEGIN = '*B'
STOP = '*S'
CONTAINS_DIGIT = '*CD'
CONTAINS_UPPER = '*CU'
CONTAINS_HYPHEN = '*CH'


def make_matrix():
    matrix = np.zeros((3, 3, 3, 3))
    index_to_value = []
    counter = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    matrix[i, j, k, l] = counter
                    index_to_value.append(np.array([i, j, k, l]))
                    counter += 1
    return matrix, index_to_value



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



def get_x_any_y_advanced(df, weeks, y_column):
    y_matrix, _ = make_matrix()
    x, y = [], []
    for i, week in enumerate(weeks):
        # if i % 5 == 0:
        #     print(i)
        x_week_0, x_week_1, x_week_2, x_week_3, x_week_4, x_week_5 = [], [], [], [], [], []
        y_week_0, y_week_1, y_week_2, y_week_3, y_week_4, y_week_5 = [], [], [], [], [], []
        week_df = df[df['year_week'] == week]
        for week_day in sorted(week_df['week_day'].unique()):
            x_day_array = week_df[week_df['week_day'] == week_day].drop([y_column, 'year_week'], axis=1).to_numpy()
            y_day_array = week_df[week_df['week_day'] == week_day][y_column].to_numpy()
            if len(x_day_array) != 24:
                continue
            x_week_0.append(x_day_array[0:4])
            y_week_0.append(y_matrix[tuple(y_day_array[:4])])
            x_week_1.append(x_day_array[4:8])
            y_week_1.append(y_matrix[tuple(y_day_array[4:8])])

            x_week_2.append(x_day_array[8:12])
            y_week_2.append(y_matrix[tuple(y_day_array[8:12])])

            x_week_3.append(x_day_array[12:16])
            y_week_3.append(y_matrix[tuple(y_day_array[12:16])])

            x_week_4.append(x_day_array[16:20])
            y_week_4.append(y_matrix[tuple(y_day_array[16:20])])

            x_week_5.append(x_day_array[20:])
            y_week_5.append(y_matrix[tuple(y_day_array[20:])])

        x.append(np.array(x_week_0))
        x.append(np.array(x_week_1))
        x.append(np.array(x_week_2))
        x.append(np.array(x_week_3))
        x.append(np.array(x_week_4))
        x.append(np.array(x_week_5))

        y.append(np.array(y_week_0))
        y.append(np.array(y_week_1))
        y.append(np.array(y_week_2))
        y.append(np.array(y_week_3))
        y.append(np.array(y_week_4))
        y.append(np.array(y_week_5))

    return x, y


def multiply_sparse(v, f):
    """
    :param v: Dense vector
    :param f: Sparse vector
    :return: Sparse multiplication of v with f
    """
    res = 0
    for i in f:
        res += v[i]
    return res


def exp_multiply_sparse(expv, f):
    """
    :param expv: exp of dense vector v
    :param f: sparse vector
    :return: The exponent of sparse multiplication of v with f
    """
    res = 1
    for i in f:
        res *= expv[i]
    return res


def add_or_append(dictionary, item, size=1):
    """
    Add size to the key item if it is in the dictionary, otherwise appends the key to the dictionary
    """
    if item not in dictionary:
        dictionary[item] = size
    else:
        dictionary[item] += size


def parse_lower(word_tag):
    """
    :param word_tag: A string in format word_tag
    :return: A tuple (word, tag) where word is lowercase
    """
    word, tag = word_tag.split('_')
    return word.lower(), tag


def get_words_arr(line):
    """
    :param line: A string
    :return:
    """
    words_tags_arr = line.split(' ')
    if len(words_tags_arr) == 0:
        raise Exception("get_words_arr got an empty sentence.")
    if words_tags_arr[-1][-1:] == '\n':
        words_tags_arr[-1] = words_tags_arr[-1][:-1]
        # removing \n from end of line
    return words_tags_arr


def get_file_tags(file):
    """
    :param file: The path to a .wtag file
    :return: All of the tags from the file
    """
    if not file.endswith('.wtag'):
        raise Exception("Function get_file_tags can only extract tags from a file of the wtag format")

    with open(file, 'r') as file:
        labels = []
        for line in file:
            labels = labels + get_line_tags(line)
    return labels


def get_line_tags(line):
    """
    :param line: A string
    :return: List of tags from line
    """
    words_tags_arr = get_words_arr(line)
    tags = []
    for word_tag in words_tags_arr:
        if word_tag == '':
            continue
        tag = word_tag.split('_')[1]
        tags.append(tag)
    return tags


def has_digit(word):
    """
    :param word:
    :return: True if word contains a digit
    """
    for char in word:
        if char.isdigit():
            return True


def has_upper(word):
    """
    :param word:
    :return: True if word contains upper letter
    """
    return not word.islower()


def has_hyphen(word):
    """
    :param word:
    :return: True if word has hyphen
    """
    for char in word:
        if char == '-':
            return True


def sparse_to_dense(sparse_vec, dim):
    """
    :param sparse_vec: A sparse vector
    :param dim: The vector dimension
    :return: A dense version of the vector
    """
    dense_vec = np.zeros(dim)
    for entrance in sparse_vec:
        dense_vec[entrance] += 1
    return dense_vec


def get_histories_and_corresponding_tags(x, y):
    """
    :return: Two lists, one of all histories, and the other of all tags in the file
    """

    histories = []
    tags = []
    for day, day_tags in zip(x, y):
        for i in range(len(day)):
            phour = day[i-1] if i > 0 else ([BEGIN] * len(day[0]))
            chour = day[i]
            nhour = day[i+1] if i < len(day) - 1 else ([STOP] * len(day[0]))
            pptag = day_tags[i-2] if i > 1 else BEGIN
            ptag = day_tags[i-1] if i > 0 else BEGIN
            ctag = day_tags[i]

            history = (chour, pptag, ptag, phour, nhour)
            histories.append(history)
            tags.append(ctag)
    return histories, tags


def get_predictions_list(predictions):
    """
    :param predictions:
    :return: Get all tags from predictions
    """
    predicted_tags = []
    for sentence in predictions:
        for word_tag_tuple in sentence:
            predicted_tags.append(word_tag_tuple[1])
    return predicted_tags


def update_dict(index_dict, key, value, count_dict, threshold=0):
    if key not in index_dict and count_dict[key] >= threshold:
        index_dict[key] = value
        return 1
    return 0


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