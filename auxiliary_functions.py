import numpy as np
import scipy.sparse as sp

BEGIN = '*B'
STOP = '*S'
CONTAINS_DIGIT = '*CD'
CONTAINS_UPPER = '*CU'
CONTAINS_HYPHEN = '*CH'


def get_x_any_y(df, dates, x_columns, y_column):
    x, y = [], []
    for date in dates:
        day_df = df[df['date'] == date]
        x.append(day_df[x_columns].to_numpy())
        y.append(day_df[y_column].to_numpy())
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
