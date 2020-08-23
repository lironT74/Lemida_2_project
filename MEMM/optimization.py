import numpy as np
from math import log
from numpy.linalg import norm
from auxiliary_functions import multiply_sparse, exp_multiply_sparse, sparse_to_dense


def calc_empirical_counts(features_list, dim):
    empirical_counts = np.zeros(dim)
    for feature in features_list:
        empirical_counts += sparse_to_dense(feature, dim)
    return empirical_counts


def calc_linear_term(v_i, features_list):
    linear_term = 0
    for feature in features_list:
        linear_term += multiply_sparse(v_i, feature)
    return linear_term


def calc_normalization_term(v_i, features_matrix):
    normalization_term = 0
    exp_v_i = np.exp(v_i)
    # We calculate the exponent of the weight vector in order to minimize the number of time we to calculate the
    # exponent of a certain number
    for history in features_matrix:
        tmp = 0
        for feature in history:
            tmp += exp_multiply_sparse(exp_v_i, feature)
        normalization_term += log(tmp)  # natural logarithm
    return normalization_term


def calc_regularization(v_i, reg_lambda):
    return 0.5 * reg_lambda * (norm(v_i) ** 2)


def calc_expected_counts(v_i, dim, features_matrix):
    expected_counts = np.zeros(dim)
    exp_v_i = np.exp(v_i)
    # We calculat the exponent of the weight vector in order to minimize the number of time we to calculate the
    # exponent of a certain number
    for history in features_matrix:
        denominator = 0
        numerator = np.zeros(dim)
        for feature in history:
            temp = exp_multiply_sparse(exp_v_i, feature)
            denominator += temp
            for f in feature:
                numerator[f] += temp
        expected_counts += numerator / denominator
    return expected_counts


def calc_objective(v_i, dim, features_list, features_matrix, empirical_counts, reg_lambda):
    linear_term = calc_linear_term(v_i, features_list)
    normalization_term = calc_normalization_term(v_i, features_matrix)

    regularization = calc_regularization(v_i, reg_lambda)

    likelihood = linear_term - normalization_term - regularization
    return -1 * likelihood


def calc_gradient(v_i, dim, features_list, features_matrix, empirical_counts, reg_lambda):
    expected_counts = calc_expected_counts(v_i, dim, features_matrix)

    regularization_grad = reg_lambda * v_i

    gradient = empirical_counts - expected_counts - regularization_grad
    return (-1) * gradient