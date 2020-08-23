import numpy as np

from math import exp
from auxiliary_functions import BEGIN, STOP, get_words_arr, exp_multiply_sparse


def calc_q(feature_ids, weights, history, ctag, denominator):
    """
    :param feature_ids: Feature2Id object
    :param weights: The weights of the Log_Linear_Memm
    :param history: Current history, presented in the following format: (cword, pptag, ptag, pword, nword)
    :param ctag: Current tag
    :param denominator: The denominator for the given history
    :return: q for a given history and tag
    """
    feature_vec = feature_ids.dense_feature_representation(history, ctag)
    numerator = exp(feature_vec @ weights)
    return numerator / denominator


def calc_q_denominator(feature_ids, weights, all_tags, history):
    """
    :param feature_ids: Feature2Id object
    :param weights: The weight of the Log_Linear_Memm
    :param all_tags: All of the tags featured in the learning data
    :param history: A certain history, presented in the following format: (cword, pptag, ptag, pword, nword)
    :return: The denominator of function calc_q for a certain history
    """
    denominator = 0
    for tag in all_tags:
        denominator += exp(weights @ feature_ids.dense_feature_representation(history, tag))

    return denominator


def memm_viterbi(feature_ids, weights, sentence, beam_size):
    """
    Viterbi prediction function based on lectures
    :param feature_ids:
    :param weights:
    :param sentence:
    :param beam_size:
    :return:
    """
    all_tags = feature_ids.get_all_tags()
    words_arr = [BEGIN] + get_words_arr(sentence) + [STOP]
    # We offset the size of the list to match the mathematical algorithm
    n = len(words_arr) - 2

    pi = [{} for i in range(n + 1)]
    bp = [{} for i in range(n + 1)]
    pi[0][(BEGIN, BEGIN)] = 1

    tags_dict = {-1: [BEGIN], 0: [BEGIN]}
    if beam_size == 0:
        tags_dict.update(dict.fromkeys([i for i in range(1, n+1)], all_tags))

    cword, nword = words_arr[0], words_arr[1]

    for k in range(1, n + 1):
        pword = cword
        cword = nword
        nword = words_arr[k + 1]

        pi[k] = dict.fromkeys([(u, v) for u in tags_dict[k-1] for v in all_tags], 0)
        for u in tags_dict[k-1]:
            for t in tags_dict[k-2]:
                if pi[k-1][t, u] == 0:
                    continue
                history = (cword, t, u, pword, nword)
                q_denominator = calc_q_denominator(feature_ids, weights, all_tags, history)
                for v in all_tags:
                    q = calc_q(feature_ids, weights, history, v, q_denominator)
                    if pi[k-1][t, u] * q > pi[k][u, v]:
                        pi[k][u, v] = pi[k-1][t, u] * q
                        bp[k][u, v] = t

        if beam_size == 0:
            continue
        beam_list = []
        for v in all_tags:
            v_probability = 0
            for u in tags_dict[k-1]:
                v_probability += pi[k][u, v]
            beam_list.append((v, v_probability))
        beam_list.sort(reverse=True, key=lambda item: item[1])
        tags_dict[k] = [beam_list[i][0] for i in range(beam_size)]

    tag_sequence = [None for i in range(n + 1)]
    max_prob = 0
    for u, v in pi[n].keys():
        if pi[n][(u, v)] > max_prob:
            max_prob = pi[n][(u, v)]
            tag_sequence[n - 1], tag_sequence[n] = u, v

    for k in range(n - 2, 0, -1):
        tag_sequence[k] = bp[k + 2][(tag_sequence[k + 1], tag_sequence[k + 2])]

    return tag_sequence[1:]
