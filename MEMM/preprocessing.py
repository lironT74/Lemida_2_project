from collections import OrderedDict
from auxiliary_functions import add_or_append, BEGIN, STOP, update_dict
import numpy as np


class FeatureStatisticsClass:
    def __init__(self):
        self.f100_count_dict = OrderedDict()  # Init all features dictionaries
        self.f103_count_dict = OrderedDict()  # Trigram features
        self.f104_count_dict = OrderedDict()  # Bigram features
        self.f105_count_dict = OrderedDict()  # Unigram features
        self.f106_count_dict = OrderedDict()  # Previous word + tag
        self.f107_count_dict = OrderedDict()  # Next word + tag

    def count_features(self, x, y, f100, f103, f104, f105, f106, f107):
        """
        Main function that calls count functions for all features
        """
        assert (len(x) == len(y)) and (len(x[0]) == len(y[0]))
        if f100:
            self.__count_f100(x, y)
        if f103:
            self.__count_f103(x, y)
        if f104:
            self.__count_f104(x, y)
        if f105:
            self.__count_f105(x, y)
        if f106:
            self.__count_f106(x, y)
        if f107:
            self.__count_f107(x, y)

    # TODO ignore date
    def __count_f100(self, x, y):
        for day, day_tags in zip(x, y):
            for hour, tag in zip(day, day_tags):
                for index, value in enumerate(hour):
                    add_or_append(self.f100_count_dict, (index, value, tag))

    def __count_f103(self, x, y):
        for day in y:
            pptag = BEGIN
            ptag = BEGIN
            for ctag in day:
                add_or_append(self.f103_count_dict, (pptag, ptag, ctag))
                pptag = ptag
                ptag = ctag

    def __count_f104(self, x, y):
        for day in y:
            ptag = BEGIN
            for ctag in day:
                add_or_append(self.f104_count_dict, (ptag, ctag))
                ptag = ctag

    def __count_f105(self, x, y):
        for day in y:
            for ctag in day:
                add_or_append(self.f105_count_dict, ctag)

    # current tag and previous value
    def __count_f106(self, x, y):
        for day, day_tags in zip(x, y):
            for i in range(len(day)):
                ctag = day_tags[i]
                phour = day[i-1] if i > 0 else ([BEGIN] * len(day[0]))
                for index, value in enumerate(phour):
                    add_or_append(self.f106_count_dict, (index, value, ctag))

    # current tags and next value
    def __count_f107(self, x, y):
        for day, day_tags in zip(x, y):
            for i in range(len(day)):
                tag = day_tags[i]
                phour = day[i+1] if i < len(day) - 1 else ([STOP] * len(day[0]))
                for index, value in enumerate(phour):
                    add_or_append(self.f107_count_dict, (index, value, tag))


class Feature2Id:
    def __init__(self, feature_statistics, threshold):
        """
        :param feature_statistics: statistics class, for each feature gives empirical counts
        :param threshold: feature count threshold - empirical count must be higher than this in order for a certain
        feature to be kept
        """
        self.feat_stats: FeatureStatisticsClass = feature_statistics
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this
        self.num_features = 0  # Total number of features accumulated
        # Internal feature indexing
        self.f100_counter = 0
        self.f103_counter = 0
        self.f104_counter = 0
        self.f105_counter = 0
        self.f106_counter = 0
        self.f107_counter = 0
        # Init all features dictionaries
        self.f100_index_dict = OrderedDict()
        self.f103_index_dict = OrderedDict()
        self.f104_index_dict = OrderedDict()
        self.f105_index_dict = OrderedDict()
        self.f106_index_dict = OrderedDict()
        self.f107_index_dict = OrderedDict()

    def initialize_index_dicts(self, x, y, f100, f103, f104, f105, f106, f107):
        """
        Initializes index dictionaries for features given in the list.
        :param f100: True if f100 should be initialized
        :param f103: True if f103 should be initialized
        :param f104: True if f104 should be initialized
        :param f105: True if f105 should be initialized
        :param f106: True if f105 should be initialized
        :param f107: True if f105 should be initialized
        """
        if f100:
            self.__initialize_f100_index_dict(x, y)
        if f103:
            self.__initialize_f103_index_dict(x, y)
        if f104:
            self.__initialize_f104_index_dict(x, y)
        if f105:
            self.__initialize_f105_index_dict(x, y)
        if f106:
            self.__initialize_f106_index_dict(x, y)
        if f107:
            self.__initialize_f107_index_dict(x, y)

    def get_all_tags(self):
        """
        A quick way to access all tags the model "knows", i.e. passed threshold.
        :return: List of tags
        """
        return list(self.f105_index_dict.keys())

    def get_master_index(self):
        """
        :return: A union of all of the dictionaries of the class
        """
        master_index = OrderedDict()
        master_index.update(self.f100_index_dict)
        master_index.update(self.f103_index_dict)
        master_index.update(self.f104_index_dict)
        master_index.update(self.f105_index_dict)
        master_index.update(self.f106_index_dict)
        master_index.update(self.f107_index_dict)
        return master_index

    def __initialize_f100_index_dict(self, x, y):
        for day, day_tags in zip(x, y):
            for hour, tag in zip(day, day_tags):
                for index, value in enumerate(hour):
                    key = (index, value, tag)
                    value = self.f100_counter + self.num_features
                    self.f100_counter += update_dict(self.f100_index_dict, key, value,
                                                     self.feat_stats.f100_count_dict, self.threshold)
                    # if key not in self.f100_index_dict and self.feat_stats.f100_count_dict[key] >= self.threshold:
                    #     self.f100_index_dict[key] = self.f100_counter + self.total_features
                    #     self.f100_counter += 1
        self.num_features += self.f100_counter

    def __initialize_f103_index_dict(self, x, y):
        for day_tags in y:
            pptag = BEGIN
            ptag = BEGIN
            for ctag in day_tags:
                key = (pptag, ptag, ctag)
                value = self.f103_counter + self.num_features
                self.f103_counter += update_dict(self.f103_index_dict, key, value,
                                                 self.feat_stats.f103_count_dict, self.threshold)
                # if key not in self.f103_index_dict and self.feat_stats.f103_count_dict[key] >= self.threshold:
                #     self.f103_index_dict[key] = self.f103_counter + self.total_features
                #     self.f103_counter += 1
                pptag = ptag
                ptag = ctag
        self.num_features += self.f103_counter

    def __initialize_f104_index_dict(self, x, y):
        for day_tags in y:
            ptag = BEGIN
            for ctag in day_tags:
                key = (ptag, ctag)
                value = self.f104_counter + self.num_features
                self.f104_counter += update_dict(self.f104_index_dict, key, value,
                                                 self.feat_stats.f104_count_dict, self.threshold)
                # if key not in self.f104_index_dict and self.feat_stats.f104_count_dict[key] >= self.threshold:
                #     self.f104_index_dict[key] = self.f104_counter + self.total_features
                #     self.f104_counter += 1
                ptag = ctag
        self.num_features += self.f104_counter

    def __initialize_f105_index_dict(self, x, y):
        for day_tags in y:
            for ctag in day_tags:
                key = ctag
                value = self.f105_counter + self.num_features
                self.f105_counter += update_dict(self.f105_index_dict, key, value,
                                                 self.feat_stats.f105_count_dict, self.threshold)
                # if key not in self.f105_index_dict and self.feat_stats.f105_count_dict[key] >= self.threshold:
                #     self.f105_index_dict[key] = self.f105_counter + self.total_features
                #     self.f105_counter += 1
        self.num_features += self.f105_counter
    
    def __initialize_f106_index_dict(self, x, y):
        for day, day_tags in zip(x, y):
            for i in range(len(day)):
                phour = day[i-1] if i > 0 else ([BEGIN] * len(day[0]))
                ctag = day_tags[i]
                for index, value in enumerate(phour):
                    key = (index, value, ctag)
                    value = self.f106_counter + self.num_features
                    self.f106_counter += update_dict(self.f106_index_dict, key, value,
                                                     self.feat_stats.f106_count_dict, self.threshold)
                    # if key not in self.f106_index_dict and self.feat_stats.f106_count_dict[key] >= self.threshold:
                    #     self.f106_index_dict[key] = self.f106_counter + self.total_features
                    #     self.f106_counter += 1
        self.num_features += self.f106_counter
    
    def __initialize_f107_index_dict(self, x, y):
        for day, day_tags in zip(x, y):
            for i in range(len(day)):
                nhour = day[i+1] if i < len(day) - 1 else ([STOP] * len(day[0]))
                ctag = day_tags[i]
                for index, value in enumerate(nhour):
                    key = (index, value, ctag)
                    value = self.f107_counter + self.num_features
                    self.f107_counter += update_dict(self.f107_index_dict, key, value,
                                                     self.feat_stats.f107_count_dict, self.threshold)
        self.num_features += self.f107_counter

    def sparse_feature_representation(self, history, ctag):
        """
        :param history: A tuple of the following format (chour, pptag, ptag, phour, nhour)
        :param ctag: The tag corresponding to the current word
        :return: A sparse feature representation of the history+ctag
        """
        phour = history[3]
        chour = history[0]
        nhour = history[4]

        pptag, ptag = history[1], history[2]
        features = []

        for index, value in enumerate(chour):
            if (index, value, ctag) in self.f100_index_dict:
                features.append(self.f100_index_dict[(index, value, ctag)])

        if (pptag, ptag, ctag) in self.f103_index_dict:
            features.append(self.f103_index_dict[(pptag, ptag, ctag)])

        if (ptag, ctag) in self.f104_index_dict:
            features.append(self.f104_index_dict[(ptag, ctag)])

        if ctag in self.f105_index_dict:
            features.append(self.f105_index_dict[ctag])

        for index, value in enumerate(phour):
            if (index, value, ctag) in self.f106_index_dict:
                features.append(self.f106_index_dict[(index, value, ctag)])

        for index, value in enumerate(nhour):
            if (index, value, ctag) in self.f107_index_dict:
                features.append(self.f107_index_dict[(index, value, ctag)])

        return np.array(features)

    def dense_feature_representation(self, history, ctag):
        """
        :param history: A tuple of the following format (chour, pptag, ptag, phour, nhour)
        :param ctag: The tag corresponding to the current word
        :return: A Dense feature representation of the history+ctag
        """
        phour, chour, nhour = history[3], history[0], history[4]
        pptag, ptag = history[1], history[2]
        features = np.zeros(self.num_features)

        for index, value in enumerate(chour):
            if (index, value, ctag) in self.f100_index_dict:
                features[self.f100_index_dict[(index, value, ctag)]] += 1

        if (pptag, ptag, ctag) in self.f103_index_dict:
            features[self.f103_index_dict[(pptag, ptag, ctag)]] += 1

        if (ptag, ctag) in self.f104_index_dict:
            features[self.f104_index_dict[(ptag, ctag)]] += 1

        if ctag in self.f105_index_dict:
            features[self.f105_index_dict[ctag]] += 1

        for index, value in enumerate(phour):
            if (index, value, ctag) in self.f100_index_dict:
                features[self.f100_index_dict[(index, value, ctag)]] += 1

        for index, value in enumerate(nhour):
            if (index, value, ctag) in self.f100_index_dict:
                features[self.f100_index_dict[(index, value, ctag)]] += 1

        return features

    def build_features_list(self, histories_list, corresponding_tags_list):
        """
        :param histories_list: All histories in the data
        :param corresponding_tags_list: The corresponding tags of the aforementioned tags
        :return: a list of the feature representation for any history and tag that shows up in the train data
        """
        row_dim = len(histories_list)
        res = [self.sparse_feature_representation(histories_list[i], corresponding_tags_list[i])
               for i in range(row_dim)]
        return res

    def build_features_matrix(self, all_histories_list, all_tags_list):
        """
        :param all_histories_list: All histories in the tags
        :param all_tags_list: All of the tags that show up in the data
        :return: A matrix of the feature representation of any possible combination of history from the data and
        tag from the data, where cell i, j is the i-th history and the j-th cell.
        """
        row_dim = len(all_histories_list)
        col_dim = len(all_tags_list)
        feature_matrix = [[self.sparse_feature_representation(all_histories_list[i], all_tags_list[j])
                           for j in range(col_dim)] for i in range(row_dim)]
        return feature_matrix

