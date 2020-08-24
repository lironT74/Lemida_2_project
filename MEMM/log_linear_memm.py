import pickle
import numpy as np
from os import remove
from time import strftime, time
from scipy.optimize import fmin_l_bfgs_b
from MEMM.inference import memm_viterbi
from MEMM.preprocessing import FeatureStatisticsClass, Feature2Id
from auxiliary_functions import get_histories_and_corresponding_tags, get_file_tags, get_predictions_list
from MEMM.optimization import calc_empirical_counts, calc_objective, calc_gradient
from numpy.linalg import norm
from os import path, makedirs


class Log_Linear_MEMM:
    def __init__(self, threshold=10, lam=0, maxiter=200,
                 f100=True, f103=True, f104=True, f105=True, f106=True, f107=True):
        self.feature_statistics = None
        self.feature2id = None
        self.weights = None
        self.lbfgs_result = None
        self.dim = None
        self.threshold = threshold
        self.fit_time = None
        self.lam = lam
        self.maxiter = maxiter
        self.iter = None  # will be set AFTER optimization finished and holds the number of iterations de-facto
        self.f100 = f100
        self.f103 = f103
        self.f104 = f104
        self.f105 = f105
        self.f106 = f106
        self.f107 = f107

    def __sub__(self, other):
        """
        Function that compares two models
        :param other: Other model
        :return: The cosine of the angle between the two models' weight vectors
        """
        return self.weights @ other.weights / (norm(self.weights) * norm(other.weights))

    def fit(self, x, y, i_print=20):
        """
        A simple interface to train a model.
        :param i_print: A parameter of fmin_l_bfgs_b, effects how often it prints
        :return self
        """
        start_time = time()
        self.__preprocess(x, y)
        self.__optimize(x, y, i_print)
        self.fit_time = time() - start_time
        return self

    def __preprocess(self, x, y):
        """
        Preprocesses the data, counts the different features and gives them an index
        :return:None
        """
        self.feature_statistics = FeatureStatisticsClass()
        self.feature_statistics.count_features(x, y, self.f100, self.f103, self.f104, self.f105, self.f106, self.f107)
        self.feature2id = Feature2Id(self.feature_statistics, self.threshold)
        self.feature2id.initialize_index_dicts(x, y, self.f100, self.f103, self.f104, self.f105, self.f106, self.f107)
        self.dim = self.feature2id.num_features

    def __optimize(self, x, y, iprint=20):
        """
        Learns the weights of the different parameters using the function fmin_l_bfgs_b
        :param iprint: How often should fmin_l_bfgs_b print
        :return: None
        """
        # initializing parameters for fmin_l_bfgs_b
        all_tags_list = self.feature2id.get_all_tags()
        all_histories, all_corresponding_tags = get_histories_and_corresponding_tags(x, y)
        features_list = self.feature2id.build_features_list(all_histories, all_corresponding_tags)
        features_matrix = self.feature2id.build_features_matrix(all_histories, all_tags_list)
        empirical_counts = calc_empirical_counts(features_list, self.dim)
        args = (self.dim, features_list, features_matrix, empirical_counts, self.lam)
        w_0 = np.random.random(self.dim)
        optimal_params = fmin_l_bfgs_b(func=calc_objective, fprime=calc_gradient, x0=w_0, args=args,
                                       maxiter=self.maxiter, iprint=iprint)
        self.lbfgs_result = optimal_params
        self.weights = optimal_params[0]
        self.iter = optimal_params[2]['nit']

    def save(self, filename='model_' + strftime("%Y-%m-%d_%H-%M-%S")):
        """
        Saves a trained model as pkl file and a txt file that describes it
        :param filename: The name of the file
        """
        pkl_path = 'dumps/' + filename + '.pkl'
        if not path.exists('dumps'):
            makedirs('dumps')
        with open(pkl_path, 'wb') as f:
            pickle.dump(self, f)

    def predict(self, x, beam_size=2):
        """
        Generates a prediction for a given input. Input can be either a sentence (string) or a file path.
        File can be in either .wtag or .words format.
        :param beam_size: a parameter of the viterbi
        :return: A "predictions" matrix with a tuple (word, pred) in the [i][j] cell, where i is the number of the line
        in the file and j is the number of the word in the line.
        """
        return [np.array(memm_viterbi(self.feature2id, self.weights, day, beam_size)) for day in x]


    @staticmethod
    def accuracy(test_path, predictions):
        """
        Calculates the accuracy of predictions given the data they've predicted
        :param test_path: The data that was predicted
        :param predictions: A predictions matrix
        :return: The accuracy of the preditctions
        """
        # getting tags
        predicted_tags = get_predictions_list(predictions)
        true_tags = get_file_tags(test_path)

        if len(predicted_tags) != len(true_tags):
            raise Exception('predicted tags and true tags dont have the same dimension')

        # calculating accuracy
        total_predictions = len(true_tags)
        correct = 0
        for true, pred in zip(true_tags, predicted_tags):
            if true == pred:
                correct += 1
        return correct / total_predictions


    @staticmethod
    def load_model(filepath):
        """
        Loads trained model from pickle file
        :param filepath: the path of the pickle file
        :return: Log_Linear_Memm object
        """
        if not filepath.endswith('.pkl'):
            raise Exception('Model can only be loaded from a file that ends with .pkl')
        with open(filepath, 'rb') as f:
            return pickle.load(f)
