import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from os import remove
from time import strftime, time
from scipy.optimize import fmin_l_bfgs_b
from inference import memm_viterbi
from preprocessing import FeatureStatisticsClass, Feature2Id
from auxiliary_functions import get_all_histories_and_corresponding_tags, get_file_tags, \
    get_predictions_list, clean_tags
from optimization import calc_empirical_counts, calc_objective, calc_gradient
from numpy.linalg import norm


sns.set()


class Log_Linear_MEMM:
    def __init__(self, threshold=10, fix_threshold=-1, lam=0, maxiter=200, fix_weights=(1, 1, 1, 1),
                 f100=True, f101=True, f102=True, f103=True, f104=True, f105=True,
                 f106=True, f107=True, f108=True, f109=True, f110=True):
        self.train_path = None
        self.feature_statistics = None
        self.feature2id = None
        self.weights = None
        self.lbfgs_result = None
        self.dim = None
        self.threshold = threshold
        self.fix_threshold = fix_threshold if fix_threshold >= 0 else threshold
        self.fix_weights = fix_weights
        self.fit_time = None
        self.lam = lam
        self.maxiter = maxiter
        self.iter = None  # will be set AFTER optimization finished and holds the number of iterations de-facto
        self.f100 = f100
        self.f101 = f101
        self.f102 = f102
        self.f103 = f103
        self.f104 = f104
        self.f105 = f105
        self.f106 = f106
        self.f107 = f107
        self.f108 = f108
        self.f109 = f109
        self.f110 = f110

    def __sub__(self, other):
        """
        Function that compares two models
        :param other: Other model
        :return: The cosine of the angle between the two models' weight vectors
        """
        return self.weights @ other.weights / (norm(self.weights) * norm(other.weights))

    def fit(self, train_path, iprint=20):
        """
        A simple interface to train a model.
        :param train_path: A path for training data, *.wtag format.
        :param iprint: A parameter of fmin_l_bfgs_b, effects how often it prints
        :return self
        """
        start_time = time()
        self.train_path = train_path
        self.__preprocess()
        self.__optimize(iprint)
        self.fit_time = time() - start_time
        return self

    def __preprocess(self):
        """
        Preprocesses the data, counts the different features and gives them an index
        :return:None
        """
        self.feature_statistics = FeatureStatisticsClass(self.train_path, self.fix_weights)
        self.feature_statistics.count_features(self.f100, self.f101, self.f102, self.f103, self.f104, self.f105,
                                               self.f106, self.f107, self.f108, self.f109, self.f110)
        self.feature2id = Feature2Id(self.train_path, self.feature_statistics, self.threshold, self.fix_threshold)
        self.feature2id.initialize_index_dicts(self.f100, self.f101, self.f102, self.f103, self.f104, self.f105,
                                               self.f106, self.f107, self.f108, self.f109, self.f110)
        self.dim = self.feature2id.total_features

    def __optimize(self, iprint=20):
        """
        Learns the weights of the different parameters using the function fmin_l_bfgs_b
        :param iprint: How often should fmin_l_bfgs_b print
        :return: None
        """
        # initializing parameters for fmin_l_bfgs_b
        all_tags_list = self.feature2id.get_all_tags()
        all_histories, all_corresponding_tags = get_all_histories_and_corresponding_tags(self.train_path)
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
        txt_path = 'dumps/' + filename + '.txt'
        with open(pkl_path, 'wb') as f:
            pickle.dump(self, f)
        with open(txt_path, 'w') as f:
            f.write('model name: ' + filename + '\n\n\n')
            f.write('train_path = ' + str(self.train_path) + '\n')
            f.write('dim = ' + str(self.dim) + '\n')
            f.write('threshold = ' + str(self.threshold) + '\n')
            f.write('fix_threshold = ' + str(self.fix_threshold) +
                    (' has no meaning' if not self.f101 and not self.f102 else '') + '\n')
            f.write('fix_weights = ' + str(self.fix_weights) + '\n')
            f.write('fit time = ' + str(round(self.fit_time, 2)) + ' sec\n')
            f.write('lam = ' + str(self.lam) + '\n')
            f.write('maxiter = ' + str(self.maxiter) + '\n')
            f.write('iter = ' + str(self.iter) + '\n')
            f.write('f100 = ' + str(self.f100) + '\n')
            f.write('f101 = ' + str(self.f101) + '\n')
            f.write('f102 = ' + str(self.f102) + '\n')
            f.write('f103 = ' + str(self.f103) + '\n')
            f.write('f104 = ' + str(self.f104) + '\n')
            f.write('f105 = ' + str(self.f105) + '\n')
            f.write('f106 = ' + str(self.f106) + '\n')
            f.write('f107 = ' + str(self.f107) + '\n')
            f.write('f108 = ' + str(self.f108) + '\n')
            f.write('f109 = ' + str(self.f109) + '\n')
            f.write('f110 = ' + str(self.f110) + '\n')

    def predict(self, input_data, beam_size=2):
        """
        Generates a prediction for a given input. Input can be either a sentence (string) or a file path.
        File can be in either .wtag or .words format.
        :param beam_size: a parameter of the viterbi
        :param input_data: string or file path
        :return: A "predictions" matrix with a tuple (word, pred) in the [i][j] cell, where i is the number of the line
        in the file and j is the number of the word in the line.
        """
        if len(input_data) > 6 and input_data[-6:] == '.words':
            return self.__predict_file(input_data, beam_size)

        if len(input_data) > 5 and input_data[-5:] == '.wtag':
            temp_file = r'data\temp' + str(round(time())) + '.words'  # using time() to allow parallel runs
            clean_tags(input_data, temp_file)
            predictions = self.__predict_file(temp_file, beam_size)
            remove(temp_file)
            return predictions

        return memm_viterbi(self.feature2id, self.weights, input_data, beam_size)

    def __predict_file(self, file, beam_size):
        """
        Predicts a .words file, as in a file without any tags
        :param file: The location of the file
        :param beam_size: Viterbi algorithm hyper parameter
        :return: The predictions of the file
        """
        with open(file, 'r') as in_file:
            predictions = []
            for line in in_file:
                line_predictions = []
                words = line.split()
                prediction = memm_viterbi(self.feature2id, self.weights, line, beam_size)
                for word, pred in zip(words, prediction):
                    line_predictions.append((word, pred))
                predictions.append(line_predictions)
        return predictions

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
    def confusion_matrix(test_path, predictions, errors_to_display=10, show=True, order='freq', slice_on_pred=False):
        """
        :param test_path: Path to a *.wtag file with some ground truth
        :param predictions: Predictions matrix that Log_Linear_MEMM.predict() returned on the same test_path
        :param errors_to_display: Number of errors to display in CM, as required in the instructions
        :param show: Whether or not to show the CM at the end
        :param order: 'freq' for frequent errors left and up, 'lexi' for lexicographic order of tags
        :param slice_on_pred: The axis to slice errors on (can be either True to slice on true predictions axis or
                              False to slice on predicted labels axis)
        :return: DataFrame with the CM values, axes, rows and cols are labeled
        """
        # getting tags
        true_tags = get_file_tags(test_path)
        predicted_tags = get_predictions_list(predictions)
        all_possible_tags = set(true_tags).union(set(predicted_tags))

        # creating "raw" confusion matrix
        n = len(all_possible_tags)
        cm = pd.DataFrame(np.zeros((n, n)), columns=all_possible_tags, index=all_possible_tags)
        for true, pred in zip(true_tags, predicted_tags):
            cm.loc[true][pred] += 1

        # renaming axes names
        cm = cm.rename_axis('predicted label', axis='columns')
        cm = cm.rename_axis('true label', axis='rows')

        # if the user requested to slice on true labels axis, than we need to transpose the raw CM
        # and the logic applies with no change at all. It is set to check a negated if statement because original logic
        # was designed to make sense for slicing on prediction axis.
        if not slice_on_pred:
            cm = cm.transpose()

        # slicing rows/cols of confusion matrix to fit it to the exercise requirements
        # finding most common error
        total_tag_predictions = cm.sum(axis=0)
        correct_tag_predictions = cm.values.diagonal()
        tag_errors = total_tag_predictions - correct_tag_predictions
        tag_errors_sorted = tag_errors.sort_values(ascending=False)
        # slicing matrix
        top_errors = list(tag_errors_sorted.index.values)[:errors_to_display]
        cm = cm[top_errors]

        # reordering rows and cols by request
        if order == 'freq':
            cm = cm.reindex(top_errors, axis=1)
            cm = cm.reindex(tag_errors_sorted.index.values, axis=0)

        if order == 'lexi':
            cm = cm.reindex(sorted(list(cm.columns)), axis=1)
            cm = cm.reindex(sorted(list(cm.index.values)), axis=0)

        # plotting
        if show:
            fig = plt.gcf()
            fig.set_size_inches(8, 12)
            ax = sns.heatmap(cm, annot=True, cmap='Blues')
            plt.show()
        return cm

    @staticmethod
    def create_predictions_file(predictions,
                                file_name=r'predictions/prediction_' + strftime("%Y-%m-%d_%H-%M-%S") + '.wtag'):
        """
        Creates a file of predictions
        :param predictions: model predictions
        :param file_name: name of the file
        :return: None
        """
        with open(file=file_name, mode='w') as predictions_file:
            predictions_lines = (' '.join([word + '_' + tag for word, tag in sentence_prediction])
                                 for sentence_prediction in predictions)
            predictions_file.write('\n'.join(predictions_lines))

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
