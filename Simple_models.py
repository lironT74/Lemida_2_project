from collections import Counter
import Data_preprocessing as dp
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC


def calc_acc(predictions, real):
    return np.average([pred==real for pred, real in zip(predictions, real)])

def one_vs_all():
    test_set_x, scaled_test_x, test_set_y, train_set_x, scaled_train_x, train_set_y = dp.prepare_train_test()

    predictions = OneVsRestClassifier(LogisticRegression(random_state=0, max_iter=5000)).fit(scaled_train_x, train_set_y).predict(scaled_test_x)
    acc = calc_acc(predictions, test_set_y)
    print(acc)

def logistic_regression():
    test_set_x, scaled_test_x, test_set_y, train_set_x, scaled_train_x, train_set_y = dp.prepare_train_test()

    lr = LogisticRegression()
    lr.fit(scaled_train_x, train_set_y)

    predictions = lr.predict(scaled_test_x)
    acc = calc_acc(predictions, test_set_y)

    print(acc)


def Naive_bayes():
    test_set_x, test_set_y, train_set_x, train_set_y = dp.prepare_train_test()

    scaler = StandardScaler()
    scaler.fit(train_set_x)

    scaled_train_x = scaler.transform(train_set_x)
    scaled_test_x = scaler.transform(test_set_x)




if __name__ == '__main__':
    logistic_regression()
    one_vs_all()

