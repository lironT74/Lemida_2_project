import Data_preprocessing as dp
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz


def calc_acc(predictions, real):
    return np.average([pred == real for pred, real in zip(predictions, real)])


def one_vs_all(model):
    test_set_x, scaled_test_x, test_set_y, train_set_x, scaled_train_x, train_set_y = dp.prepare_train_test()

    ovr = OneVsRestClassifier(model)
    ovr.fit(scaled_train_x, train_set_y)

    predictions = ovr.predict(scaled_test_x)
    acc = calc_acc(predictions, test_set_y)

    print(acc)


def desicion_tree():
    test_set_x, scaled_test_x, test_set_y, train_set_x, scaled_train_x, train_set_y = dp.prepare_train_test()

    dt = DecisionTreeClassifier()
    dt.fit(scaled_train_x, train_set_y)

    from graphviz import Source

    export_graphviz(dt, out_file="./tree.dot",
                    feature_names=train_set_x.columns,
                    class_names=["low", "medium", "high"],
                    rounded=True, proportion=False,
                    precision=2, filled=True)

    Source.from_file("tree.dot")

    import pydot

    (graph,) = pydot.graph_from_dot_file('./tree.dot')

    graph.write_png('./tree.png')

    predictions = dt.predict(scaled_test_x)
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
    test_set_x, scaled_test_x, test_set_y, train_set_x, scaled_train_x, train_set_y = dp.prepare_train_test()

    nb = GaussianNB()
    nb.fit(scaled_train_x, train_set_y)

    predictions = nb.predict(scaled_test_x)
    acc = calc_acc(predictions, test_set_y)

    print(acc)


if __name__ == '__main__':
    # logistic_regression()
    # Naive_bayes()
    desicion_tree()

    # one_vs_all(Perceptron(max_iter=5000))
    # one_vs_all(LogisticRegression(max_iter=5000))
    # one_vs_all(LinearSVC(max_iter=5000))
    #
    # one_vs_all(DecisionTreeClassifier())
    #
    # one_vs_all(SVC(max_iter=500000, kernel="linear"))
    # one_vs_all(SVC(max_iter=500000, kernel="rbf"))
    # one_vs_all(SVC(max_iter=500000, kernel="poly"))
    # one_vs_all(SVC(max_iter=500000, kernel="sigmoid"))
