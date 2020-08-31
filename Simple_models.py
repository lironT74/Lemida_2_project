import data_preprocessing as dp
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
# import graphviz
from graphviz import Source
import pydot


def calc_acc(predictions, real):
    return np.average(predictions == real)


def one_vs_all(model, name='model'):
    test_set_x, test_set_y, train_set_x, train_set_y = dp.prepare_train_test()

    ovr = OneVsRestClassifier(model)
    ovr.fit(train_set_x, train_set_y)

    predictions = ovr.predict(test_set_x)
    acc = calc_acc(predictions, test_set_y)

    print(f'{f"One vs All {name} accuracy:":50} {round(acc, 3)}')


def decision_tree(max_depth=None, scale=True):
    test_set_x, test_set_y, train_set_x, train_set_y = dp.prepare_train_test(scale)
    data_columns = dp.x_columns

    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(train_set_x, train_set_y)

    export_graphviz(dt, out_file=f"./tree_graphs/tree.dot",
                    feature_names=data_columns,
                    class_names=["low", "medium", "high"],
                    rounded=True, proportion=False,
                    precision=2, filled=True)

    Source.from_file(f"./tree_graphs/tree.dot")

    (graph,) = pydot.graph_from_dot_file(f"./tree_graphs/tree.dot")

    graph.write_png(f'./tree_graphs/tree_{dt.tree_.max_depth}_.png')

    predictions = dt.predict(test_set_x)
    acc = calc_acc(predictions, test_set_y)
    print(f'{"Decision Tree max depth:":50} {dt.tree_.max_depth}')
    print(f'{"Decision Tree accuracy:":50} {round(acc, 3)}')


def logistic_regression():
    test_set_x, test_set_y, train_set_x, train_set_y = dp.prepare_train_test()

    lr = LogisticRegression(solver='lbfgs', multi_class='auto')
    lr.fit(train_set_x, train_set_y)

    predictions = lr.predict(test_set_x)
    acc = calc_acc(predictions, test_set_y)
    print(f'{"Logistic Regression accuracy:":50} {round(acc, 3)}')


def naive_bayes():
    test_set_x, test_set_y, train_set_x, train_set_y = dp.prepare_train_test()

    nb = GaussianNB()
    nb.fit(train_set_x, train_set_y)

    predictions = nb.predict(test_set_x)
    acc = calc_acc(predictions, test_set_y)
    print(f'{"Naive Bayes accuracy:":50} {round(acc, 3)}')


if __name__ == '__main__':
    # logistic_regression()
    # naive_bayes()
    for i in range(1, 25):
        decision_tree(max_depth=i, scale=False)
    decision_tree(scale=False)
    #
    # one_vs_all(Perceptron(max_iter=5000), 'perceptron')
    # one_vs_all(LogisticRegression(max_iter=5000, solver='lbfgs'), 'logistic regression')
    # one_vs_all(LinearSVC(max_iter=5000), 'svm')
    #
    # one_vs_all(DecisionTreeClassifier(), 'decision tree')
    #
    # one_vs_all(SVC(max_iter=500000, kernel="linear"), 'linear svm')
    # one_vs_all(SVC(max_iter=500000, kernel="rbf"), 'rbf svm')
    # one_vs_all(SVC(max_iter=500000, kernel="poly"), 'poly svm')
    # one_vs_all(SVC(max_iter=500000, kernel="sigmoid"), 'sigmoid svm')
