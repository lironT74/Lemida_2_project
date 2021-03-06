from data_preprocessing import *
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz
from graphviz import Source
import pydot
import matplotlib.pyplot as plt


def calc_acc(predictions, real):
    return np.average(predictions == real)


def one_vs_all(model, data, name='model'):
    test_set_x, test_set_y, train_set_x, train_set_y = data

    ovr = OneVsRestClassifier(model)
    ovr.fit(train_set_x, train_set_y)

    predictions = ovr.predict(test_set_x)
    acc = calc_acc(predictions, test_set_y)

    print(f'{f"One vs All {name} accuracy:":50} {round(acc, 3)}')


def decision_tree(data, max_depth=None, save_tree=False):
    test_set_x, test_set_y, train_set_x, train_set_y = data
    data_columns = X_COLUMNS

    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(train_set_x, train_set_y)

    if save_tree:
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
    print(f'{"Decision Tree max depth:":50}  {dt.tree_.max_depth}')
    print(f'{"Decision Tree accuracy:":50} {round(acc, 3)}')
    return acc


def logistic_regression(data):
    test_set_x, test_set_y, train_set_x, train_set_y = data

    lr = LogisticRegression(solver='lbfgs', multi_class='auto')
    lr.fit(train_set_x, train_set_y)

    predictions = lr.predict(test_set_x)
    acc = calc_acc(predictions, test_set_y)
    print(f'{"Logistic Regression accuracy:":50} {round(acc, 3)}')


def naive_bayes(data):
    test_set_x, test_set_y, train_set_x, train_set_y = data

    nb = GaussianNB()
    nb.fit(train_set_x, train_set_y)

    predictions = nb.predict(test_set_x)
    acc = calc_acc(predictions, test_set_y)
    print(f'{"Naive Bayes accuracy:":50} {round(acc, 3)}')


def decision_tree_graph(data):
    acc = []
    for i in range(1, 25):
        acc.append(decision_tree(data, max_depth=i))

    best_depth = np.argmax(acc)
    print(f'The tree that has the highest test accuracy has a depth of {best_depth + 1} '
          f'and accuracy of {acc[best_depth]}')

    plt.rcParams['font.size'] = 25
    plt.rcParams["figure.figsize"] = (12, 8)

    plt.title("DecisionTree_X validation accuracy")
    plt.plot(range(1, len(acc) + 1), acc, color="fuchsia")
    plt.xlabel("X")
    plt.xticks(range(1, len(acc) + 1), fontsize=17)
    plt.ylabel("validation accuracy", rotation=90)
    plt.savefig("./desiciontreeacc.png", transparent=True)


if __name__ == '__main__':

    data = prepare_train_test()
    test_validation_set_x, test_validation_set_y, train_set_x, train_set_y = data  # for validation
    validation_x, test_set_x, validation_y, test_set_y = train_test_split(test_validation_set_x,
                                                                          test_validation_set_y,
                                                                          test_size=2/3, random_state=57)

    # data = validation_x, validation_y, train_set_x, train_set_y
    # decision_tree_graph(data)


    data = test_set_x, test_set_y, train_set_x, train_set_y


    logistic_regression(data)
    naive_bayes(data)
    decision_tree(data, max_depth=10, save_tree=False)
    one_vs_all(Perceptron(max_iter=500000), data, 'perceptron')
    one_vs_all(LogisticRegression(max_iter=500000, solver='lbfgs'),data,  'logistic regression')
    one_vs_all(DecisionTreeClassifier(), data, 'decision tree')
    one_vs_all(SVC(max_iter=500000, kernel="linear"),data, 'linear svm')
    one_vs_all(SVC(max_iter=500000, kernel="rbf"),data,  'rbf svm')
    one_vs_all(SVC(max_iter=500000, kernel="poly"), data, 'poly svm')
    one_vs_all(SVC(max_iter=500000, kernel="sigmoid"),data,  'sigmoid svm')




