import Data_preprocessing as dp
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    pass


def logistic_regression():
    train, test = dp.prepare_train_test()

    lr = LogisticRegression()
