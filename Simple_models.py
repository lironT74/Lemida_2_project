import Data_preprocessing as dp
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import StandardScaler

def logistic_regression():
    test_set_x, test_set_y, train_set_x, train_set_y = dp.prepare_train_test()

    scaler = StandardScaler()
    scaler.fit(train_set_x)

    scaled_train_x = scaler.transform(train_set_x)
    scaled_test_x = scaler.transform(test_set_x)

    lr = LogisticRegression(max_iter=5000)
    lr.fit(scaled_train_x, train_set_y)

    predictions = lr.predict(scaled_test_x)
    acc = np.average([pred==real for pred, real in zip(predictions, test_set_y)])

    print(acc)


if __name__ == '__main__':
    logistic_regression()

