from MEMM.log_linear_memm import Log_Linear_MEMM
from data_preprocessing import prepare_grouped_data


def acc(m1, m2):
    equal_sum = 0
    total = 0
    assert len(m1) == len(m2)
    for v1, v2 in zip(m1, m2):
        assert len(v1) == len(v2)
        for x1, x2 in zip(v1, v2):
            equal_sum += x1 == x2
            total += 1
    accuracy = equal_sum / (len(m1) * len(m1[0]))
    return accuracy


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = prepare_grouped_data(categorized=True, scale=False)
    model = Log_Linear_MEMM(threshold=0)

    # Fitting
    model.fit(X_train, y_train, i_print=10)
    model.save('t=0_categorized_model')

    # Loading
    # model = Log_Linear_MEMM.load_model(r'dumps/t=10_categorized_model.pkl')

    # Predicting
    predictions = model.predict(X_train)
    print(f'model train accuracy {round(acc(predictions, y_train), 3)}')
    predictions = model.predict(X_test)
    print(f'model test accuracy {round(acc(predictions, y_test), 3)}')
