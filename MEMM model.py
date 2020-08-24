from MEMM.log_linear_memm import Log_Linear_MEMM
from data_preprocessing import prepare_grouped_data

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = prepare_grouped_data(scale=True)
    model = Log_Linear_MEMM()
    model.fit(X_train, y_train)
