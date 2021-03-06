import pycrfsuite
from data_preprocessing import prepare_grouped_data
import numpy as np

X_COLUMNS = ['t1', 't2', 'hum', 'wind_speed', 'weather_code', 'is_holiday', 'is_weekend', 'season', 'hour', 'month',
             'day']
Y_COLUMN = 'cnt_categories'


def train_and_save_model(X_train, y_train, model_path='crf_model.crfsuite'):
    X_train = [date2features(x) for x in X_train]
    y_train = [[str(label) for label in y] for y in y_train]
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 150,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    trainer.train(model_path)


def evaluate_model(X_test, y_test, model_path='crf_model.crfsuite'):
    X_test = [date2features(x) for x in X_test]
    y_test = [[str(label) for label in y] for y in y_test]
    tagger = pycrfsuite.Tagger()
    tagger.open(model_path)
    acc = 0
    for date, labels in zip(X_test, y_test):
        predicted = np.array(tagger.tag(date))
        corrected = np.array(labels)
        acc += np.sum(predicted == corrected) / (24 * len(X_test))
    print(f'Accuracy: {acc}')


def hour2features(hours, i):
    curr_hour = hours[i]
    features = [column + '=' + str(value) for column, value in zip(X_COLUMNS, curr_hour)]
    features.append('bias')
    if i > 0:
        prev_hour = hours[i-1]
        prev_features = ['prev_'+column + '=' + str(value) for column, value in zip(X_COLUMNS, prev_hour)]
        features.extend(prev_features)
    else:
        features.append('BOS')

    return features


def date2features(day):
    return [hour2features(day, i) for i in range(len(day))]

if __name__ == '__main__':
    model_location = 'dumps/crf_model.crfsuite'
    X_train, y_train, X_test, y_test = prepare_grouped_data(scale=False)
    train_and_save_model(X_train, y_train, model_path=model_location)
    evaluate_model(X_test, y_test, model_path=model_location)

