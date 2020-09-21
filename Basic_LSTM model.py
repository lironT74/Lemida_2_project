import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_preprocessing import *
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from auxiliary_functions import create_month_dict
from sklearn.model_selection import train_test_split


class LSTM_Tagger(nn.Module):
    def __init__(self, vector_emb_dim, hidden_dim, num_classes):
        super(LSTM_Tagger, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(input_size=vector_emb_dim, hidden_size=hidden_dim,
                            num_layers=2, bidirectional=True, batch_first=False)
        self.hidden_to_count = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, hours_array, get_hidden_layer=False):
        hours_tensor = torch.from_numpy(hours_array).float().to(self.device)

        lstm_out, _ = self.lstm(
            hours_tensor.view(hours_tensor.shape[0], 1, -1))  # [seq_length, batch_size, 2*hidden_dim]

        if get_hidden_layer:
            return lstm_out

        class_weights = self.hidden_to_count(lstm_out.view(hours_tensor.shape[0], -1))  # [seq_length, tag_dim]
        # return class_weights

        count_type_scores = F.log_softmax(class_weights, dim=1)  # [seq_length, tag_dim]
        return count_type_scores


def evaluate(model, device, X_test, y_test):
    acc = 0
    with torch.no_grad():
        for day_index in range(len(X_test)):
            hours_array = X_test[day_index]
            counts_tensor = torch.from_numpy(y_test[day_index]).to(device)
            counts_scores = model(hours_array)
            _, indices = torch.max(counts_scores, 1)
            acc += np.sum(counts_tensor.to("cpu").numpy() == indices.to("cpu").numpy())
        acc = acc / (len(X_test) * len(X_test[0]))
        # TODO change this if window size is not fixed
    return acc


def evaluate_per_hour(model, X_test, y_test):
    with torch.no_grad():
        counts_scores = model(X_test)
        _, predictions = torch.max(counts_scores, 1)
        predictions = predictions.to("cpu").numpy()

    return np.average([pred == real for pred, real in zip(predictions, y_test)])


def whole_year():
    X_train, y_train, X_test, y_test = divide_data_to_two_years(scale=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(X_train.shape)
    print(X_test.shape)

    # CUDA_LAUNCH_BLOCKING=1

    EPOCHS = 40
    VECTOR_EMBEDDING_DIM = X_train[0].shape[1]
    HIDDEN_DIM = 100
    COUNT_TYPE_SIZE = 3

    model = LSTM_Tagger(VECTOR_EMBEDDING_DIM, HIDDEN_DIM, COUNT_TYPE_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        model.cuda()
    loss_function = nn.NLLLoss()
    # loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    accumulate_grad_steps = 70  # This is the actual batch_size, while we officially use batch_size=1

    # Training start
    print("Training Started")
    accuracy_list = []
    loss_list = []
    epochs = EPOCHS

    acc = 0  # to keep track of accuracy
    printable_loss = 0  # To keep track of the loss value
    i = 0

    counts_tensor = torch.from_numpy(y_train[0]).to(device)
    counts_scores = model(X_train[0])

    loss = loss_function(counts_scores, counts_tensor)
    loss /= accumulate_grad_steps
    loss.backward()

    if i % accumulate_grad_steps == 0:
        optimizer.step()
        model.zero_grad()
    printable_loss += loss.item()
    _, indices = torch.max(counts_scores, 1)

    acc += np.mean(counts_tensor.to("cpu").numpy() == indices.to("cpu").numpy())
    printable_loss = accumulate_grad_steps * (printable_loss / len(X_train))
    acc = acc / len(X_train[0])

    loss_list.append(float(printable_loss))
    accuracy_list.append(float(acc))
    test_acc = evaluate_per_hour(model, X_test[0], y_test)
    e_interval = i
    print("Epoch {} Completed\t Loss {:.3f}\t Train Accuracy: {:.3f}\t Test Accuracy: {:.3f}"
          .format(0 + 1,
                  np.mean(loss_list[-e_interval:]),
                  np.mean(accuracy_list[-e_interval:]),
                  test_acc))


def train_model(verbose=True, hidden_dim=100, X_train=None, y_train=None, X_test=None, y_test=None):
    if X_train is None:
        X_train, y_train, X_test, y_test = prepare_grouped_data(scale=True)
    epochs = 40
    vector_embedding_dim = X_train[0].shape[1]
    hidden_dim = hidden_dim
    count_type_size = 3
    accumulate_grad_steps = 70

    model = LSTM_Tagger(vector_embedding_dim, hidden_dim, count_type_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        model.cuda()

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training start
    if verbose:
        print("Training Started")
    accuracy_list = []
    loss_list = []
    epochs = epochs
    best_acc = 0
    for epoch in range(epochs):
        acc = 0
        printable_loss = 0
        i = 0
        for day_index in np.random.permutation(len(X_train)):
            i += 1

            hours_array = X_train[day_index]
            counts_tensor = torch.from_numpy(y_train[day_index]).to(device)

            counts_scores = model(hours_array)
            loss = loss_function(counts_scores, counts_tensor)
            loss /= accumulate_grad_steps
            loss.backward()

            if i % accumulate_grad_steps == 0:
                optimizer.step()
                model.zero_grad()
            printable_loss += loss.item()
            _, indices = torch.max(counts_scores, 1)

            acc += np.mean(counts_tensor.to("cpu").numpy() == indices.to("cpu").numpy())

        if verbose:
            printable_loss = accumulate_grad_steps * (printable_loss / len(X_train))
            acc = acc / len(X_train)
            loss_list.append(float(printable_loss))
            accuracy_list.append(float(acc))
            test_acc = evaluate(model, device, X_test, y_test)
            best_acc = test_acc if test_acc > best_acc else best_acc
            e_interval = i
            print("Epoch {} Completed\t Loss {:.3f}\t Train Accuracy: {:.3f}\t Test Accuracy: {:.3f}"
                  .format(epoch + 1,
                          np.mean(loss_list[-e_interval:]),
                          np.mean(accuracy_list[-e_interval:]),
                          test_acc))
    return model, best_acc


def save_model(model, model_fname):
    with open(f'dumps/{model_fname}', 'wb') as f:
        pickle.dump(model, f)


def load_model(model_fname):
    with open(f'dumps/{model_fname}', 'rb') as f:
        model = pickle.load(f)
    return model


def LSTM_CM(model):
    _, _, X_test, y_test = prepare_grouped_data(scale=True)
    confusion_matrix = np.zeros((3, 3))
    for x, y in zip(X_test, y_test):
        _, predictions = torch.max(model(x), 1)
        for pred, label in zip(predictions, y):
            confusion_matrix[label][pred] += 1

    for i in range(len(confusion_matrix)):
        confusion_matrix[i] = np.round(confusion_matrix[i] / np.sum(confusion_matrix[i]), 3)

    draw_confusion_matrix(confusion_matrix, xtick_labels=['low', 'medium', 'high'],
                          ytick_labels=['low', 'medium', 'high'], title='Confusion matrix for LSTM predictions')


def draw_confusion_matrix(confusion_matrix, xtick_labels=None, ytick_labels=None, title=None, fontsize=10):
    fig, ax = plt.subplots()

    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=np.max(confusion_matrix)))
    im = ax.imshow(confusion_matrix, cmap='jet', norm=plt.Normalize(vmin=0, vmax=np.max(confusion_matrix)))
    im = ax.imshow(confusion_matrix, cmap='jet')

    divider1 = make_axes_locatable(ax)
    cax = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(sm, ax=ax, cax=cax).ax.tick_params(labelsize=fontsize)

    ax.set_xticks(np.arange(confusion_matrix.shape[0]))
    ax.set_yticks(np.arange(confusion_matrix.shape[1]))

    if xtick_labels:
        ax.set_xticklabels(xtick_labels, fontsize=fontsize)
    if ytick_labels:
        ax.set_yticklabels(ytick_labels, fontsize=fontsize)

    ax.set_xlabel("predictions", fontsize=fontsize)
    ax.set_ylabel("labels", fontsize=fontsize)

    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", color='w', fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize + 4)

    # plt.savefig(f"./cms/{title}.png")
    plt.show()


def LSTM_confusion_matrix_per_day(model):
    _, _, X_test, y_test = prepare_grouped_data(scale=True)
    confusion_matrix = np.zeros((7, 3, 3))
    day = 0
    for x, y in zip(X_test, y_test):
        day = (day + 1) % 7
        _, predictions = torch.max(model(x), 1)
        for pred, label in zip(predictions, y):
            confusion_matrix[day][label][pred] += 1

    for day in range(7):
        for label in range(3):
            confusion_matrix[day][label] = np.round(
                confusion_matrix[day][label] / np.sum(confusion_matrix[day][label]), 3)

    for day in range(7):
        draw_confusion_matrix(confusion_matrix[day], xtick_labels=['low', 'medium', 'high'],
                              ytick_labels=['low', 'medium', 'high'], title=f'LSTM confusion matrix for day {day + 1}')


def LSTM_confusion_matrix_per_month(model):
    _, _, X_test, y_test = prepare_grouped_data(scale=True)

    month_dict = create_month_dict(X_test)

    confusion_matrix = np.zeros((12, 3, 3))
    for x, y in zip(X_test, y_test):
        month = month_dict[x[0, -2]]
        _, predictions = torch.max(model(x), 1)
        for pred, label in zip(predictions, y):
            confusion_matrix[month][label][pred] += 1

    for month in range(12):
        for label in range(3):
            confusion_matrix[month][label] = np.round(
                confusion_matrix[month][label] / np.sum(confusion_matrix[month][label]), 3)

    for month in range(12):
        draw_confusion_matrix(confusion_matrix[month], xtick_labels=['low', 'medium', 'high'],
                              ytick_labels=['low', 'medium', 'high'],
                              title=f'LSTM confusion matrix for month {month + 1}')


def LSTM_error_rate_per_hour(model, mode="percentage"):
    _, _, X_test, y_test = prepare_grouped_data(scale=True)
    n = len(X_test)

    errors = np.zeros(24)
    counts = np.zeros(24)
    for x, y in zip(X_test, y_test):
        _, predictions = torch.max(model(x), 1)
        for i in range(len(x)):
            if predictions[i] != y[i]:
                errors[i] += 1
            counts[i] += 1


    if mode == "percentage":
        error_rate = errors / np.sum(errors)
    else:
        error_rate = errors / counts

    plt.bar(np.arange(1, 25), error_rate)
    plt.xticks(np.arange(1, 25))
    # plt.yticks(np.arange(0, 1.01, 0.05))
    if mode == "percentage":
        plt.title('LSTM Error Distribution among the hours')
    else:
        plt.title('LSTM Error Rate per Hour')

    plt.show()


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = prepare_grouped_data(scale=True)
    X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=2 / 3,
                                                                          random_state=57)
    print('Validation started')
    best_acc = 0
    best_dim = 50
    for hidden_dim in [50, 100, 200]:
        print('---------------------------')
        print(f'Hidden dim: {hidden_dim}')
        _, acc = train_model(verbose=True, hidden_dim=hidden_dim,
                    X_train=X_train, y_train=y_train, X_test=X_validation, y_test=y_validation)
        best_acc, best_dim = (acc, hidden_dim) if acc > best_acc else (best_acc, best_dim)
    print(f'Best accuracy: {best_acc}\tBest dim: {best_dim}')
    _, acc = train_model(verbose=True, hidden_dim=hidden_dim,
                    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f'Test accuracy of the model is {acc}')

