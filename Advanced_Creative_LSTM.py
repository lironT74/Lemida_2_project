import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data_preprocessing import prepare_grouped_data_advanced
from auxiliary_functions import *
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

class LSTM_Advanced_Creative(nn.Module):
    def __init__(self, vector_emb_dim, hidden_dim, num_classes):
        super(LSTM_Advanced_Creative, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vec_emb_dim = vector_emb_dim

        self.lstm = nn.LSTM(input_size=vector_emb_dim, hidden_size=hidden_dim,
                            num_layers=2, bidirectional=True, batch_first=False)

        self.MLP = nn.Linear(vector_emb_dim, vector_emb_dim)
        self.hidden_to_count = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, hours_array, get_hidden_layer=False):

        hours_tensor = torch.tensor(hours_array.reshape(hours_array.shape[0], self.vec_emb_dim)).float().to(self.device)

        hours_tensor = self.MLP(hours_tensor)
        lstm_out, _ = self.lstm(hours_tensor.view(hours_tensor.shape[0], 1, -1))  # [seq_length, batch_size, 2*hidden_dim]

        if get_hidden_layer:
            return lstm_out

        class_weights = self.hidden_to_count(lstm_out.view(hours_tensor.shape[0], -1))  # [seq_length, tag_dim]
        # return class_weights

        count_type_scores = F.log_softmax(class_weights, dim=1)  # [seq_length, tag_dim]
        return count_type_scores


def evaluate_creative(X_test, y_test, model, num_of_hours, block_index):

    acc = 0
    overall = 0

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")


    with torch.no_grad():
        for week in np.random.permutation(len(X_test)):

            hours_array = np.array(X_test[week][block_index])
            counts_tensor = torch.tensor(y_test[week][block_index]).to(device)
            counts_scores = model(hours_array)
            _, indices = torch.max(counts_scores, 1)
            for true_index, predicted_index in zip(counts_tensor.to("cpu").numpy(), indices.to("cpu").numpy()):
                true_labels = index_to_value(int(true_index), num_of_hours)
                predicted_labels = index_to_value(int(predicted_index), num_of_hours)
                acc += np.sum(true_labels == predicted_labels)
                overall += len(true_labels)

        acc = acc / overall
        # TODO change this if window size is not fixed
    return acc


def evaluate_advanced(X_test, y_test, model, num_of_hours):
    acc = 0
    overall = 0

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")


    with torch.no_grad():

        for week in np.random.permutation(len(X_test)):
            for block_index in range(24 // num_of_hours):

                hours_array = np.array(X_test[week][block_index])
                counts_tensor = torch.tensor(y_test[week][block_index]).to(device)
                counts_scores = model(hours_array)
                _, indices = torch.max(counts_scores, 1)

                for true_index, predicted_index in zip(counts_tensor.to("cpu").numpy(), indices.to("cpu").numpy()):

                    true_labels = index_to_value(int(true_index), num_of_hours)
                    predicted_labels = index_to_value(int(predicted_index), num_of_hours)

                    acc += np.sum(true_labels == predicted_labels)
                    overall += len(true_labels)

        acc = acc / overall
        # TODO change this if window size is not fixed
    return acc


def train_model_advanced(verbose=True, hidden_dim=100, X_train=None, y_train=None, X_test=None, y_test=None,
                         num_of_hours = 4, EPOCHS = 40, HIDDEN_DIM = 100):
    if X_train is None:
        X_train, y_train, X_test, y_test = prepare_grouped_data_advanced(num_of_hours)


    EPOCHS = EPOCHS

    VECTOR_EMBEDDING_DIM = len(X_train[0][0][0])  # number of hours in block * hour vector dim

    HIDDEN_DIM = HIDDEN_DIM
    COUNT_TYPE_SIZE = 3 ** num_of_hours

    model = LSTM_Advanced_Creative(VECTOR_EMBEDDING_DIM, HIDDEN_DIM, COUNT_TYPE_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        model.cuda()

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    accumulate_grad_steps = 70
    if verbose:
        print("Training Started")
    accuracy_list = []
    loss_list = []
    epochs = EPOCHS
    max_test_acc = 0


    for epoch in range(epochs):
        acc = 0
        printable_loss = 0
        i = 0
        overall = 0

        for week in np.random.permutation(len(X_train)):
            for block_index in range(24 // num_of_hours):

                i += 1

                hours_array = np.array(X_train[week][block_index])

                counts_tensor = torch.tensor(y_train[week][block_index]).long().to(device)

                # print(f"week {week}, block {block_index}: {y_train[week][block_index]}, {hours_array}")

                counts_scores = model(hours_array)

                loss = loss_function(counts_scores, counts_tensor)
                loss /= accumulate_grad_steps
                loss.backward()

                if i % accumulate_grad_steps == 0:
                    optimizer.step()
                    model.zero_grad()
                printable_loss += loss.item()
                _, indices = torch.max(counts_scores, 1)

                for true_index, predicted_index in zip(counts_tensor.to("cpu").numpy(), indices.to("cpu").numpy()):
                    true_labels = index_to_value(int(true_index), num_of_hours)
                    predicted_labels = index_to_value(int(predicted_index), num_of_hours)
                    acc += np.sum(true_labels == predicted_labels)
                    overall += len(true_labels)

        printable_loss = accumulate_grad_steps * (printable_loss / len(X_train))
        acc = acc / overall
        loss_list.append(float(printable_loss))
        accuracy_list.append(float(acc))

        test_acc = evaluate_advanced(X_test, y_test, model, num_of_hours)

        if test_acc > max_test_acc:
            max_test_acc = test_acc
        e_interval = i
        if verbose:
            print("Epoch {} Completed\t Loss {:.3f}\t Train Accuracy: {:.3f}\t Test Accuracy: {:.3f}"
                  .format(epoch + 1,
                          np.mean(loss_list[-e_interval:]),
                          np.mean(accuracy_list[-e_interval:]),
                          test_acc))

    return max_test_acc

def train_model_creative(block_index, X_train, y_train, X_test, y_test, num_of_hours, EPOCHS=10, HIDDEN_DIM=100):

    EPOCHS = EPOCHS

    VECTOR_EMBEDDING_DIM = len(X_train[0][block_index][0])  # number of hours in block * hour vector dim


    HIDDEN_DIM = HIDDEN_DIM
    COUNT_TYPE_SIZE = 3 ** num_of_hours

    model = LSTM_Advanced_Creative(VECTOR_EMBEDDING_DIM, HIDDEN_DIM, COUNT_TYPE_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        model.cuda()

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    accumulate_grad_steps = 70

    accuracy_list = []
    loss_list = []
    epochs = EPOCHS

    max_test_acc = 0

    for epoch in range(epochs):
        acc = 0
        printable_loss = 0
        i = 0
        overall = 0

        for week in np.random.permutation(len(X_train)):
            i += 1

            hours_array = np.array(X_train[week][block_index])

            counts_tensor = torch.tensor(y_train[week][block_index]).long().to(device)

            # print(f"week {week}, block {block_index}: {y_train[week][block_index]}, {hours_array}")

            counts_scores = model(hours_array)

            loss = loss_function(counts_scores, counts_tensor)
            loss /= accumulate_grad_steps
            loss.backward()

            if i % accumulate_grad_steps == 0:
                optimizer.step()
                model.zero_grad()
            printable_loss += loss.item()
            _, indices = torch.max(counts_scores, 1)

            for true_index, predicted_index in zip(counts_tensor.to("cpu").numpy(), indices.to("cpu").numpy()):
                true_labels = index_to_value(int(true_index), num_of_hours)
                predicted_labels = index_to_value(int(predicted_index), num_of_hours)
                acc += np.sum(true_labels == predicted_labels)
                overall += len(true_labels)


        printable_loss = accumulate_grad_steps * (printable_loss / len(X_train))
        acc = acc / overall
        loss_list.append(float(printable_loss))
        accuracy_list.append(float(acc))
        test_acc = evaluate_creative(X_test, y_test, model, num_of_hours, block_index)

        if test_acc > max_test_acc:
            max_test_acc = test_acc

        e_interval = i
        print("Epoch {} Completed\t Loss {:.3f}\t Train Accuracy: {:.3f}\t Test Accuracy: {:.3f}"
              .format(epoch + 1,
                      np.mean(loss_list[-e_interval:]),
                      np.mean(accuracy_list[-e_interval:]),
                      test_acc))

    return max_test_acc


if __name__ == '__main__':
    data_set = {}
    validation_set = {}
    for num_of_hours in range(1, 13):
        if 24 % num_of_hours == 0:
            print(num_of_hours)
            X_train, y_train, X_test, y_test = prepare_grouped_data_advanced(num_of_hours)
            data_set[num_of_hours] = prepare_grouped_data_advanced(num_of_hours)
            X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=2 / 3,
                                                                          random_state=57)
            validation_set[num_of_hours] = X_validation, y_validation
    EPOCHS = 40

    print(f"\n\n\n\nAdvanced Model: ")

    Advanced_acc = {}
    best_acc = 0
    best_model = 1
    for num_of_hours in data_set:
        print(f"\nAdvanced Model with {num_of_hours} hours in block: \n")
        X_train, y_train, X_test, y_test = data_set[num_of_hours]
        X_validation, y_validation = validation_set[num_of_hours]
        Advanced_acc[num_of_hours] = train_model_advanced(num_of_hours=num_of_hours, EPOCHS=EPOCHS, verbose=False,
                                                          X_train=X_train, y_train=y_train, X_test=X_validation, y_test=y_validation)


    for num_of_hours, acc in Advanced_acc.items():
        best_acc, best_model = (acc, num_of_hours) if acc > best_acc else (best_acc, best_model)
        print(f' model {num_of_hours} hours in block: {acc}')
    print(f"\n\nHighest validation acc among epochs of Advanced models: {best_acc} \t best num of hours: {best_model}")
    Advanced_acc = list(Advanced_acc)

    plt.title("Advanced LSTM accuracy")
    plt.plot(range(1, len(Advanced_acc) + 1, 1), Advanced_acc, color="red")
    plt.xlabel("Number of hours in block")
    plt.xticks(range(1, len(Advanced_acc) + 1), [1,2,3,4,6,8,12])
    plt.ylabel("Highest acc among epochs", rotation=90)
    plt.show()

    X_train, y_train, X_test, y_test = data_set[best_model]
    test_acc = train_model_advanced(num_of_hours=best_model, EPOCHS=EPOCHS, verbose=False,
                         X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"\n\nTest acc among epochs of Advanced models: {test_acc}")

    # print(f"\n\n\n\nCreative Model: ")
    #
    # Creative_acc = {}
    # for num_of_hours in range(1, 13):
    #
    #     if 24 % num_of_hours == 0:
    #         print(f'\nTraining of model {num_of_hours} hours in block started')
    #
    #         X_train, y_train, X_test, y_test = prepare_grouped_data_advanced(num_of_hours)
    #
    #         best_test_accs = []
    #
    #         for block_index in range(24 // num_of_hours):
    #             print(f"\nModel of hours: {block_index * num_of_hours}:{(block_index + 1) * num_of_hours}")
    #             best_test_accs.append(train_model_creative(block_index, X_train, y_train, X_test, y_test, num_of_hours=num_of_hours, EPOCHS=EPOCHS))
    #
    #         print()
    #         for i, acc in enumerate(best_test_accs):
    #             print(f'Hours: {i * num_of_hours}:{(i + 1) * num_of_hours} max test acc among epochs: {acc}')
    #
    #         print(f'\nAverage acc of max acc among epochs: {np.average(best_test_accs)}')
    #         Creative_acc[num_of_hours] = np.average(best_test_accs)
    #
    #
    # print(f"\n\nAverage highest acc among epochs of creative models: ")
    # for num_of_hourss, acc in Creative_acc.items():
    #     print(f' model {num_of_hourss} hours in block: {acc}')

    # Advanced_acc = [0.9626916914625773, 0.9619440623567171, 0.9620951068228808,
    #                 0.9641214351425943, 0.8476454293628809, 0.8484288354898336,
    #                 0.8491620111731844]
    #
    # plt.title("Advanced LSTM accuracy")
    # plt.plot(range(1, len(Advanced_acc) + 1, 1), Advanced_acc, color="red")
    # plt.xlabel("Number of hours in block")
    # plt.xticks(range(1, len(Advanced_acc) + 1), [1,2,3,4,6,8,12])
    # plt.ylabel("Highest acc among epochs", rotation=90)
    # plt.show()

    # Creative_acc = [0.9574304175280083, 0.9546112066037796, 0.9510909463716049,
    #                 0.905185386455199, 0.8476519337016575,  0.8484346224677717,  0.8491885143570537]
    #
    # plt.title("Creative LSTM accuracy")
    # plt.plot(range(1, len(Creative_acc) + 1, 1), Creative_acc, color="blue")
    # plt.xlabel("Number of hours in block")
    # plt.xticks(range(1, len(Creative_acc) + 1), [1, 2, 3, 4, 6, 8, 12])
    # plt.ylabel("Average highest acc among epochs", rotation=90)
    # plt.show()

