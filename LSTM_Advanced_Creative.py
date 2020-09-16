import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data_preprocessing import prepare_grouped_data_advanced
from auxiliary_functions import *


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


def train_model_advanced(num_of_hours = 4, EPOCHS = 40, HIDDEN_DIM = 100):

    num_of_hours = num_of_hours
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

    print("Training Started")
    accuracy_list = []
    loss_list = []
    epochs = EPOCHS
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
        e_interval = i
        print("Epoch {} Completed\t Loss {:.3f}\t Train Accuracy: {:.3f}\t Test Accuracy: {:.3f}"
              .format(epoch + 1,
                      np.mean(loss_list[-e_interval:]),
                      np.mean(accuracy_list[-e_interval:]),
                      test_acc))


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

    EPOCHS = 3

    print(f"\n\n\n\nAdvanced Model: ")

    for num_of_hours in range(1, 13):
        if 24 % num_of_hours == 0:
            print(f"\nAdvanced Model with {num_of_hours} hours in block: \n")
            train_model_advanced(num_of_hours=num_of_hours, EPOCHS=EPOCHS)


    print(f"\n\n\n\nCreative Model: ")

    Creative_acc = {}
    for num_of_hours in range(1, 13):

        if 24 % num_of_hours == 0:
            print(f'\nTraining of model {num_of_hours} hours in block started')

            X_train, y_train, X_test, y_test = prepare_grouped_data_advanced(num_of_hours)

            best_test_accs = []

            for block_index in range(24 // num_of_hours):
                print(f"Model of hours: {block_index * num_of_hours}:{(block_index + 1) * num_of_hours}")
                best_test_accs.append(train_model_creative(block_index, X_train, y_train, X_test, y_test, num_of_hours=num_of_hours, EPOCHS=EPOCHS))

            for i, acc in enumerate(best_test_accs):
                print(f'Hours: {i * num_of_hours}:{(i + 1) * num_of_hours} max test acc among epochs: {acc}')

            print(f'Average acc of max acc among epochs: {np.average(best_test_accs)}')
            Creative_acc[num_of_hours] = np.average(best_test_accs)


    print(f"Average highest acc among epochs of creative models: ")
    for num_of_hourss, acc in Creative_acc.items():
        print(f' model {num_of_hourss} hours in block: {acc}')

