import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from data_preprocessing import *


class LSTM_Tagger(nn.Module):
    def __init__(self, vector_emb_dim, hidden_dim, num_classes):
        super(LSTM_Tagger, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(input_size=vector_emb_dim, hidden_size=hidden_dim,
                            num_layers=2, bidirectional=True, batch_first=False)
        self.hidden_to_count = nn.Linear(hidden_dim * 2, num_classes)


    def forward(self, hours_array, get_hidden_layer=False):

        hours_tensor = torch.from_numpy(hours_array).float().to(self.device)

        lstm_out, _ = self.lstm(hours_tensor.view(hours_tensor.shape[0], 1, -1))  # [seq_length, batch_size, 2*hidden_dim]

        if get_hidden_layer:
            return lstm_out

        class_weights = self.hidden_to_count(lstm_out.view(hours_tensor.shape[0], -1))  # [seq_length, tag_dim]
        # return class_weights

        count_type_scores = F.log_softmax(class_weights, dim=1)  # [seq_length, tag_dim]
        return count_type_scores


def evaluate(X_test, y_test):
    acc = 0
    with torch.no_grad():
        for day_index in np.random.permutation(len(X_test)):
            hours_array = X_test[day_index]
            counts_tensor = torch.from_numpy(y_test[day_index]).to(device)
            counts_scores = model(hours_array)
            _, indices = torch.max(counts_scores, 1)
            acc += np.sum(counts_tensor.to("cpu").numpy() == indices.to("cpu").numpy())
        acc = acc / (len(X_test) * len(X_test[0]))
        # TODO change this if window size is not fixed
    return acc


def evaluate_per_hour(X_test, y_test):
    with torch.no_grad():
        counts_scores = model(X_test)
        _, predictions = torch.max(counts_scores, 1)
        predictions = predictions.to("cpu").numpy()

    return np.average([pred==real for pred, real in zip(predictions, y_test)])



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
    test_acc = evaluate_per_hour(X_test[0], y_test)
    e_interval = i
    print("Epoch {} Completed\t Loss {:.3f}\t Train Accuracy: {:.3f}\t Test Accuracy: {:.3f}"
          .format(0 + 1,
                  np.mean(loss_list[-e_interval:]),
                  np.mean(accuracy_list[-e_interval:]),
                  test_acc))



if __name__ == '__main__':
    X_train, y_train, X_test, y_test = prepare_grouped_data(scale=True)
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


    for epoch in range(epochs):
        acc = 0  # to keep track of accuracy
        printable_loss = 0  # To keep track of the loss value
        i = 0
        for day_index in np.random.permutation(len(X_train)):
            i += 1

            # hours_array = scalar.transform(X_train[day_index])
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


        printable_loss = accumulate_grad_steps * (printable_loss / len(X_train))
        acc = acc / len(X_train)
        loss_list.append(float(printable_loss))
        accuracy_list.append(float(acc))
        test_acc = evaluate_per_hour(X_test, y_test)
        e_interval = i
        print("Epoch {} Completed\t Loss {:.3f}\t Train Accuracy: {:.3f}\t Test Accuracy: {:.3f}"
              .format(epoch + 1,
                      np.mean(loss_list[-e_interval:]),
                      np.mean(accuracy_list[-e_interval:]),
                      test_acc))
