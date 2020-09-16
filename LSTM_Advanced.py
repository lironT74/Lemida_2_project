import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from auxiliary_functions import make_matrix
from data_preprocessing import prepare_grouped_data_advanced


class LSTM_Advanced(nn.Module):
    def __init__(self, vector_emb_dim, hidden_dim, num_classes):
        super(LSTM_Advanced, self).__init__()
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


def evaluate(X_test, y_test, model, index_to_value):
    acc = 0
    overall = 0
    with torch.no_grad():

        for week in np.random.permutation(len(X_test)):
            for block_index in range(24 // num_of_hours):

                hours_array = X_test[week][block_index]
                counts_tensor = torch.tensor(y_test[week][block_index]).to(device)
                counts_scores = model(hours_array)
                _, indices = torch.max(counts_scores, 1)

                for true_index, predicted_index in zip(counts_tensor.to("cpu").numpy(), indices.to("cpu").numpy()):
                    true_labels = index_to_value[int(true_index)]
                    predicted_labels = index_to_value[int(predicted_index)]
                    acc += np.sum(true_labels == predicted_labels)
                    overall += len(true_labels)

        acc = acc / overall
        # TODO change this if window size is not fixed
    return acc


if __name__ == '__main__':
    num_of_hours = 4
    X_train, y_train, X_test, y_test = prepare_grouped_data_advanced(scale=False)


    _, index_to_value = make_matrix()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print(X_train[0].shape)
    # # CUDA_LAUNCH_BLOCKING=1
    #
    EPOCHS = 40

    VECTOR_EMBEDDING_DIM = len(X_train[0][0][0]) #number of hours in block * hour vector dim


    HIDDEN_DIM = 100
    COUNT_TYPE_SIZE = 3 ** num_of_hours

    model = LSTM_Advanced(VECTOR_EMBEDDING_DIM, HIDDEN_DIM, COUNT_TYPE_SIZE)

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
            for block_index in range(24//num_of_hours):

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
                    true_labels = index_to_value[int(true_index)]
                    predicted_labels = index_to_value[int(predicted_index)]
                    acc += np.sum(true_labels == predicted_labels)
                    overall += len(true_labels)


        printable_loss = accumulate_grad_steps * (printable_loss / len(X_train))
        acc = acc / overall
        loss_list.append(float(printable_loss))
        accuracy_list.append(float(acc))
        test_acc = evaluate(X_test, y_test, model, index_to_value)
        e_interval = i
        print("Epoch {} Completed\t Loss {:.3f}\t Train Accuracy: {:.3f}\t Test Accuracy: {:.3f}"
              .format(epoch + 1,
                      np.mean(loss_list[-e_interval:]),
                      np.mean(accuracy_list[-e_interval:]),
                      test_acc))
