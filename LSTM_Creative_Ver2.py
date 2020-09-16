import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_preprocessing import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM_Creative(nn.Module):
    def __init__(self, vector_emb_dim, hidden_dim, num_classes):
        # self.num_of_models = int(24 / num_of_hours)
        super(LSTM_Creative, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(input_size=vector_emb_dim, hidden_size=hidden_dim,
                    num_layers=2, bidirectional=True, batch_first=False)
        self.MLP = nn.Linear(vector_emb_dim, vector_emb_dim)
        self.hidden_to_count =nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, hours_arrays, get_hidden_layer=False):
        hours_tensor = torch.tensor(hours_arrays).float().to(self.device)
        lstm_out, _ = self.lstm(
            hours_tensor.view(hours_tensor.shape[0], 1, -1))  # [seq_length, batch_size, 2*hidden_dim]

        if get_hidden_layer:
            return lstm_out

        class_weights = self.hidden_to_count(lstm_out.view(hours_tensor.shape[0], -1))  # [seq_length, tag_dim]

        count_type_scores = F.log_softmax(class_weights, dim=1)  # [seq_length, tag_dim]


        return count_type_scores

def evaluate(X_test, y_test, model, num_of_hours, INDEX_OF_MODEL):
    acc = 0
    overall = 0
    with torch.no_grad():
        for day_index in np.random.permutation(len(X_test)):
            hours_array = X_test[day_index][INDEX_OF_MODEL]
            counts_tensor = torch.tensor(y_test[day_index][INDEX_OF_MODEL]).to(device)
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

def index_to_value(x, num_of_hours):
    values = []
    while x > 0:
        values.append(x % 3)
        x //= 3
    while len(values) < num_of_hours:
        values.append(0)
    values.reverse()
    return np.array(values)

def fit_models(num_of_hours):

    X_train, y_train, X_test, y_test = prepare_grouped_data_advanced(creative=True, num_of_hours=num_of_hours) # (seq_len, week, 4_hours, vector_dim) , (seq_len, dim)

    best_test_accs = []
    for index_of_model in range(24 // num_of_hours):
        best_test_accs.append(0)
        EPOCHS = 40

        VECTOR_EMBEDDING_DIM = int(np.shape(X_train[0])[-1] * np.shape(X_train[0])[-2])

        HIDDEN_DIM = 100
        INDEX_OF_MODEL = index_of_model
        # print(INDEX_OF_MODEL)

        model = LSTM_Creative(VECTOR_EMBEDDING_DIM, HIDDEN_DIM, num_classes= 3 ** num_of_hours)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        if use_cuda:
            model.cuda()
        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        accumulate_grad_steps = 30  # This is the actual batch_size, while we officially use batch_size=1

        # Training start
        # print('-----------------------')
        # print(f"Training of model {INDEX_OF_MODEL} Started")

        accuracy_list = []
        loss_list = []
        epochs = EPOCHS
        for epoch in range(epochs):
            acc = 0  # to keep track of accuracy
            printable_loss = 0  # To keep track of the loss value
            i = 0
            overall = 0
            for week_index in np.random.permutation(len(X_train)):
                i += 1
                hours_array = X_train[week_index][INDEX_OF_MODEL]

                print(y_train[week_index])
                # counts_tensor = torch.tensor(y_train[week_index][INDEX_OF_MODEL]).to(device)


                counts_scores = model(hours_array)

                loss = loss_function(counts_scores, counts_tensor)

                loss /= accumulate_grad_steps * (24 // num_of_hours)
                loss.backward()
                printable_loss += loss.item()

                if i % accumulate_grad_steps == 0:
                    optimizer.step()
                    model.zero_grad()

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
            test_acc = evaluate(X_test, y_test, model, num_of_hours, INDEX_OF_MODEL)
            if test_acc > best_test_accs[-1]:
                best_test_accs[-1] = test_acc
            e_interval = i
            # print("Epoch {} Completed\t Loss {:.3f}\t Train Accuracy: {:.3f}\t Test Accuracy: {:.3f}"
            #       .format(epoch + 1,
            #               np.mean(loss_list[-e_interval:]),
            #               np.mean(accuracy_list[-e_interval:]),
            #               test_acc))
    print('finished')
    print(f'Total accuracy is {sum(best_test_accs) / len(best_test_accs)}')
    for i, acc in enumerate(best_test_accs):
        print(f'Hours: {i*num_of_hours}:{(i+1)*num_of_hours} and acc {acc}')

if __name__ == '__main__':
    for i in range(1, 25):
        if 24 % i == 0:
            print(f'Training of model {i} started')
            fit_models(num_of_hours=i)