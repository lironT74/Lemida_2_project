
from LSTM_visualizations import *

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
    return acc



def train_model(verbose=True, hidden_dim=100, X_train=None, y_train=None, X_test=None, y_test=None, epochs=40):

    if X_train is None:
        X_train, y_train, X_test, y_test = prepare_grouped_data(scale=True)
    epochs = epochs
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


if __name__ == '__main__':
    X_train, y_train, X_test_and_validation, y_test_and_validation = prepare_grouped_data(scale=True)
    X_validation, X_test, y_validation, y_test = train_test_split(X_test_and_validation, y_test_and_validation, test_size=2 / 3,
                                                                          random_state=57)


    print('Validation started')
    best_acc = 0
    best_dim = 50
    epochs = 40

    for hidden_dim in [50, 100, 200]:
        print('---------------------------')
        print(f'Hidden dim: {hidden_dim}')
        _, acc = train_model(verbose=True, hidden_dim=hidden_dim,
                    X_train=X_train, y_train=y_train, X_test=X_validation, y_test=y_validation, epochs=epochs)
        best_acc, best_dim = (acc, hidden_dim) if acc > best_acc else (best_acc, best_dim)

    print(f'Best accuracy: {best_acc}\tBest dim: {best_dim}')
    _, acc = train_model(verbose=True, hidden_dim=best_dim,
                    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, epochs=epochs)
    print(f'Test accuracy of the model is {acc}')

