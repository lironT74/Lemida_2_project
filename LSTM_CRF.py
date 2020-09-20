import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from data_preprocessing import prepare_grouped_data

START_INDEX = 3
STOP_INDEX = 4


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

class BiLSTM_CRF(nn.Module):

    def __init__(self, label_to_idx, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag_to_ix = label_to_idx
        self.labelset_size = len(label_to_idx)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=2,
                            bidirectional=True, batch_first=False)

        # Maps the output of the LSTM into tag space.
        self.hidden2label = nn.Linear(hidden_dim * 2, self.labelset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.labelset_size, self.labelset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[START_INDEX, :] = -10000
        self.transitions.data[:, STOP_INDEX] = -10000

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim),
                torch.randn(2, 1, self.hidden_dim))

    def _get_lstm_features(self, hours_array):
        # embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        hours_tensor = torch.from_numpy(hours_array).float().to(self.device)
        lstm_out, _ = self.lstm(hours_tensor.view(hours_tensor.shape[0], 1, -1))
        lstm_out = lstm_out.view(len(hours_array), 2*self.hidden_dim)
        lstm_feats = self.hidden2label(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, labels):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1, device=self.device)
        start_index_tensor = torch.tensor([START_INDEX], dtype=torch.long, device=self.device)
        labels = torch.cat([start_index_tensor, labels])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[labels[i + 1], labels[i]] + feat[labels[i + 1]]
        score = score + self.transitions[STOP_INDEX, labels[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.labelset_size), -10000., device=self.device)
        init_vvars[0][START_INDEX] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_label in range(self.labelset_size):
                # next_label_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_label.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_label_var = forward_var + self.transitions[next_label]
                best_label_id = argmax(next_label_var)
                bptrs_t.append(best_label_id)
                viterbivars_t.append(next_label_var[0][best_label_id].view(1))

            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[STOP_INDEX]
        best_label_id = argmax(terminal_var)
        path_score = terminal_var[0][best_label_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_label_id]
        for bptrs_t in reversed(backpointers):
            best_label_id = bptrs_t[best_label_id]
            best_path.append(best_label_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == START_INDEX # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, hours_array, labels):
        feats = self._get_lstm_features(hours_array)  # [seq_len, label_set_size]
        viterbi_score, best_path = self._viterbi_decode(feats)
        gold_score = self._score_sentence(feats, labels)
        return (viterbi_score - gold_score), best_path

    def forward(self, hours_array):
        lstm_feats = self._get_lstm_features(hours_array)

        # Find the best path, given the features.
        score, label_seq = self._viterbi_decode(lstm_feats)
        return score, label_seq

def evaluate(X_test, y_test, model):
    acc = 0
    with torch.no_grad():
        for day_index in np.random.permutation(len(X_test)):
            hours_array = X_test[day_index]
            counts_tensor = torch.from_numpy(y_test[day_index]).to(device)
            _, best_path = model(hours_array)
            acc += np.sum(counts_tensor.to("cpu").numpy() == np.array(best_path))
        acc = acc / (len(X_test) * len(X_test[0]))
        # TODO change this if window size is not fixed
    return acc


if __name__ == '__main__':
    print('LSTM-CRF started')
    X_train, y_train, X_test, y_test = prepare_grouped_data(scale=True)
    START_LABEL = "<START>"
    STOP_LABEL = "<STOP>"
    VECTOR_EMBEDDING_DIM = X_train[0].shape[1]
    HIDDEN_DIM = 100
    COUNT_TYPE_SIZE = 3


    accumulate_grad_steps = 3

    label_to_idx = {"0": 0, "1": 1, "2": 2, START_LABEL: 3, STOP_LABEL: 4}
    model = BiLSTM_CRF(label_to_idx, VECTOR_EMBEDDING_DIM, HIDDEN_DIM)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    accuracy_list = []
    loss_list = []
    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
        i = 0

        acc = 0  # to keep track of accuracy
        printable_loss = 0  # To keep track of the loss value
        for day_index in np.random.permutation(len(X_train)):
            i += 1

            # hours_array = scalar.transform(X_train[day_index])
            hours_array = X_train[day_index]
            counts_tensor = torch.from_numpy(y_train[day_index]).to(device)
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.

            # Step 3. Run our forward pass.
            loss, best_path = model.neg_log_likelihood(hours_array, counts_tensor)
            loss.backward()
            if i % accumulate_grad_steps == 0:
                optimizer.step()
                model.zero_grad()
            printable_loss += loss.item()
            acc += np.mean(counts_tensor.to("cpu").numpy() == np.array(best_path))

        printable_loss = accumulate_grad_steps * (printable_loss / len(X_train))
        acc = acc / len(X_train)
        loss_list.append(float(printable_loss))
        accuracy_list.append(float(acc))
        test_acc = evaluate(X_test, y_test, model)
        # test_acc = evaluate(X_test, y_test)
        e_interval = i
        print("Epoch {} Completed\t Loss {:.9f}\t Train Accuracy: {:.3f}\t Test Accuracy: {:.3f}"
              .format(epoch + 1,
                      np.mean(loss_list[-e_interval:]),
                      np.mean(accuracy_list[-e_interval:]),
                      test_acc))

    # Check predictions after training
# if __name__ == '__main__':
#     print('hey5')
#     X_train, y_train, X_test, y_test = prepare_grouped_data(scale=True)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     print(X_train[0].shape)
#     # CUDA_LAUNCH_BLOCKING=1
#
#     EPOCHS = 40
#     VECTOR_EMBEDDING_DIM = X_train[0].shape[1]
#     HIDDEN_DIM = 100
#     COUNT_TYPE_SIZE = 3
#
#     model = LSTM_Tagger(VECTOR_EMBEDDING_DIM, HIDDEN_DIM, COUNT_TYPE_SIZE)
#
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda:0" if use_cuda else "cpu")
#
#     if use_cuda:
#         model.cuda()
#     loss_function = nn.NLLLoss()
#     # loss_function = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.01)
#
#     accumulate_grad_steps = 70  # This is the actual batch_size, while we officially use batch_size=1
#
#     # Training start
#     print("Training Started")
#     accuracy_list = []
#     loss_list = []
#     epochs = EPOCHS
#     for epoch in range(epochs):
#         acc = 0  # to keep track of accuracy
#         printable_loss = 0  # To keep track of the loss value
#         i = 0
#         for day_index in np.random.permutation(len(X_train)):
#             i += 1
#
#             # hours_array = scalar.transform(X_train[day_index])
#             hours_array = X_train[day_index]
#             counts_tensor = torch.from_numpy(y_train[day_index]).to(device)
#
#             counts_scores = model(hours_array)
#             loss = loss_function(counts_scores, counts_tensor)
#             loss /= accumulate_grad_steps
#             loss.backward()
#
#             if i % accumulate_grad_steps == 0:
#                 optimizer.step()
#                 model.zero_grad()
#             printable_loss += loss.item()
#             _, indices = torch.max(counts_scores, 1)
#
#             acc += np.mean(counts_tensor.to("cpu").numpy() == indices.to("cpu").numpy())
#
#         printable_loss = accumulate_grad_steps * (printable_loss / len(X_train))
#         acc = acc / len(X_train)
#         loss_list.append(float(printable_loss))
#         accuracy_list.append(float(acc))
#         test_acc = evaluate(X_test, y_test)
#         e_interval = i
#         print("Epoch {} Completed\t Loss {:.3f}\t Train Accuracy: {:.3f}\t Test Accuracy: {:.3f}"
#               .format(epoch + 1,
#                       np.mean(loss_list[-e_interval:]),
#                       np.mean(accuracy_list[-e_interval:]),
#                       test_acc))
