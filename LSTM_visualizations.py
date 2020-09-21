import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from auxiliary_functions import create_month_dict
from sklearn.model_selection import train_test_split
from data_preprocessing import *

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

