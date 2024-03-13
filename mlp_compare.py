import numpy as np
import pandas as pd
import click
from glob import glob
from matplotlib import pyplot as plt
from utilities import load_data_from_file, load_trained_model
from utilities import encode_one_hot, normalize_minmax


def plot_metrics(models):
    """
    Plots the graphs of the metrics of every stored model,
    to compare them
    Arguments:
        models (np.ndarray): list of all stored models
    """
    fig, axs = plt.subplots(2, 2, figsize=(100, 100))
    for model in models:
        axs[0, 0].plot(model.acc_log, label=model.name)
        axs[0, 0].set_title('Accuracy on Train set')
        axs[0, 1].plot(model.val_acc_log, label=model.name)
        axs[0, 1].set_title('Accuracy on Validation set')
        axs[1, 0].plot(model.loss_log, label=model.name)
        axs[1, 0].set_title('Loss on Train set')
        axs[1, 1].plot(model.val_loss_log, label=model.name)
        axs[1, 1].set_title('Loss on Validation set')

    for ax in axs.flat:
        ax.set(xlabel='epochs', ylabel='value')
        ax.grid()
        ax.legend(loc='best')
        ax.label_outer()

    plt.show()


@click.command()
@click.option('--data', default='data/data_test.csv',
              help='Dataset file path')
@click.option('--directory', default='./models',
              help='Path to the stored trained model')
def main(data, directory):
    # Preparation of the dataset for the prediction
    dataset = load_data_from_file(data)
    x = dataset.drop(dataset.columns[0], axis=1)
    x = np.array(normalize_minmax(x))
    y = np.array(dataset.iloc[:, 0]).reshape(-1, 1)
    y = encode_one_hot(y)

    # Load every model in the model directory
    models = []
    model_paths = glob(f"{directory}/*.pkl", recursive=True)
    for model in model_paths:
        models.append(load_trained_model(model))

    # Plot the graphs
    plot_metrics(models)


if __name__ == '__main__':
    main()
