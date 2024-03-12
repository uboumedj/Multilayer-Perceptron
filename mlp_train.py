import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import click
import pickle
import layers
import model
from utilities import split_data, load_data_from_file
from utilities import encode_one_hot, normalize_minmax
from utilities import convert_predictions_to_labels
from utilities import binary_cross_entropy


def separate_batches(x, y, batch_size):
    """
    Separates the dataset into random batches according to the
    user-specified batch size
    Arguments:
        x (np.ndarray): training dataset
        y (np.ndarray): target variables
        batch_size (int): size of the function's batches
    Yields:
        Batches until whole dataset has been cycled through
    """
    if len(x) != len(y):
        print(f"error: lengths of data and target are incompatible")
        return None
    indices = np.random.permutation(len(x))
    for start_idx in range(0, len(x) - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield x[excerpt], y[excerpt]


def confirm_file_save():
    """
    Asks for confirmation before saving model (you don't want to save 
    a random experimental model)
    Returns:
        Boolean representing the user's decision
    """
    confirmation = input("Do you want to save the resulting model ? (Y/N)\n")
    if confirmation.strip() in ['y', 'Y', 'yes', 'Yes', 'YES']:
        return True
    print(f"Model was discarded !")
    return False


@click.command()
@click.option('--file', default='data/data_train.csv',
              help='Dataset file path')
@click.option('--batch_size', default=8,
              help='Size of mini-batches for training')
@click.option('--epochs', default=84,
              help='Number of epochs to train the model')
@click.option('--learning_rate', default=0.0314,
              help='Learning rate of the model')
@click.option('--loss', default='binaryCrossentropy',
              help='Loss function used during training')
@click.option('-l', '--layer', default=(24, 24), multiple=True,
              help='Size of hidden layers')
@click.option('--model_path', default='models/mlp.pkl',
              help="Path of the saved model's pickle file")
def main(file, batch_size, epochs, learning_rate, loss, layer, model_path):
    # Preparation of the dataset for the training
    dataset = load_data_from_file(file)
    x = dataset.drop(dataset.columns[0], axis=1)
    x = np.array(normalize_minmax(x))
    y = np.array(dataset.iloc[:, 0]).reshape(-1, 1)
    x_train, x_valid, y_train, y_valid = split_data(x, y, 0.801)
    y_train = encode_one_hot(y_train)
    y_valid = encode_one_hot(y_valid)

    loss_log, val_loss_log, acc_log, val_acc_log = [], [], [], []

    """layer_list = []
    layer_list.append(layers.DenseLayer(x_train.shape[1], 24, activation='sigmoid'))
    layer_list.append(layers.DenseLayer(24, 24, activation='sigmoid', weights_initializer='heUniform'))
    layer_list.append(layers.DenseLayer(24, 24, activation='sigmoid', weights_initializer='heUniform'))
    layer_list.append(layers.DenseLayer(24, 2, activation='softmax', weights_initializer='heUniform'))
    network = model.createNetwork(layer_list)"""
    # Network structure definition
    layer_list = []
    layer_list.append(layers.DenseLayer(x_train.shape[1], 50, activation='relu', learning_rate=learning_rate))
    layer_list.append(layers.DenseLayer(50, 100, activation='relu', weights_initializer='heUniform'))
    layer_list.append(layers.DenseLayer(100, 2, activation='sigmoid', weights_initializer='heUniform'))
    network = model.createNetwork(layer_list)

    # Training loop
    for epoch in range(epochs):
        # Train the model on each mini-batch
        for x_batch, y_batch in separate_batches(x_train, y_train, batch_size=batch_size):
            model.train(network, x_batch, y_batch)

        # Compute intermediary metrics on training set and validation set
        predictions_train = network.predict(x_train)
        labels_train = convert_predictions_to_labels(predictions_train)
        acc_log.append(np.mean(labels_train == y_train))
        loss_log.append(binary_cross_entropy(y_train, predictions_train))

        predictions_valid = network.predict(x_valid)
        labels_valid = convert_predictions_to_labels(predictions_valid)
        val_acc_log.append(np.mean(labels_valid == y_valid))
        val_loss_log.append(binary_cross_entropy(y_valid, predictions_valid))

        print(f"epoch {epoch}/{epochs} - ", end="")
        print(f"loss: {loss_log[-1]:.5f} - ", end="")
        print(f"val_loss: {val_loss_log[-1]:.5f} - ", end="")
        print(f"acc: {acc_log[-1]:.5f} - ", end="")
        print(f"val_acc: {val_acc_log[-1]:.5f}")
    
    # Concluding actions (model save, metric display)
    if confirm_file_save():
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        file = open(model_path, "wb")
        print(f"> saving model '{model_path}' to disk...")
        pickle.dump(network, file)
    plt.plot(acc_log, label='train accuracy')
    plt.plot(val_acc_log, label='val accuracy')
    plt.legend(loc='best')
    plt.show()
    plt.plot(loss_log, label='train loss')
    plt.plot(val_loss_log, label='val loss')
    plt.legend(loc='best')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
