import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import click
import pickle
import layers
import model
from utilities import split_data, load_data_from_file
from utilities import encode_one_hot, normalize_minmax
from utilities import convert_predictions_to_labels
from utilities import binary_cross_entropy

from tqdm import trange
from IPython.display import clear_output



def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


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
@click.option('--model_path', default='mlp.pkl',
              help="Path of the saved model's pickle file")
def main(file, batch_size, epochs, learning_rate, loss, layer, model_path):
    # Preparation of the dataset for the training
    dataset = load_data_from_file(file)
    x = dataset.drop(dataset.columns[0], axis=1)
    x = np.array(normalize_minmax(x))
    y = np.array(dataset.iloc[:, 0]).reshape(-1, 1)
    x_train, x_valid, y_train, y_valid = split_data(x, y, 0.801)
    y_train_encoded = encode_one_hot(y_train)
    y_valid_encoded = encode_one_hot(y_valid)

    loss_log, val_loss_log, acc_log, val_acc_log = [], [], [], []

    """layer_list = []
    layer_list.append(layers.DenseLayer(x_train.shape[1], 24, activation='sigmoid'))
    layer_list.append(layers.DenseLayer(24, 24, activation='sigmoid', weights_initializer='heUniform'))
    layer_list.append(layers.DenseLayer(24, 24, activation='sigmoid', weights_initializer='heUniform'))
    layer_list.append(layers.DenseLayer(24, 2, activation='softmax', weights_initializer='heUniform'))"""
    # Network structure definition
    layer_list = []
    layer_list.append(layers.DenseLayer(x_train.shape[1], 50, activation='relu'))
    layer_list.append(layers.DenseLayer(50, 100, activation='relu', weights_initializer='heUniform'))
    layer_list.append(layers.DenseLayer(100, 2, activation='sigmoid', weights_initializer='heUniform'))
    network = model.createNetwork(layer_list)

    # Training loop
    for epoch in range(epochs):
        # Train the model on each mini-batch
        for x_batch, y_batch in iterate_minibatches(x_train, y_train_encoded, batchsize=batch_size, shuffle=True):
            model.train(network, x_batch, y_batch)

        # Compute intermediary metrics on training set and validation set
        predictions_on_train = model.predict(network, x_train)
        labels_on_train = convert_predictions_to_labels(predictions_on_train)
        acc_log.append(np.mean(labels_on_train == y_train_encoded))
        loss_log.append(binary_cross_entropy(y_train_encoded, predictions_on_train))

        predictions_on_valid = model.predict(network, x_valid)
        labels_on_valid= convert_predictions_to_labels(predictions_on_valid)
        val_acc_log.append(np.mean(labels_on_valid == y_valid_encoded))
        val_loss_log.append(binary_cross_entropy(y_valid_encoded, predictions_on_valid))

        print(f"epoch {epoch}/{epochs} - ", end="")
        print(f"loss: {loss_log[-1]} - ", end="")
        print(f"val_loss: {val_loss_log[-1]} - ", end="")
        print(f"acc: {acc_log[-1]} - ", end="")
        print(f"val_acc: {val_acc_log[-1]}")
    
    # Concluding actions (model save, metric display)
    if confirm_file_save():
        file = open(model_path, "wb")
        print(f"> saving model '{model_path}' to disk...")
        pickle.dump(network, file)
    plt.plot(acc_log,label='train accuracy')
    plt.plot(val_acc_log,label='val accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    plt.plot(loss_log,label='train loss')
    plt.plot(val_loss_log,label='val loss')
    plt.legend(loc='best')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
