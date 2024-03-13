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

# A FAIRE: Fonction de création de layers qui prend en compte les parametres
# du programme. Ensuite: RENDRE

def plot_metrics(network):
    """
    Plots the graphs of the metrics of the model
    Arguments:
        network (Model class): the trained network
    """
    fig, axs = plt.subplots(1, 2, figsize=(100, 100))
    axs[0].plot(network.acc_log, label='Training accuracy')
    axs[0].plot(network.val_acc_log, label='Validation accuracy')
    axs[0].set_title('Accuracy')
    axs[1].plot(network.loss_log, label='Training loss')
    axs[1].plot(network.val_loss_log, label='Validation loss')
    axs[1].set_title('Loss')

    for ax in axs.flat:
        ax.set(xlabel='epochs', ylabel='value')
        ax.grid()
        ax.legend(loc='best')
    plt.show()


def generate_layer_list(learning_rate, layer_sizes, x_train, y_train):
    """
    Generates a list of layers according to the program's parameters
    Arguments:
        learning_rate (float): model's learning rate
        layer_sizes (tuple): tuple containing the sizes of hidden layers
        x_train (np.ndarray): the training dataset
        y_train (np.ndarray): the target column of the dataset
    Returns:
        The layer list used to initialise the model
    """
    # Define amount of neurons in each layer
    neuron_amounts = [x_train.shape[1]]
    for size in layer_sizes:
        neuron_amounts.append(size)
    neuron_amounts.append(len(np.unique(y_train)))
    
    # Create layers
    layer_list = []
    for i in range(0, len(neuron_amounts)):
        in_size = neuron_amounts[i] if i == 0 else neuron_amounts[i - 1]
        out_size = neuron_amounts[i]
        activation = 'softmax' if i == len(neuron_amounts) - 1 else 'sigmoid'
        weights = 'heUniform' if activation == 'relu' else 'xavierUniform'
        new_layer = layers.DenseLayer(input_size=in_size,
                                      output_size=out_size,
                                      activation=activation,
                                      learning_rate=learning_rate,
                                      weights_initializer=weights)
        layer_list.append(new_layer)
    return layer_list


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
              help='Loss function used to evaluate model')
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

    # Network structure definition
    layer_list = []
    layer_list = generate_layer_list(learning_rate=learning_rate,
                                     layer_sizes=layer,
                                     x_train=x_train,
                                     y_train=y_train)

    # layer_list = []
    # layer_list.append(layers.DenseLayer(x_train.shape[1], 24, activation='relu', weights_initializer='heUniform'))
    # layer_list.append(layers.DenseLayer(24, 24, activation='relu', weights_initializer='heUniform'))
    # layer_list.append(layers.DenseLayer(24, 24, activation='sigmoid', weights_initializer='xavierUniform'))
    # layer_list.append(layers.DenseLayer(24, 2, activation='softmax', weights_initializer='heUniform'))

    network = model.createNetwork(layer_list)

    # Training loop
    model.fit(network, x_train, y_train, x_valid, y_valid, epochs, batch_size)
    print(network)
    exit()
    
    # Concluding actions (model save, metric display)
    if confirm_file_save():
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        file = open(model_path, "wb")
        print(f"> saving model '{model_path}' to disk...")
        network.name = model_path
        pickle.dump(network, file)
    plot_metrics(network)


if __name__ == '__main__':
    main()
