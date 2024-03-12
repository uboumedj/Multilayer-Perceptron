import numpy as np
import pandas as pd
import click
import pickle
import layers
import model
from utilities import load_data_from_file
from utilities import convert_predictions_to_labels
from utilities import encode_one_hot, normalize_minmax
from utilities import binary_cross_entropy, mean_squared_error
from sklearn import preprocessing


def load_trained_model(file_path):
    """
    Loads a trained model stored in a pickle file.
    Arguments:
        file_path (str): path to the stored model file
    Returns:
        The Model object stored in the file
    """
    try:
        file = open(file_path, "rb")
        model = pickle.load(file)
        return model
    except FileNotFoundError:
        print(f"error: The file {file_path} doesn't exist !")
    except pickle.UnpicklingError:
        print(f"error: The file {file_path} is not a pkl file !")
    return None


@click.command()
@click.option('--data', default='data/data_test.csv',
              help='Dataset file path')
@click.option('--model_path', default='models/mlp.pkl',
              help='Path to the stored trained model')
@click.option('-d', '--details', is_flag=True, default=False,
              help='View details of every example prediction')
def main(data, model_path, details):
    # Preparation of the dataset for the prediction
    dataset = load_data_from_file(data)
    x = dataset.drop(dataset.columns[0], axis=1)
    x = np.array(normalize_minmax(x))
    y = np.array(dataset.iloc[:, 0]).reshape(-1, 1)
    y = np.array(dataset.iloc[:, 0]).reshape(-1, 1)
    y = encode_one_hot(y)

    # Load the model
    network = load_trained_model(model_path)

    # Use the model to predict the test dataset's labels
    if network is not None:
        predictions = network.predict(x)
        prediction_labels = convert_predictions_to_labels(predictions)

        # Display metrics
        accuracy = np.mean(prediction_labels == y)
        loss = binary_cross_entropy(y, predictions)
        loss_2 = mean_squared_error(y, predictions)
        good_predictions = (prediction_labels.T[1] == y.T[1]).sum()

        # Display detailed info on each prediction if user wants it
        if details:
            for i in range(predictions.shape[0]):
                print(f"->({prediction_labels[i][1]}, {y[i][1]})", end="")
                print(f" - raw [{predictions[i][0]:.3f} ", end="")
                print(f"{predictions[i][1]:.3f}]")

        print(f"Accuracy on Test set: {accuracy:.3f} ({accuracy * 100:.1f}%)")
        print(f"Correctly predicted labels: ({good_predictions}/{y.shape[0]})")
        print(f"Error 1 on Test set: {loss:.3f} (Binary Cross-entropy)")
        print(f"Error 2 on Test set: {loss_2:.3f} (Mean Squared Error)")


if __name__ == '__main__':
    main()
