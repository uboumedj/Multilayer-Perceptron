import numpy as np
import pandas as pd
import click
from glob import glob
from utilities import load_data_from_file, load_trained_model
from utilities import convert_predictions_to_labels
from utilities import encode_one_hot, normalize_minmax
from utilities import binary_cross_entropy, mean_squared_error


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
    y = np.array(dataset.iloc[:, 0]).reshape(-1, 1)
    y = encode_one_hot(y)


    # Load every model in the model directory
    models = []
    model_paths = glob(f"{directory}/*.pkl", recursive=True)
    for model in model_paths:
        models.append(load_trained_model(model))
    
    


if __name__ == '__main__':
    main()
