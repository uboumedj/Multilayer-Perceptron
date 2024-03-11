import numpy as np
import pandas as pd
import click
from matplotlib import pyplot as plt
from utilities import load_data_from_file, split_dataframe, save_data_to_file


def show_class_proportion(dataset):
    """
    Prints the distribution of target values in a dataset
    Arguments:
        dataset (pd.dataFrame): the dataset
    """
    # Create an array containing all possible target classes
    target = dataset.iloc[:, 0]
    target_possibilities = target.unique()
    target_possibilities.sort()

    # Count the occurence of each target class in the dataset
    print("|", end="")
    for case in target_possibilities:
        nb_of_occurences = target.value_counts()[case]
        proportion = (nb_of_occurences * 100) / target.shape[0]
        print(f" {case}: {nb_of_occurences} ({proportion:.1f}%) |", end="")
    print("")


def show_random_attribute_pairings(dataset):
    """
    Displays on graphs four pairings of random attributes of the dataset
    Arguments:
        dataset (pd.dataFrame): the dataset
    """
    # Define target column
    target = dataset.iloc[:, 0]

    # Create random values for x and y pairings
    x = np.random.randint(low=2, high=31, size=4)
    y = np.random.randint(low=2, high=31, size=4)

    # Plot four graphs
    for i in range(0, 4):
        ax = dataset.plot.scatter(x=x[i], y=y[i], c=target, colormap='viridis')
    plt.show()


@click.command()
@click.option('--proportion', default=0.75, type=float,
              help='Proportion used for splitting')
@click.option('-v', '--visualisation', is_flag=True, default=False,
              help='Visualise some dataset statistics')
@click.option('--filepath', default="data/data.csv", type=str,
              help='File path to the dataset')
def main(proportion, visualisation, filepath):
    # Load the dataset from the specified file
    raw_dataset = load_data_from_file(filepath)

    # Delete first column (useless data IDs) and transform label column
    # into numeric values (0 and 1)
    dataset = raw_dataset.drop(raw_dataset.columns[0], axis=1)
    target = dataset.iloc[:, 0]
    target = target.apply(lambda x: 1 if x == 'M' else 0)
    dataset = dataset.drop(dataset.columns[0], axis=1)
    dataset = pd.concat([target, dataset], axis=1)

    # Split data into train and test sets, and save them to different
    # files
    dataset_train, dataset_test = split_dataframe(dataset, proportion)
    destination_file_prefix = filepath.split(".")[0]
    save_data_to_file(dataset_train, f"{destination_file_prefix}_train.csv")
    save_data_to_file(dataset_test, f"{destination_file_prefix}_test.csv")

    if visualisation:
        print("Class distribution for Training set: ", end="")
        show_class_proportion(dataset_train)
        print("Class distribution for Test set: ", end="")
        show_class_proportion(dataset_test)
        print("Displaying random attribute pairings...")
        show_random_attribute_pairings(dataset)
        print("Class 0 corresponds to Benign tumors, Class 1 to Malignant")


if __name__ == '__main__':
    main()
