import numpy as np
import pandas as pd


def load_data_from_file(path):
    """
    Loads a file's data to a pandas DataFrame
    Arguments:
        path (string): path of the file
    Returns:
        pd.dataFrame containing the file's data
    """
    if (not isinstance(path, str)):
        print("error: Path of dataset file must be a string")
        return None
    try:
        df = pd.read_csv(path, header=None)
    except FileNotFoundError:
        print("error: Specified file was not found")
        return None
    print(f"Loading dataset of dimensions {df.shape[0]} x {df.shape[1]}")
    return df


def save_data_to_file(dataset, path):
    """
    Saves a pandas DataFrame dataset to a csv file
    Arguments:
        dataset (pd.dataFrame): dataset
        path (string): path of the file
    """
    if (not isinstance(path, str)):
        print("error: Path of file must be a string")
        return None
    else:
        if (not isinstance(dataset, pd.DataFrame)):
            dataset = pd.DataFrame(dataset)
        dataset.to_csv(path, header=False, index=False)
        print(f"Saved a dataset of shape {dataset.shape} to file '{path}'")


def split_data(x, y, proportion):
    """
    Shuffles and splits the dataset (x and y) into a training and a test set,
    while respecting given proportion of examples to be kept in training set.
    Args:
        x (numpy.array): a matrix of dimension m * n.
        y (numpy.array): a vector of dimension m * 1.
        proportion (float): proportion of dataset assigned to the training set.
    Returns:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
        None if an error occured
    """
    if (not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)
       or not isinstance(proportion, float)):
        return None
    if (x.shape[0] != y.shape[0] or x.size == 0 or y.size == 0):
        return None
    full_dataset = np.concatenate((x, y), 1)
    np.random.shuffle(full_dataset)
    x = full_dataset[..., :-1]
    y = full_dataset[..., -1:]
    separation_index = int(proportion * x.shape[0])
    x_train, x_test = x[:separation_index], x[separation_index:]
    y_train, y_test = y[:separation_index], y[separation_index:]
    print(f"x_train shape : {x_train.shape}")
    print(f"x_valid shape : {x_test.shape}")
    return x_train, x_test, y_train, y_test


def split_dataframe(dataset, proportion):
    """
    Shuffles and splits the dataset into a training and validation set,
    respecting the given proportion
    Arguments:
        dataset (pd.DataFrame): the dataset
        proportion (float): proportion of dataset assigned to training set
    Returns:
        (dataset_train, dataset_test) as a tuple of dataframes
        None if an error occured
    """
    if (not isinstance(dataset, pd.DataFrame)
       or not isinstance(proportion, float)):
        return None
    dataset_train = dataset.sample(frac=proportion)
    dataset_test = dataset.drop(dataset_train.index)
    return dataset_train, dataset_test


def convert_predictions_to_labels(predictions):
    """
    Converts the predictions of a model to represent the
    predicted labels in a one-hot matrix
    Arguments:
        predictions (np.ndarray): model's predictions
    Returns:
        The encoded predictions in a matrix (number_of_classes x m)
    """
    predicted_labels = np.zeros_like(predictions)
    predicted_label_indices = np.argmax(predictions, axis=1)
    i = 0
    for index in predicted_label_indices:
        predicted_labels[i][index] = 1
        i += 1
    return predicted_labels.astype(int)


def encode_one_hot(dataset):
    """
    Encode the given dataset containing target variables into
    a One-Hot matrix, where each example is represented by a
    vector with two possible values, 0 or 1, with 1 representing
    the example's class and 0 the other possible classes.
    Arguments:
        dataset (np.ndarray): target column of the dataset (1 x m)
    Returns:
        The encoded dataset in a matrix (number_of_classes x m)
    """
    dataset = dataset.astype(int)
    number_of_unique_labels = len(np.unique(dataset))
    encoded_dataset = np.zeros((dataset.shape[0], number_of_unique_labels))
    i = 0
    for example in dataset:
        encoded_dataset[i][example] = 1
        i += 1
    return encoded_dataset.astype(int)


def normalize_minmax(dataset):
    normalized_dataset = dataset.copy()
    for col in normalized_dataset.columns:
        max = normalized_dataset[col].max()
        min = normalized_dataset[col].min()
        normalized_dataset[col] = (normalized_dataset[col] - min) / (max - min)
    return normalized_dataset


def binary_cross_entropy(y, y_hat):
    """
    Calculates the binary cross-entropy loss value of the
    predictions y_hat compared to the true labels y
    Arguments:
        y (numpy.array): the true labels
        y_hat (numpy.array): the predictions
    Returns:
        The binary cross-entropy value
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if (y.size == 0 or y_hat.size == 0):
        return None
    y = y.astype(float)
    eps = 1e-15
    J_value = y * np.log(y_hat + eps)
    J_value += (1 - y) * np.log((1 - y_hat) + eps)
    J_value = (- np.sum(J_value) / y.shape[0])
    return J_value


def mean_squared_error(y, y_hat):
    """
    Calculates the loss value of the predictions y_hat compared to the
    true labels y
    Arguments:
        y (numpy.array): the true labels
        y_hat (numpy.array): the predictions
    Returns:
        The mean squared error value
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if (y.size == 0 or y_hat.size == 0):
        return None
    mse = y - y_hat
    mse = mse * mse
    mse = np.sum(mse)
    mse = mse / y.shape[0]
    return mse
