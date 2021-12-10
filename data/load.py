from pandas import read_parquet, Series

from data import file


def load_train_dataset(normalize=False):
    """
    Load training data and reshape it to array of (#images, 28, 28, 1) where 28x28 is the image size and 1 channel due to
    the fact it's grayscale.
    :param normalize:
    :return:
    """
    x, y = _load_data(file.data_train)
    x, y = _reshape_input(x, y)
    if normalize:
        x = _normalize(x)
    return x, y


def load_test_dataset(normalize=False):
    """
    Load test data and reshape it to array of (#images, 28, 28, 1) where 28x28 is the image size and 1 channel due to
    the fact it's grayscale.
    :param normalize:
    :return:
    """
    x, y = _load_data(file.data_test)
    x, y = _reshape_input(x, y)
    if normalize:
        x = _normalize(x)
    return x, y


def _load_data(path):
    data = read_parquet(path)
    return data.image, data.label


def _reshape_input(x, y):
    no_elements = x.shape[0]
    images = x.apply(Series).stack().to_numpy().reshape(no_elements, 28, 28, 1)
    return images, y


def _normalize(x):
    return x / 255
