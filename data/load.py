from pandas import read_parquet, Series

from data import file


def load_train_dataset(normalize=False):
    x, y = load_train_data()
    x, y = _reshape_input(x, y)
    if normalize:
        x = _normalize(x)
    return x, y


def load_train_data():
    return _load_data(file.data_train)


def load_test_dataset(normalize=False):
    x, y = load_test_data()
    x, y = _reshape_input(x, y)
    if normalize:
        x = _normalize(x)
    return x, y


def load_test_data():
    return _load_data(file.data_test)


def _load_data(path):
    data = read_parquet(path)
    images = data.image
    return images, data.label


def _reshape_input(x, y):
    no_elements = x.shape[0]
    images = x.apply(Series).stack().to_numpy().reshape(no_elements, 28, 28, 1)
    return images, y


def _normalize(x):
    return x / 255
