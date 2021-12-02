from pandas import read_parquet, Series

from data import file


def load_train_data():
    return _load_data(file.data_train)


def load_test_data():
    return _load_data(file.data_test)


def _load_data(path):
    data = read_parquet(path)
    return (data.image, data.label)
