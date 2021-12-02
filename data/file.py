import os

data_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(f"{data_dir}/..")

data_train = f"{root_dir}/data/raw/mnist_normalized_train.parq"
data_test = f"{root_dir}/data/raw/mnist_normalized_test.parq"
