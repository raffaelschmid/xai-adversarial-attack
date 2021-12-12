import os

data_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(f"{data_dir}/..")

data_train = f"{root_dir}/data/raw/mnist_normalized_train.parq"
data_test = f"{root_dir}/data/raw/mnist_normalized_test.parq"

model_convolutional = f"{root_dir}/data/model/convolutional"
model_convolutional_dataset = f"{root_dir}/data/model/convolutional_dataset"

adversarial_images = f"{root_dir}/data/processed/adversarial_images.parq"

shap_original = f"{root_dir}/data/processed/shap/original.png"
shap_original_missmatches = f"{root_dir}/data/processed/shap/original_missmatches.png"
shap_original_label = f"{root_dir}/data/processed/shap/original_missmatches.parquet"

