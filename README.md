# Explainable AI - Adversarial Attack

The project is hosted [here](https://github.com/raffaelschmid/xai-adversarial-attack)

The purpose of this project was to accomplish an assignment given
during [Explainable AI](https://www.fhnw.ch/de/weiterbildung/technik/explainable-ai) course at
[FHNW](https://www.fhnw.ch).

## Project

The primary goal of the project is to make SHAP based explanations for a model that was trained based on MNIST. MNIST (
Modified National Institute of Standards and Technology) is a large database of handwritten digits. Afterwards
adversarial images are generated in order to analyze the changes in prediction using SHAP.

The content of this project is aligned as follows:

1. Modelling
    1. [Preprocessing](./notebook/modeling/00_preprocessing.ipynb)
    2. [Training](./notebook/modeling/01_train_model.ipynb)
2. Evaluation
    1. [Confusion Matrix](./notebook/evaluation/01_confusion_matrix.ipynb)
    2. [Explanation SHAP](./notebook/evaluation/02_explain_shap_original.ipynb)
    3. [Adversarial Image Generation](./notebook/evaluation/03_adversarial_images_generation_fgsm.ipynb)
    4. [Explanation SHAP Adversarial Images](./notebook/evaluation/04_explain_shap_adversarial_images.ipynb)
3. Summary
    1. [Reflexion](./notebook/summary/01_reflexion.ipynb)
    2. [References](./notebook/summary/02_references.ipynb)

## Development

### Folder Structure

| Folder         | Description                                                                                                                                      |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| /data          | Storage location for data. Raw data is downloaded to raw, after processing stored to processed. Trained models are stored in model subdirectory. |
| /models        | Location for model definitions.                                                                                                                  |
| /notebooks     | Jupyter notebooks are stored in the eda (Exploratory Data Analysis), modeling (Modeling) and evaluation (Evaluation) directories.                |
| /reporting     | Extracted Python code that is used during reporting.                                                                                             |

### Environment Setup

```
# 1. Install virtual environment in .venv
<python-home>/bin/python -m venv .venv
~/.pyenv/versions/3.6.9/bin/python -m venv .venv

2. Activate .venv
source ./.venv/bin/activate

3. Install requirements
pip install -r requirements.txt 
```
