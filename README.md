# Explainable AI - Adversarial Attack

The purpose of this project was to accomplish an assignment that was given during the
[Explainable AI](https://www.fhnw.ch/de/weiterbildung/technik/explainable-ai) course at
[FHNW](https://www.fhnw.ch).

## Project

### Introduction

Twenty years from now nearly everything will be powered by some a Machine Learning based model (cars, planes, companies,
etc.). Over time it will get more and more interesting to trick those models by executing Adversarial Attacks which
means to change input to fool the model. Likewise, it will be important to make models against this kind of malicious
attacks.

### Idea

The idea of this project is to train a model based of the handwritten digits dataset (MNIST) and analyze some of the
images using an explainer method out of the course. MNIST (Modified National Institute of Standards and Technology) is a
large database of handwritten digits that is commonly used for training various image processing systems. Afterwards I
will choose an algorithm to generate Adversarial images and try to trick the image classifier based on the newly
generated dataset.

### Procedure

1. Train a Model using Tensorflow and the MNIST dataset
2. Analyze some images in the validation set using an explainer methodology covered in the course (LIME, SHAP, …)
3. Search method to generate Adversarial Images that let the performance (recall) of the model go down.
4. Analyze the resulting Adversarial Images using previously selected explainer methodology

### Resources

| Title                                                                                                           | Description                                                                    | Source     |
|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|------------|
| [Adversarial exaple using FGSM](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm)               | Tutorial creates an adversial example using Fast Gradient Signed Method (FGSM) | Tensorflow |
| [Cleverhans Tutorial](https://github.com/cleverhans-lab/cleverhans/blob/master/tutorials/tf2/mnist_tutorial.py) | Cleverhans is a tool for benchmarking machine learning systems                 | Cleverhans |
| [Adversarial Machine Learning](https://en.wikipedia.org/wiki/Adversarial_machine_learning)                      | Adversarial Machine Learning Wikipedia Article                                 | Wikipedia  |

## Project Structure

The **project structure** follows the directory convention that can
befound [here](https://towardsdatascience.com/manage-your-data-science-project-structure-in-early-stage-95f91d4d0600)
with a the exception that source files are not located in /src folder. The reason for this is to not have src in import
statements `import src.reporting` nor to deviate from Pip defaults.

| Folder         | Description                                                                                                                                      |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| /data          | Storage location for data. Raw data is downloaded to raw, after processing stored to processed. Trained models are stored in model subdirectory. |
| /docs          | Documentation that is not part of the code.                                                                                                      |
| /models        | Location for model definitions.                                                                                                                  |
| /notebooks     | Jupyter notebooks are stored in the eda (Exploratory Data Analysis), modeling (Modeling) and evaluation (Evaluation) directories.                |
| /preprocessing | Extracted Python code that is used during data preprocessing.                                                                                    |
| /reporting     | Extracted Python code that is used during reporting.                                                                                             |
| /tests         | Module tests        <br/>                                                                                                                             |

## Development

### Setup

1. Install virtual environment in .venv

```
<python-home>/bin/python -m venv .venv
~/.pyenv/versions/3.6.9/bin/python -m venv .venv
```

2. Activate .venv

```
source ./.venv/bin/activate
```

3. Install requirements

```
pip install -r requirements.txt 
```
