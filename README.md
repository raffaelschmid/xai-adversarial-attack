# Explainable AI - Adversarial Attack

The purpose of this project was to accomplish an assignment that was given during the
[Explainable AI](https://www.fhnw.ch/de/weiterbildung/technik/explainable-ai) course at
[FHNW](https://www.fhnw.ch).

[//]: # (TODO: write idea)

The **project structure** follows the directory convention that can
befound [here](https://towardsdatascience.com/manage-your-data-science-project-structure-in-early-stage-95f91d4d0600)
with a the exception that source files are not located in /src folder. The reason for this is to not have src in import
statements `import src.reporting` nor to deviate from Pip defaults.

| Folder         | Description                                                                                                                                                |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| /data          | Storage location for data. Raw data is downloaded to raw, after processing stored to processed. Trained models are stored in model subdirectory.           |
| /docs          | Documentation that is not part of the code.                                                                                                                |
| /models        | Location for model definitions.                                                                                                                            |
| /notebooks     | Jupyter notebooks are stored in the eda (Exploratory Data Analysis), modeling (Modeling) and evaluation (Evaluation) directories.                          |
| /preprocessing | Extracted Python code that is used during data preprocessing.                                                                                              |
| /reporting     | Extracted Python code that is used during reporting.                                                                                                       |
| /tests         | Module tests                                    