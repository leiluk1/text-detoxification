# Practical Machine Learning and Deep Learning - Assignment 1 - Text De-toxification

## Task description

Text Detoxification Task is a process of transforming the text with toxic style into the text with the same meaning but with neutral style.

> Formal definition of text detoxification task can be found in [Text Detoxification using Large Pre-trained Neural Models by Dale et al., page 14](https://arxiv.org/abs/2109.08914)

Your assignment is to create a solution for detoxing text with high level of toxicity. It can be a model or set of models, or any algorithm that would work. 

## Data Labeling

Level of Toxicity is labeled with annotating binary classification by people. Text is passed to annotators for them to put specific label toxic/non-toxic. Then number of positive / toxic assesments are divided by the total number of annotators. This process is performed for every entry in the data, resulting in toxicity dataset.

By this process, we have text with toxicity level. However, for training the model it is best to have sample with high toxicity level and its paraphrazed version with low toxicity level. This gives an opportunity for the model to distiguish from the overall meaning of the text and concentrate on decreasing the level of toxicity (dirung the training process). That is why the data that is provided for you has the pair structure. Dataset structure is described in next section.

## Data Description

The dataset is a subset of the ParaNMT corpus (50M sentence pairs). The filtered ParaNMT-detox corpus (500K sentence pairs) can be downloaded from [here](https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip). This is the main dataset for the assignment detoxification task.

The data is given in the `.tsv` format, means columns are separated by `\t` symbol.

| Column | Type | Discription | 
| ----- | ------- | ---------- |
| reference | str | First item from the pair | 
| ref_tox | float | toxicity level of reference text | 
| translation | str | Second item from the pair - paraphrazed version of the reference|
| trn_tox | float | toxicity level of translation text |
| similarity | float | cosine similarity of the texts |
| lenght_diff | float | relative length difference between texts |

## Evaluation criterias

This assignment is on creating the solution, not on evaluating your algorithm. Major part of the grade will be dedicated to the structure of the solution, your development choices and your explonation on how you approached the problem.

Submission should be a link to GitHub repository. It should be open repository, so that the course team could assess it easily.

The structure of the repository should has the following structure:

```
text-detoxification
├── README.md # The top-level README
│
├── data 
│   ├── external # Data from third party sources
│   ├── interim  # Intermediate data that has been transformed.
│   └── raw      # The original, immutable data
│
├── models       # Trained and serialized models, final checkpoints
│
├── notebooks    #  Jupyter notebooks. Naming convention is a number (for ordering),
│                   and a short delimited description, e.g.
│                   "1.0-initial-data-exporation.ipynb"            
│ 
├── references   # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports      # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures  # Generated graphics and figures to be used in reporting
│
├── requirements.txt # The requirements file for reproducing the analysis environment, e.g.
│                      generated with pip freeze › requirements. txt'
└── src                 # Source code for use in this assignment
    │                 
    ├── data            # Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── models          # Scripts to train models and then use trained models to make predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │   
    └── visualization   # Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```


In the top `README.md` file put your name, email and group number. Additionaly, put basic commands how to use your repository. How to transform data, train model and make a predictions.

In the `reports` directory create at least two report about your work. In the **first report**, describe your path in solution creation process. List any architectures, ideas, problems and data that leads to your final solution. In the **second report**, describe your final solution.

In the `notebooks` directory put at least two notebooks. **First notebook** should contain your initial data exploration and basic ideas behind data preprocessing. **Second notebook** should contain information about final solution training and visualization.

In the `src` directory you should put all the code that is used for the final solution. Provide the script for creation intermediate data in `src/data/`. Provide `train` and `prediction` scripts in `src/models`. Provide visualization script in `src/visualization/`.

## Grading criterias

Full assignment without any problems is said to be the `100%` solution.

| Criteria | Weight (%) | Comment |
| ---- | ----- | ----- |
| Structure and code quality | 25 | Code quality, structure, comments, clean repo, commit history, reproducibility (manual seeding) |
| Visualization, notebooks quality | 10 | Jupyter notebooks, visualizations |
| Solution building | 40 |  Solution exploration, references, ideas decription, final report structure |
| Final score, evaluation  | 15 | Evaluation function, final score, quality of results |
| Usability, documentation | 10 | Docstrings, arguments parsing, README |

If **PMLDL Course Team** will have any questions about your assignment or your work fails to show your results you will be called solution defence procedure. 

## Report Examples
### Solution Bulding Report Example

```
# Baseline: Dictionary based
...
# Hypothesis 1: Custom embeddings
...
# Hypothesis 2: More rnn layers
...
# Hypothesis 3: Pretrained embeddings
...
# Results
...
```

### Final Solution Report Example

```
# Introduction
...
# Data analysis
...
# Model Specification
...
# Training Process
...
# Evaluation
...
# Results
...
```

**Good luck! Have fun!**
