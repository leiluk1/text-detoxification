# Text Detoxification

This project is intended for the first assignment in the Practical Machine Learning and Deep Learning course at Innopolis University.

Name: Leila Khaertdinova

Email: l.khaertdinova@innopolis.university

Group number: BS21 DS-02 

## Task description

Text Detoxification Task is a process of transforming the text with toxic style into the text with the same meaning but with neutral style.


## Dataset Description

The dataset is a subset of the ParaNMT corpus (50M sentence pairs). The filtered ParaNMT-detox corpus (500K sentence pairs). It is already downloaded from [here](https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip), you can find it in `data/raw` folder. This is the main dataset for this assignment detoxification task.


## Getting started

To run this project, run the following commands in the repo root directory:

1. Create the virtual environment
    ```
    python3 -m venv .venv
    source .venv/bin/activate
    ```
2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Download the English language model for spaCy
    ```
    python -m spacy download en_core_web_sm 
    ```
4. Make sure you have a compatible version of Python 3.9.13 before running the code.
5. For dataset preprocess and creation, run the following command:
    ```
    python ./src/data/make_dataset.py
    ```
    (You can provide an optional argument ```--size {cut_size}``` to specify the size of the data to be processed)
6. To train the models, run the following commands:
    ```
    # for training the pytorch transfromer
    python ./src/models/pytorch_transformer/train.py --batch_size {batch_size} --epochs {num_epochs}

    # for fine-tuning the t5 model
    python ./src/models/t5/train.py --batch_size {batch_size} --epochs {num_epochs}
    ```
    (You can provide the arguments ```batch_size```, ```num_epochs``` to specify the batch size and number of epochs)

7. To get the predictions of the models on test set (5000 text examples), run the commands:
    ```
    # for the pytorch transfromer
    python ./src/models/pytorch_transformer/predict.py 

    # for the t5 model
    python ./src/models/t5/predict.py 
    ```
8. For inference:
    ```
    # for the pytorch transfromer
    python ./src/models/pytorch_transformer/predict.py --inference {your_sentence_example}

    # for the t5 model
    python ./src/models/t5/predict.py --inference {your_sentence_example}
    ```
    (You can provide ```your_sentence_example``` to make inference on your example)