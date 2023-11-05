# Text Detoxification
*Name*: Leila Khaertdinova
*Email*: l.khaertdinova@innopolis.university
*Group number*: BS21 DS-02 

This project is intended for the first assignment in the Practical Machine Learning and Deep Learning course at Innopolis University.

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
    You can provide an optional argument `--size <CUT_SIZE>` to specify the size of the data to be processed.
6. To train the models, run the following commands:
    ```
    # for training the pytorch transformer
    python ./src/models/pytorch_transformer/train.py --batch_size <BATCH_SIZE> --epochs <NUM_EPOCHS>

    # for fine-tuning the t5 model
    python ./src/models/t5/train.py --batch_size <BATCH_SIZE> --epochs <NUM_EPOCHS>
    ```
    You can provide the arguments `<BATCH_SIZE>`, `<NUM_EPOCHS>` to specify the batch size and number of epochs for the training.

7. To download weights, run the following command:
    ```
    python ./src/data/download_weights.py 
    ```
    You can also provide ```--model <MODEL_NAME>``` (`t5` or `transfromer`) to download weights for a specified model.
8. To get the prediction results on a test set (5000 text examples), run the commands:
    ```
    # for the pytorch transformer
    python ./src/models/pytorch_transformer/predict.py 

    # for the t5 model
    python ./src/models/t5/predict.py 
    ```
9. To run on your example:
    ```
    # for the pytorch transformer
    python ./src/models/pytorch_transformer/predict.py --inference "<YOUR_EXAMPLE>"

    # for the t5 model
    python ./src/models/t5/predict.py --inference "<YOUR_EXAMPLE>"
    ```
    You can provide `<YOUR_EXAMPLE>` to make inference on your own example.