from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split


def create_dataset(df):
    """
    Create a HF Dataset from a pandas DataFrame containing reference and its' detoxified version.

    Args:
        df (DataFrame): input dataframe.

    Returns:
        Dataset: Hugging Face Dataset object containing the reference and its' detoxified version.
    """
    return Dataset.from_dict(
        {"translation": [{"toxic": source, 
                          "non-toxic": target} 
                         
        for source, target in zip(df["reference"], 
                                  df["detox_reference"])]}
    )


def create_dataset_dict(train_df, val_size=0.2):
    """
    Create a dictionary of datasets for training and validation.

    Args:
        train_df (DataFrame): The training dataset.
        val_size (float): The validation ratio.

    Returns:
        DatasetDict: A dictionary containing the training and validation datasets.
    """
    df_train, df_valid = train_test_split(train_df, 
                                          test_size=val_size, 
                                          random_state=420)
    
    return DatasetDict(
        train=create_dataset(df_train),
        validation=create_dataset(df_valid))


def preprocess_function(examples, tokenizer, prefix, max_length=128, source="toxic", target="non-toxic"):
    """
    Preprocesses the input examples for T5 model training.

    Args:
        examples (dict): An input dictionary.
        tokenizer (T5Tokenizer): The tokenizer to use for tokenizing the sources and targets.
        prefix (str): The prefix for model input.
        source (str, optional): The key for the source text ("toxic") in the dictionary.
        target (str, optional): The key for the target text ("non-toxic") in the dictionary.

    Returns:
        dict: A dictionary containing the preprocessed inputs and labels for T5 model training.
    """
    inputs = [prefix + ex[source] for ex in examples["translation"]]
    targets = [ex[target] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(targets, max_length=max_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs


# simple postprocessing for text
def postprocess_text(preds, labels):
    """
    Postprocesses the predicted and label texts by stripping any leading or trailing whitespaces.

    Args:
        preds (List[str]): The predicted texts.
        labels (List[str]): The label texts.

    Returns:
        Tuple[List[str], List[List[str]]]: A tuple containing the postprocessed predicted texts and label texts.
    """
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

