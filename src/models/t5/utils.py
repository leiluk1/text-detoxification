from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split


def create_dataset(df):
    """
    Create a Hugging Face Dataset from a pandas DataFrame containing reference 
    and its' detoxified version.

    Args:
        df: The input dataframe.

    Returns:
        HF Dataset object containing the reference and its' detoxified version.
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
        train_df: The training dataset.
        val_size: The validation ratio.

    Returns:
        A dictionary containing the training and validation datasets.
    """
    df_train, df_valid = train_test_split(train_df, 
                                          test_size=val_size, 
                                          random_state=420)
    
    return DatasetDict(train=create_dataset(df_train),
        validation=create_dataset(df_valid))


def preprocess_function(examples, tokenizer, prefix, max_length=128, 
                        source="toxic", target="non-toxic"):
    """
    Preprocesses the input examples for T5 model training.

    Args:
        examples: An input examples.
        tokenizer: The tokenizer.
        prefix: The prefix for model input.
        source: The key for the source text ("toxic") in the dictionary.
        target: The key for the target text ("non-toxic") in the dictionary.

    Returns:
        A dictionary containing the preprocessed inputs and labels for T5 model training.
    """
    inputs = [prefix + ex[source] for ex in examples["translation"]]
    targets = [ex[target] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(targets, max_length=max_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs


def postprocess_text(preds, labels):
    """
    Postprocesses the predicted and label texts.

    Args:
        preds: The predicted texts.
        labels: The label texts.

    Returns:
        The postprocessed predicted texts and label texts.
    """
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

