import logging
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_model(model_dir="./models/t5/best", model_checkpoint="t5-base"):
    """
    Load a fine-tuned T5 model and tokenizer from the specified directory and checkpoint.

    Args:
        model_dir (str): Path to the pre-trained model.
        model_checkpoint (str): Pre-trained checkpoint to use.

    Returns:
        Tuple containing the loaded model and tokenizer.
    """
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()
    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return model, tokenizer


def detoxify(model, inference_request, tokenizer):
    """
    Detoxify the given text using the fine-tuned T5 model.

    Args:
        model (T5ForConditionalGeneration): The T5 model to use for detoxification.
        inference_request (str): The text to detoxify.
        tokenizer (T5Tokenizer): The tokenizer to use for tokenizing the input text.

    Returns:
        str: The detoxified text.
    """
    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids, 
                             num_beams=5,
                             num_return_sequences=1)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True, temperature=0)


def generate_predictions(model, tokenizer, max_length=128):
    """
    Generates predictions for the test set using the provided T5 model and tokenizer.
    
    Args:
        model: A T5 model.
        tokenizer: A T5 tokenizer.
    
    Saves the predictions to a CSV file at './data/interim/t5_results.csv'.
    """
    df_test = pd.read_csv('./data/interim/test.csv')
    results = []
    for _, row in df_test.iterrows():
        res = detoxify(model, row["source"][:max_length], tokenizer)
        results.append(res)
        
    df_test["t5_result"] = results
    df_test.to_csv('./data/interim/t5_results.csv', index=False)




if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    
    logging.info("Loading model and tokenizer...")
    model, tokenizer = load_model()

    logging.info("Generating predictions...")
    generate_predictions(model, tokenizer)
    logging.info("Done!")
