import logging
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# loading the model and tokenizer
def load_model(model_dir="./models/t5/best", model_checkpoint="t5-base"):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()
    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return model, tokenizer

# detoxify function to generate predictions
def detoxify(model, inference_request, tokenizer):
    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids, 
                             num_beams=5,
                             num_return_sequences=1)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True, temperature=0)

# function to generate predictions for test data
def generate_predictions(model, tokenizer):
    df_test = pd.read_csv('./data/interim/test.csv')
    results = []
    for _, row in df_test.iterrows():
        res = detoxify(model, row["source"], tokenizer)
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
