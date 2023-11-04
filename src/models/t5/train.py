import argparse
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
from IPython.display import display, HTML
import numpy as np
from utils import create_dataset_dict, preprocess_function, postprocess_text
from datasets import load_metric

import warnings 
warnings.filterwarnings('ignore')


# Specify model checkpoint
model_checkpoint = "t5-base"
# Path to the training dataset
train_csv_path = './data/interim/train.csv'

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Load the BLUE and ROUGE metrics
bleu = load_metric("sacrebleu")
rouge = load_metric("rouge")
    

def compute_metrics(eval_preds):
    """
    Computes evaluation function to pass to trainer.

    Args:
        eval_preds: tuple of predictions and labels.

    Returns:
        A dictionary containing the computed metrics.
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result_bleu = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    result_rouge = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    result = {"bleu": result_bleu["score"],
              "rouge1": result_rouge["rouge1"].mid.fmeasure,
              "rouge2": result_rouge["rouge2"].mid.fmeasure}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def train_model(batch_size, epochs, output_dir='./models/t5/'):
    """
    Train a T5 model for text detoxification using the provided dataset.

    Args:
        batch_size: The batch size.
        epochs: The number of epochs.
        output_dir: The directory where save the results.
    """

    # Set seed for reproducibility
    transformers.set_seed(420)
    
    # Create a model for the pretrained model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    df_train = pd.read_csv(train_csv_path)
    dataset = create_dataset_dict(df_train)
    
    # Prefix for model input
    prefix = "paraphrase:"
    
    # Preprocess the dataset
    tokenized_dataset = dataset.map(lambda examples: preprocess_function(examples, tokenizer, prefix), batched=True)
    
    # Specify training arguments
    arguments = Seq2SeqTrainingArguments(
        f"{output_dir}t5-finetuned-detox",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=True,
        disable_tqdm=True,
        report_to='tensorboard',
    )

    # Create the batch for training
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Train the model
    trainer = Seq2SeqTrainer(
        model,
        arguments,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    trainer.train()

    # Saving the best model
    trainer.save_model(f"{output_dir}best")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune T5 model for text detoxification')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
    args = parser.parse_args()
    
    train_model(args.batch_size, args.epochs)
    print(f'Training completed. Model saved at models/t5.')
