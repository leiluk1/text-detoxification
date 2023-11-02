import argparse
import logging
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
from IPython.display import display, HTML
from utils import create_dataset_dict, preprocess_function

import warnings 
warnings.filterwarnings('ignore')


# Specify model checkpoint
model_checkpoint = "t5-base"
# Path to the training dataset
train_csv_path = './data/interim/train.csv'


def train_model(batch_size, epochs, output_dir='./models/t5/'):
    """
    Train a T5 model for text detoxification using the provided dataset.

    Args:
        batch_size (int): The batch size to use for training.
        epochs (int): The number of epochs to train for.
        output_dir (str): The directory where save the results.

    Returns:
        None
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

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

    args = Seq2SeqTrainingArguments(
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
        args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    trainer.train()

    # Saving model
    trainer.save_model(f"{output_dir}best")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune T5 model for text detoxification')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs for training')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f'Training T5 model with batch size {args.batch_size} and {args.epochs} epochs')
    
    train_model(args.batch_size, args.epochs)
    logging.info(f'Training completed. Model saved at models/t5.')
