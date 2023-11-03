import argparse
import spacy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pandas as pd
from custom_dataset import TextDetoxificationDataset
from utils import collate_batch, train_epoch, evaluate
from architecture import Seq2SeqTransformer

import warnings
warnings.filterwarnings("ignore")

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Define the model architecture
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512


def create_datasets():
    df = pd.read_csv('./data/interim/df.csv')
    full_dataset = TextDetoxificationDataset(df)
    
    vocab = full_dataset.vocab
    torch.save(vocab, './models/pytorch_transformer/vocab.pth')
    vocab_size = len(vocab)

    # Create train, test and validation datasets
    train_size = int(0.8 * len(full_dataset))
    test_size = 5000
    val_size = len(full_dataset) - train_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], 
                                              generator=torch.Generator().manual_seed(420))

    return train_dataset, val_dataset, vocab_size



def get_model(vocab_size, device):
    torch.manual_seed(420)

    model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, vocab_size, FFN_HID_DIM)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model.to(device)



def train_model(transformer, train_dataloader, val_dataloader, optimizer, loss_fn, num_epochs, 
                device, ckpt_path = './models/pytorch_transformer/best.pt'):
    best_loss_so_far = float('inf')

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(transformer, train_dataloader, optimizer, loss_fn, device)

        val_loss = evaluate(transformer, val_dataloader, loss_fn, device)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")

        if val_loss < best_loss_so_far:
            torch.save(transformer.state_dict(), ckpt_path)
            best_loss_so_far = val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train pytroch trasnformer model for text detoxification')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    train_dataset, val_dataset, vocab_size = create_datasets()
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    
    model = get_model(vocab_size, device)
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    train_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, args.epochs, device=device)
    
    print(f'Training completed. Best model saved at models/pytorch_transformer.')