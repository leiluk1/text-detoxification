import spacy
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# Define the maximum sequence length
max_size = 128

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


def collate_batch(batch):
    """
    Collate a batch data samples into batch tensors.

    Args:
        batch: A list of tuples containing source and target samples.

    Returns:
        The padded source and target tensors.
    """
    src_batch, tgt_batch = [], []

    for src_sample, tgt_sample in batch:
        src_batch.append(Tensor(src_sample)[:max_size])
        tgt_batch.append(Tensor(tgt_sample)[:max_size])

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)

    return src_batch.type(torch.LongTensor), tgt_batch.type(torch.LongTensor)


def generate_square_subsequent_mask(sz, device):
    """
    Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with 0.0.

    Args:
        sz: The size of the mask.
        device: The device.

    Returns:
        The generated mask.
    """
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, device):
    """
    Create masks for source and target sequences.

    Args:
        src: The source tensor.
        tgt: The target tensor.
        device: The device.

    Returns:
        The source mask, target mask, source padding mask and target padding mask.
    """
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(model, loader, optimizer, loss_fn, device):
    """
    Train the model for one epoch.

    Args:
        model: The model to train.
        loader: The dataloader for the training data.
        optimizer: The optimizer.
        loss_fn: The loss function.
        device: The device.

    Returns:
        The average loss for the epoch.
    """
    model.train()
    losses = 0

    for src, tgt in tqdm(loader):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device=device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(loader))


def evaluate(model, loader, loss_fn, device):
    """
    Evaluate the model on the validation set.

    Args:
        model: The model to evaluate.
        loader: The data loader for the validation data.
        loss_fn: The loss function.
        device: The device.

    Returns:
        The average loss for the validation set.
    """
    model.eval()
    losses = 0

    for src, tgt in tqdm(loader):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device=device)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(loader))

