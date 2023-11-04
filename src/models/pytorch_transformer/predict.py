import argparse
import spacy
import re
import torch
from torch import Tensor
from torchtext.data.utils import get_tokenizer
import pandas as pd
from utils import generate_square_subsequent_mask
from architecture import Seq2SeqTransformer
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# Define the tokenizer 
token_transform = get_tokenizer('spacy', language='en_core_web_sm')

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Define the maximum size of the sequence
max_size = 128

# Define the model architecture
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3


def load_model(vocab, device, ckpt_path='./models/pytorch_transformer/best.pt'):
    """
    Load a pre-trained transformer model from a checkpoint file.

    Args:
        vocab: The vocabulary.
        device: The device.
        ckpt_path: Path to the checkpoint file (default './models/pytorch_transformer/best.pt').

    Returns:
        Pre-trained transformer model.
    """
    torch.manual_seed(420)
    
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                     NHEAD, len(vocab), FFN_HID_DIM)
    transformer = transformer.to(device)
    ckpt = torch.load(ckpt_path)
    transformer.load_state_dict(ckpt)
    return transformer


def greedy_decode(model, device, src, src_mask, max_len, start_symbol):
    """
    Generate output sequence using greedy algorithm.

    Args:
        model: The model.
        device: The device.
        src: The source sequence.
        src_mask: The mask for the source sequence.
        max_len: The maximum length of the sequence.
        start_symbol: The start symbol for decoding.

    Returns:
        The output sequence.
    """
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device)
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


def detoxify(model, src_sentence, vocab, device):
    """
    Paraphrase the input source sentence to its' detoxified version.

    Args:
        model: The model to use for detoxification.
        src_sentence: The source sentence to detoxify.
        vocab: The vocabulary.
        device: The device.

    Returns:
        The detoxified output sentence.
    """
    model.eval()
    src = Tensor([BOS_IDX] + vocab(token_transform(src_sentence.lower())) + [EOS_IDX])
    src = src.view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, device, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    output = " ".join(vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
    output = re.sub(r'\s+([^\w\s])', r'\1', output)
    return output

def generate_predictions(model, vocab, device, max_length=128):
    """
    Generate predictions for the test set using the provided model and vocab.
    
    Args:
        model: Transformer model.
        vocab: The vocabulary.
        max_length: Maximum length of the input sequence.
    
    Saves the predictions to a CSV file at './data/interim/transformer_results.csv'.
    """
    df_test = pd.read_csv('./data/interim/test.csv')
    
    results = []
    for _, row in tqdm(df_test.iterrows(), total=df_test.shape[0], desc="Generating predictions..."):
        res = detoxify(model, row['reference'][:max_length], vocab, device)
        results.append(res)
        
    df_test['tranformer_result'] = results
    df_test.to_csv('./data/interim/transformer_results.csv', index=False)
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detoxify text using a Pytorch Transformer model.')
    parser.add_argument('--inference', type=str, help='Inference example to detoxify', default=None)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vocab = torch.load('./models/pytorch_transformer/vocab.pth')
    model = load_model(vocab, device)
    
    args = parser.parse_args()
    if args.inference is not None:
        detoxified_text = detoxify(model, args.inference[:max_size], vocab, device)
        print(f"Detoxified text: {detoxified_text}")
    else:
        generate_predictions(model, vocab, device)
        print("Done!")
