import spacy
import torch
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

import warnings
warnings.filterwarnings("ignore")

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Define the tokenizer 
token_transform = get_tokenizer('spacy', language='en_core_web_sm')


class TextDetoxificationDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataframe: pd.DataFrame, vocab = None):
        self.dataframe = dataframe
        self._preprocess()
        self.vocab = vocab or self._create_vocab()

    def _preprocess(self):
        # Lowercase and then tokenize 
        self.dataframe['reference'] = self.dataframe['reference'].str.lower()
        self.dataframe['detox_reference'] = self.dataframe['detox_reference'].str.lower()
        
        self.toxic_sent = self.dataframe['reference'].apply(token_transform).to_list()
        self.detoxified_sent = self.dataframe['detox_reference'].apply(token_transform).to_list()

        self.sentences = self.toxic_sent + self.detoxified_sent
    
    def _create_vocab(self):
        # Creates vocabulary that is used for encoding the sequence of tokens
        vocab = build_vocab_from_iterator(self.sentences, 
                                          min_freq=2,
                                          specials=special_symbols, 
                                          special_first=True)
        
        vocab.set_default_index(UNK_IDX)
        return vocab

    def _get_toxic_sent(self, index: int) -> list:
        # Retrieves toxic sentence from dataset by index
        sent = self.toxic_sent[index]
        return [BOS_IDX] + self.vocab(sent) + [EOS_IDX]
    
    def _get_detoxified_sent(self, index: int) -> list:
        # Retrieves detoxified sentence from dataset by index
        sent = self.detoxified_sent[index]
        return [BOS_IDX] + self.vocab(sent) + [EOS_IDX]

    def __getitem__(self, index) -> tuple[list, list]:
        return self._get_toxic_sent(index), self._get_detoxified_sent(index)
    
    def __len__(self) -> int:
        return len(self.toxic_sent)