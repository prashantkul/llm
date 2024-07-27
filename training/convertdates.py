from torch import nn
import torch.nn.functional as F
import torch

def create_date_vocab():
    vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
    for i in range(10):  # Digits
        vocab[str(i)] = len(vocab)
    for char in '/-,. ':  # Common date separators and space
        vocab[char] = len(vocab)
    for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
        vocab[month] = len(vocab)
    return vocab

def date_to_indices(date_string, vocab):
    return [vocab.get(char, vocab['<UNK>']) for char in date_string]

def indices_to_date(indices, vocab):
    index_to_char = {v: k for k, v in vocab.items()}
    return ''.join([index_to_char[i] for i in indices if i not in [vocab['<PAD>'], vocab['<START>'], vocab['<END>'], vocab['<UNK>']]])

def prepare_date_data(input_dates, output_dates, vocab, max_length):
    def tokenize(date_string):
        tokens = [vocab.get(char, vocab['<UNK>']) for char in date_string]
        tokens = [vocab['<START>']] + tokens + [vocab['<END>']]
        padding = [vocab['<PAD>']] * (max_length - len(tokens))
        return tokens + padding

    input_indices = [tokenize(date) for date in input_dates]
    output_indices = [tokenize(date) for date in output_dates]
    
    return torch.tensor(input_indices), torch.tensor(output_indices)