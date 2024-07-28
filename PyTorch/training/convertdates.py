from torch import nn
import torch.nn.functional as F
import torch

from datetime import datetime, timedelta
import random

FULL_MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

def custom_strftime(format_str, date):
    format_str = format_str.replace('%B', FULL_MONTHS[date.month - 1])
    format_str = format_str.replace('%d', f'{date.day:02d}')
    format_str = format_str.replace('%m', f'{date.month:02d}')
    return date.strftime(format_str)

def generate_date_dataset(num_samples, start_date="1900-01-01", end_date="2099-12-31", seed=None):
    if seed is not None:
        random.seed(seed)
    
    input_formats = [
        "%Y-%m-%d",   # e.g., 2023-07-26
        "%d/%m/%Y",   # e.g., 26/07/2023
        "%m/%d/%Y",   # e.g., 07/26/2023
        "%B %d, %Y",  # e.g., July 26, 2023
        "%Y/%m/%d",   # e.g., 2023/07/26
        "%d-%m-%Y",   # e.g., 26-07-2023
    ]

    output_formats = [
        "%B %d, %Y",  # e.g., July 26, 2023
        "%Y-%m-%d",   # e.g., 2023-07-26
        "%d/%m/%Y",   # e.g., 26/07/2023
        "%m/%d/%Y",   # e.g., 07/26/2023
        "%d-%m-%Y",   # e.g., 26-07-2023
    ]

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    date_range = (end_date - start_date).days

    input_dates = []
    output_dates = []

    for _ in range(num_samples):
        random_days = random.randint(0, date_range)
        date = start_date + timedelta(days=random_days)
        
        input_format = random.choice(input_formats)
        output_format = random.choice(output_formats)
        
        # Ensure input and output formats are not the same
        while input_format == output_format:
            output_format = random.choice(output_formats)
        
        input_date = custom_strftime(input_format, date)
        output_date = custom_strftime(output_format, date)
        
        input_dates.append(input_date)
        output_dates.append(output_date)

    return input_dates, output_dates

def create_date_vocab():
    vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
    for i in range(10):  # Digits
        vocab[str(i)] = len(vocab)
    for char in '/-,. ':  # Common date separators and space
        vocab[char] = len(vocab)
    for month in FULL_MONTHS:
        for char in month:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab

# Create and print the vocabulary
date_vocab = create_date_vocab()
print("\nVocabulary:")
for char, idx in date_vocab.items():
    print(f"'{char}': {idx}")

def date_to_indices(date_string, vocab):
    return [vocab.get(char, vocab['<UNK>']) for char in date_string]

def indices_to_date(indices, vocab):
    index_to_char = {v: k for k, v in vocab.items()}
    return ''.join([index_to_char[i] for i in indices if i not in [vocab['<PAD>'], vocab['<START>'], vocab['<END>'], vocab['<UNK>']]])

# Updated prepare_date_data function
def prepare_date_data(input_dates, output_dates, vocab, max_length):
    def tokenize(date_string):
        tokens = [vocab.get(char, vocab['<UNK>']) for char in date_string]
        tokens = [vocab['<START>']] + tokens + [vocab['<END>']]
        padding = [vocab['<PAD>']] * (max_length - len(tokens))
        return tokens + padding

    input_indices = [tokenize(date) for date in input_dates]
    output_indices = [tokenize(date) for date in output_dates]
    
    return torch.tensor(input_indices), torch.tensor(output_indices)