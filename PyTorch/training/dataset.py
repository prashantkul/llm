import random
from datetime import datetime, timedelta
import csv
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os

def generate_dataset(num_samples, formats, start_date="1900-01-01", end_date="2099-12-31"):
    OUTPUT_FORMAT = "%m/%d/%Y"
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    date_range = (end_date - start_date).days

    data = []
    for _ in range(num_samples):
        date = start_date + timedelta(days=random.randint(0, date_range))
        input_format = random.choice(formats)
        
        input_date = date.strftime(input_format)
        output_date = date.strftime(OUTPUT_FORMAT)
        
        data.append([input_date, output_date])

    return data

class DateDataset(Dataset):
    def __init__(self, data, vocab=None, max_length=24):
        self.data = data
        self.max_length = max_length
        if vocab is None:
            self.vocab = self.create_vocab()
        else:
            self.vocab = vocab
        self.char_to_idx = self.vocab
        self.idx_to_char = {idx: char for char, idx in self.vocab.items()}

    def create_vocab(self):
        vocab_set = set()
        for input_date, output_date in self.data:
            vocab_set.update(input_date + output_date)
        vocab_list = ['<PAD>', '<START>', '<END>', '<UNK>'] + sorted(list(vocab_set))
        return {char: idx for idx, char in enumerate(vocab_list)}

    def date_to_indices(self, date_string):
        indices = [self.char_to_idx['<START>']]
        indices.extend(self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in date_string)
        indices.append(self.char_to_idx['<END>'])
        
        if len(indices) < self.max_length:
            indices.extend([self.char_to_idx['<PAD>']] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]
        
        return indices

    def indices_to_date(self, indices):
        return ''.join(self.idx_to_char[idx.item()] for idx in indices 
                   if self.idx_to_char[idx.item()] not in ['<PAD>', '<START>', '<END>', '<UNK>'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_date, output_date = self.data[idx]
        return {
            'input': torch.tensor(self.date_to_indices(input_date), dtype=torch.long),
            'output': torch.tensor(self.date_to_indices(output_date), dtype=torch.long)
        }

def create_shared_vocab(all_data):
    vocab_set = set()
    for data in all_data:
        for input_date, output_date in data:
            vocab_set.update(input_date + output_date)
    vocab_list = ['<PAD>', '<START>', '<END>', '<UNK>'] + sorted(list(vocab_set))
    return {char: idx for idx, char in enumerate(vocab_list)}

def create_datasets_and_loaders(num_samples, batch_size=32, train_ratio=0.7, val_ratio=0.15):
    all_data = {}
    for difficulty, formats in DateFormats.FORMATS.items():
        print(f"\nGenerating {difficulty} dataset:")
        all_data[difficulty] = generate_dataset(num_samples, formats)
        
        print("Raw data samples:")
        for i in range(5):
            print(f"Input: {all_data[difficulty][i][0]} | Output: {all_data[difficulty][i][1]}")

    shared_vocab = create_shared_vocab(all_data.values())
    
    datasets = {}
    dataloaders = {}
    
    for difficulty, data in all_data.items():
        full_dataset = DateDataset(data, vocab=shared_vocab)
        
        # Split the dataset
        total_size = len(full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        datasets[difficulty] = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        
        dataloaders[difficulty] = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=batch_size),
            'test': DataLoader(test_dataset, batch_size=batch_size)
        }
    
    return datasets, dataloaders, shared_vocab

# Define formats for each difficulty level
class DateFormats:
    FORMATS = {
        'easy': ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"],
        'medium': ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y"],
        'hard': ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%Y/%m/%d", "%d-%m-%Y", "%Y.%m.%d", "%d.%m.%Y", "%m.%d.%Y"]
    }

    @classmethod
    def get_formats(cls, difficulty):
        return cls.FORMATS.get(difficulty, [])

    @classmethod
    def all_formats(cls):
        return [format for formats in cls.FORMATS.values() for format in formats]