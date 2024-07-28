import torch
from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.functional import log_softmax
from tqdm import tqdm

from training.dataset import create_datasets_and_loaders
from training.dateformattransformer import DateFormatTransformer

import os, csv

def save_to_csv(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['input', 'output'])  # Header
        writer.writerows(data)

def generate_date_dataset(samples: int):
    num_samples = 50000
    batch_size = 32

    # Generate datasets and create dataloaders
    datasets, dataloaders, vocab = create_datasets_and_loaders(num_samples, batch_size)

    print(f"Shared vocabulary size: {len(vocab)}")
    print("Printing a sample from each dataset..")
    for difficulty in ['easy', 'medium', 'hard']:
        print(f"\nInspecting {difficulty} dataset:")
        inspect_dataloader_data(dataloaders[difficulty]['train'], datasets[difficulty])
    
    return datasets, dataloaders, vocab

# def save_dataset(x_train, x_val, x_test, y_train, y_val, y_test, date_vocab, max_length):
#     # Save the datasets and vocabulary
#     torch.save({    
#         'x_train': x_train,
#         'x_val': x_val,
#         'x_test': x_test,
#         'y_train': y_train,
#         'y_val': y_val,
#         'y_test': y_test,
#         'date_vocab': date_vocab,
#         'max_length': max_length
#     }, 'date_dataset.pt')

def calculate_accuracy(outputs, targets, vocab):
    # Assuming outputs are logits
    predictions = outputs.argmax(dim=-1)
    
    # Adjust predictions or targets to match in length
    min_len = min(predictions.size(1), targets.size(1))
    predictions = predictions[:, :min_len]
    targets = targets[:, :min_len]
    
    correct = (predictions == targets).float()
    
    # Mask out padding tokens
    mask = (targets != vocab['<PAD>']).float()
    accuracy = (correct * mask).sum() / mask.sum()
    
    return accuracy.item()

def train_with_curriculum(model, dataloaders, num_epochs, device, criterion, optimizer, patience, vocab):
    difficulties = ['easy']
    
    for difficulty in difficulties:
        print(f"Training on {difficulty} dataset")
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            total_train_loss = 0
            total_train_accuracy = 0
            num_train_batches = 0
            
            for batch in dataloaders[difficulty]['train']:
                inputs = batch['input'].to(device)
                targets = batch['output'].to(device)
                
                optimizer.zero_grad()
                output = model(inputs, targets[:, :-1])  # The model will create masks internally
                loss = criterion(output.contiguous().view(-1, len(vocab)), targets[:, 1:].contiguous().view(-1))
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                total_train_accuracy += calculate_accuracy(output, targets[:, 1:], vocab)
                num_train_batches += 1
            
            avg_train_loss = total_train_loss / num_train_batches
            avg_train_accuracy = total_train_accuracy / num_train_batches
            
            # Validation
            model.eval()
            total_val_loss = 0
            total_val_accuracy = 0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch in dataloaders[difficulty]['val']:
                    inputs = batch['input'].to(device)
                    targets = batch['output'].to(device)
                    
                    output = model(inputs, targets[:, :-1])
                    loss = criterion(output.contiguous().view(-1, len(vocab)), targets[:, 1:].contiguous().view(-1))
                    
                    total_val_loss += loss.item()
                    total_val_accuracy += calculate_accuracy(output, targets[:, 1:], vocab)
                    num_val_batches += 1
            
            avg_val_loss = total_val_loss / num_val_batches
            avg_val_accuracy = total_val_accuracy / num_val_batches
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), f'best_model_{difficulty}.pth')
                print("Saved new best model")
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            print("-" * 50)


def inspect_dataloader_data(dataloader, dataset_dict):
    print("Inspecting data...")
    for batch in dataloader:
        inputs = batch['input']
        targets = batch['output']
        
        print("\nSample batch:")
        original_dataset = dataset_dict['train'].dataset
        for i in range(min(5, inputs.size(0))):
            try:
                input_date = original_dataset.indices_to_date(inputs[i])
                target_date = original_dataset.indices_to_date(targets[i])
                print(f"Input: {input_date} | Target: {target_date}")
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                print(f"Input indices: {inputs[i]}")
                print(f"Target indices: {targets[i]}")
        
        # Only process one batch
        break

def troubelshoot_predictions(input_data, target_data, vocab, state_dict, model, device):
    
    # Take just the first sample
    input_tensor = input_data[0].unsqueeze(0).to(device)    
    
    print("Vocabulary:")
    for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
        print(f"{idx}: {token}")
    
    print("Model state keys:")
    for key in state_dict.keys():
        print(key)
    
    print("Input tokenization:")
    input_str = "2064/06/02"
    input_tokens = [vocab.get(char, vocab['<UNK>']) for char in input_str]
    print(f"Input: {input_str}")
    print(f"Tokens: {input_tokens}")
    
    print("Input tensor shape:", input_tensor.shape)
    
    print("Raw model output:")
    with torch.no_grad():
        output = model(input_tensor)
    print("Output shape:", output.shape)
    print("First 10 time steps, first 10 logits:")
    print(output[0, :10, :10])  # Print first 10 time steps, first 10 logits
    
    # Convert output to predicted indices
    predicted_indices = output.argmax(dim=-1).squeeze()
    print("Predicted indices shape:", predicted_indices.shape)
    print("Predicted indices:", predicted_indices)
    
    # Create a reverse vocabulary for debugging
    reverse_vocab = {v: k for k, v in vocab.items()}
    
    # Convert input, target, and prediction to strings
    input_str = ''.join([reverse_vocab[t.item()] for t in input_tensor.squeeze() if t.item() not in [vocab['<PAD>'], vocab['<START>'], vocab['<END>']]])
    target_str = ''.join([reverse_vocab[t.item()] for t in target_data[0] if t.item() not in [vocab['<PAD>'], vocab['<START>'], vocab['<END>']]])
    pred_str = ''.join([reverse_vocab[p.item()] for p in predicted_indices if p.item() not in [vocab['<PAD>'], vocab['<START>'], vocab['<END>']]])
    
    print(f"Input:      {input_str}")
    print(f"Target:     {target_str}")
    print(f"Prediction: {pred_str}")

    
    

def date_to_indices(date_string, vocab, max_length):
    """ Convert a date string to a sequence of indices. Indics are padded to max_length. """
    tokens = [vocab.get(char, vocab['<UNK>']) for char in date_string]
    tokens = [vocab['<START>']] + tokens + [vocab['<END>']]
    padding = [vocab['<PAD>']] * (max_length - len(tokens))
    return tokens + padding

def test_model(model, test_dataloader, device, vocab):
    model.eval()
    correct = 0
    total = 0
    
    idx_to_char = {idx: char for char, idx in vocab.items()}
    
    print("Testing model...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            inputs = batch['input'].to(device)
            targets = batch['output'].to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=-1)
            
            # Compare predictions with targets
            mask = (targets != vocab['<PAD>']) & (targets != vocab['<START>']) & (targets != vocab['<END>'])
            correct += ((predicted == targets) & mask).sum().item()
            total += mask.sum().item()
            
            # Print some examples
            if total % 1000 == 0:
                print("\nExample conversions:")
                for i in range(min(3, inputs.size(0))):
                    input_date = ''.join([idx_to_char[idx.item()] for idx in inputs[i] if idx.item() not in [vocab['<PAD>'], vocab['<START>'], vocab['<END>']]])
                    target_date = ''.join([idx_to_char[idx.item()] for idx in targets[i] if idx.item() not in [vocab['<PAD>'], vocab['<START>'], vocab['<END>']]])
                    pred_date = ''.join([idx_to_char[idx.item()] for idx in predicted[i] if idx.item() not in [vocab['<PAD>'], vocab['<START>'], vocab['<END>']]])
                    print(f"Input: {input_date} | Target: {target_date} | Predicted: {pred_date}")

    accuracy = correct / total
    print(f"\nTest Accuracy: {accuracy:.4f}")
    return accuracy


# Generate main function
if __name__ == "__main__":
    max_length = 24
    # Move the model to the appropriate device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    
    mode = "training" # "training" or "evaluation"
    print(f"Mode: {mode}")
    dataset_gen = True
    #test_dataset_generation()
   
    n = 50000
    epochs = 50
    patience = 5
   
    # # Generate the datasets 
    if dataset_gen:
        dataset, dataloaders , vocab = generate_date_dataset(n)
        
    else:
        print("Loading dataset from file")
        # loaded_dataset = torch.load('date_dataset.pt')
    
     # Initialize the model
    model = DateFormatTransformer(
        d_model=256, 
        ffn_hidden=512, 
        num_heads=8, 
        drop_prob=0.2, 
        num_layers=6,
        max_sequence_length=24,
        input_vocab_size=len(vocab),
        output_vocab_size=len(vocab),
        pad_idx=vocab['<PAD>']
    ).to(device)
    
    if mode == "training":
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5)
        # Train the model
        train_with_curriculum(model, dataloaders, epochs, device, criterion, optimizer, patience, vocab)

    else:
        model.load_state_dict(torch.load('best_model_easy.pth'))
        model.to(device)

    # Test the model on 'easy' difficulty level 
    for difficulty in ['easy']:
        print(f"\nTesting on {difficulty} dataset:")
        test_dataloader = dataloaders[difficulty]['test']
        idx_to_char = {idx: char for char, idx in vocab.items()}
    
        print("Inspecting test data...")
        for batch in test_dataloader:
            inputs = batch['input']
            targets = batch['output']
            
            print("\nSample batch:")
            for i in range(min(5, inputs.size(0))):
                input_date = ''.join([idx_to_char[idx.item()] for idx in inputs[i] if idx.item() not in [vocab['<PAD>'], vocab['<START>'], vocab['<END>']]])
                target_date = ''.join([idx_to_char[idx.item()] for idx in targets[i] if idx.item() not in [vocab['<PAD>'], vocab['<START>'], vocab['<END>']]])
                print(f"Input: {input_date} | Target: {target_date}")
            
            # Only process one batch
            break
        
    test_model(model, test_dataloader, device, vocab)