from training.dataset import generate_date_dataset
from training.convertdates import create_date_vocab, date_to_indices, indices_to_date, prepare_date_data
from training.dateformattransformer import DateFormatTransformer
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Generate the dataset
input_dates, output_dates = generate_date_dataset(25000)  

# Create the vocabulary for the model. The date vocab includes digits, common date separators, and month names.
date_vocab = create_date_vocab()

# This is the max lenght of the date string that the model can handle.
max_length = 20  

# Prepare the data. This includes tokenizing the dates and padding them to a maximum length.
x, y = prepare_date_data(input_dates, output_dates, date_vocab, max_length)

# Split the data into train, validation, and test sets
x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.1, random_state=42)

print(f"Train set size: {len(x_train)}")
print(f"Validation set size: {len(x_val)}")
print(f"Test set size: {len(x_test)}")

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


# Initialize the model
model = DateFormatTransformer(
    d_model=128, 
    ffn_hidden=256, 
    num_heads=4, 
    drop_prob=0.1, 
    num_layers=3,
    max_sequence_length=max_length,
    input_vocab_size=len(date_vocab),
    output_vocab_size=len(date_vocab),
    pad_idx=date_vocab['<PAD>']
)

# Move the model to the appropriate device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=date_vocab['<PAD>'])
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Training loop with validation
num_epochs = 10
batch_size = 32
best_val_loss = float('inf')

# Training loop
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    total_train_accuracy = 0
    num_batches = 0

    for i in range(0, len(x_train), batch_size):
        batch_x = x_train[i:i+batch_size].to(device)
        batch_y = y_train[i:i+batch_size].to(device)
        
        optimizer.zero_grad()
        output = model(batch_x, batch_y[:, :-1])  # The model will create masks internally
        loss = criterion(output.contiguous().view(-1, len(date_vocab)), batch_y[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        total_train_accuracy += calculate_accuracy(output, batch_y[:, 1:], date_vocab)
        num_batches += 1
    
    avg_train_loss = total_train_loss / num_batches
    avg_train_accuracy = total_train_accuracy / num_batches

    # Validation step (similar changes)
    model.eval()
    total_val_loss = 0
    total_val_accuracy = 0
    num_val_batches = 0

    with torch.no_grad():
        for i in range(0, len(x_val), batch_size):
            batch_x = x_val[i:i+batch_size].to(device)
            batch_y = y_val[i:i+batch_size].to(device)
            
            output = model(batch_x, batch_y[:, :-1])
            loss = criterion(output.contiguous().view(-1, len(date_vocab)), batch_y[:, 1:].contiguous().view(-1))
            
            total_val_loss += loss.item()
            total_val_accuracy += calculate_accuracy(output, batch_y[:, 1:], date_vocab)
            num_val_batches += 1

    avg_val_loss = total_val_loss / num_val_batches
    avg_val_accuracy = total_val_accuracy / num_val_batches

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}")
    
    

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("Saved new best model")

    print("-" * 50)

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# def evaluate_full_dates(model, input_data, target_data, vocab, device):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for i in range(len(input_data)):
#             input_tensor = input_data[i].unsqueeze(0).to(device)
#             target_tensor = target_data[i].to(device)
            
#             output = model(input_tensor)  # No need to pass target for inference
#             predicted_indices = output.argmax(dim=-1).squeeze()
            
#             # Ensure predicted_indices has the same length as target_tensor
#             if len(predicted_indices.shape) == 1:
#                 predicted_indices = predicted_indices[:target_tensor.shape[0]]
#             else:
#                 predicted_indices = predicted_indices[:, :target_tensor.shape[0]]
            
#             # Pad predicted_indices if it's shorter than target_tensor
#             if predicted_indices.shape[0] < target_tensor.shape[0]:
#                 padding = torch.full((target_tensor.shape[0] - predicted_indices.shape[0],), 
#                                      vocab['<PAD>'], 
#                                      device=device)
#                 predicted_indices = torch.cat([predicted_indices, padding])
            
#             # Compare the non-padded part of the prediction with the target
#             mask = (target_tensor != vocab['<PAD>']) & (target_tensor != vocab['<START>']) & (target_tensor != vocab['<END>'])
#             correct += (predicted_indices[mask] == target_tensor[mask]).all().item()
#             total += 1
    
#     return correct / total

# # Use the function like this:
# test_accuracy = evaluate_full_dates(model, x_test, y_test, date_vocab, device)
# print(f"Full date conversion accuracy on test set: {test_accuracy:.4f}")


def date_to_indices(date_string, vocab, max_length):
    """ Convert a date string to a sequence of indices. Indics are padded to max_length. """
    tokens = [vocab.get(char, vocab['<UNK>']) for char in date_string]
    tokens = [vocab['<START>']] + tokens + [vocab['<END>']]
    padding = [vocab['<PAD>']] * (max_length - len(tokens))
    return tokens + padding

