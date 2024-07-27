import torch
from training.convertdates import create_date_vocab, date_to_indices, indices_to_date
from training.dateformattransformer import DateFormatTransformer

# Load the vocabulary and model parameters
date_vocab = create_date_vocab()
max_length = 20  

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

# Load the best model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

def convert_date(model, input_date, vocab, max_length):
    model.eval()
    with torch.no_grad():
        input_indices = torch.tensor([date_to_indices(input_date, vocab)]).to(device)
        if input_indices.size(1) < max_length:
            padding = torch.full((1, max_length - input_indices.size(1)), vocab['<PAD>'], device=device)
            input_indices = torch.cat([input_indices, padding], dim=1)
        
        output = torch.tensor([[vocab['<START>']]], device=device)
        for _ in range(max_length):
            predictions = model(input_indices, output)
            next_char = predictions[0, -1].argmax()
            if next_char == vocab['<END>']:
                break
            output = torch.cat([output, next_char.unsqueeze(0).unsqueeze(0)], dim=1)
        
        return indices_to_date(output.squeeze().tolist(), vocab)

# Test examples
test_inputs = [
    "2023-07-26",
    "15/08/1947",
    "Dec 25, 1999",
    "1985-10-21",
    "07/04/2022",
    "November 11, 2011",
    "22 Jan 2000",
    "1969/07/20",
    "April 1, 2025",
    "31-12-2020"
]

print("Testing the model on various date formats:")
for test_input in test_inputs:
    converted_date = convert_date(model, test_input, date_vocab, max_length)
    print(f"Input: {test_input} -> Output: {converted_date}")
    
    
# print("Vocabulary:")
# for token, index in date_vocab.items():
#     print(f"{token}: {index}")