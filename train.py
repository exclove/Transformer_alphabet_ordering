import torch
import torch.nn as nn
import torch.optim as optim
from main import get_config
from main import Transformer
from data_preparation import generate_alphabet_data, build_vocab_alphabet, preprocess_alphabet_data
from global_name_space import get_config


# Get arguments from globalnamespace.py
parser = get_config()
args = parser.parse_args()


# Device setting
if torch.cuda.is_available():
    device = torch.device("cuda") 
    print("CUDA is available")

else :
    print("You can only use CPU")
    device = torch.device("cpu")
    

data = generate_alphabet_data()
vocab = build_vocab_alphabet()
inputs, targets = preprocess_alphabet_data(data, vocab)

vocab_size = len(vocab)
embed_size = args.embed_size
num_layers=args.num_layers
forward_expansion=args.forward_expansion
heads=args.heads
dropout=args.dropout
device= device
max_length=args.max_length
learning_rate = args.lr
num_epochs = args.num_epochs
batch_size = args.batch_size

# 모델 초기화
model = Transformer(
    src_vocab_size=vocab_size,
    trg_vocab_size=vocab_size,
    src_pad_idx=vocab["<pad>"],
    trg_pad_idx=vocab["<pad>"],
    embed_size=embed_size,
    num_layers=num_layers,
    forward_expansion=forward_expansion,
    heads=heads,
    dropout=dropout,
    max_length=max_length,
    device=device
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

# 데이터 로더 설정
dataset = torch.utils.data.TensorDataset(inputs, targets)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch_idx, (src, trg) in enumerate(loader):
        # Move data to the selected device
        src = src.to(device)
        trg_input = trg[:, :-1].to(device)  # Remove last token for decoder input
        trg_output = trg[:, 1:].to(device)  # Remove first token for target

        # Forward pass
        output = model(src, trg_input)  # (batch_size, seq_len, vocab_size)

        # Reshape for loss calculation
        output = output.reshape(-1, vocab_size)  # Flatten output
        trg_output = trg_output.reshape(-1)  # Flatten target

        # Compute loss
        loss = criterion(output, trg_output)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Log epoch loss
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(loader):.4f}")

    # Save the model every 5 epochs
    if (epoch + 1) % 5 == 0:
        model_path = f"model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Checkpoint saved at {model_path}")



