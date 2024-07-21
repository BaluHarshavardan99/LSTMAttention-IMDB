import torch
import torch.optim as optim
import torch.nn as nn
from model import LSTMWithAttention
from data_loader import load_data
from train import train, evaluate

BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
N_EPOCHS = 2
CLIP = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
vocab, train_dataloader, valid_dataloader, test_dataloader = load_data(BATCH_SIZE, device)

# Initialize model
model = LSTMWithAttention(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
model = model.to(device)

# Define optimizer and criterion
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)

# Training loop
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, CLIP)
    valid_loss, valid_acc = evaluate(model, valid_dataloader, criterion)

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

# Evaluate on the test set
test_loss, test_acc = evaluate(model, test_dataloader, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
