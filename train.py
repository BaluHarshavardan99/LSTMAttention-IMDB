# import torch
# import torch.optim as optim
# import torch.nn as nn
# from utils import binary_accuracy, save_checkpoint, load_checkpoint
# from tqdm import tqdm

# def train(model, iterator, optimizer, criterion, clip, scheduler):
#     epoch_loss = 0
#     epoch_acc = 0
#     model.train()
    
#     for batch in tqdm(iterator):
#         optimizer.zero_grad()
#         text, labels, lengths = batch
#         predictions = model(text, lengths).squeeze(1)
#         loss = criterion(predictions, labels)
#         acc = binary_accuracy(predictions, labels)
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), clip)
#         optimizer.step()
#         epoch_loss += loss.item()
#         epoch_acc += acc.item()
    
#     scheduler.step()
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)

# def evaluate(model, iterator, criterion):
#     epoch_loss = 0
#     epoch_acc = 0
#     model.eval()
    
#     with torch.no_grad():
#         for batch in tqdm(iterator):
#             text, labels, lengths = batch
#             predictions = model(text, lengths).squeeze(1)
#             loss = criterion(predictions, labels)
#             acc = binary_accuracy(predictions, labels)
#             epoch_loss += loss.item()
#             epoch_acc += acc.item()
        
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)



import torch
import torch.nn as nn

def train(model, dataloader, optimizer, criterion, clip, scheduler=None):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for text, labels, lengths in dataloader:
        optimizer.zero_grad()

        predictions = model(text, lengths).squeeze(1)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if scheduler is not None:
            scheduler.step()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for text, labels, lengths in dataloader:
            predictions = model(text, lengths).squeeze(1)
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    # Calculate the average loss and accuracy manually
    num_batches = len(dataloader)
    return epoch_loss / num_batches, epoch_acc / num_batches

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc
