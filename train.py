import torch
import torch.nn as nn

def train(model, dataloader, optimizer, criterion, clip, scheduler=None):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0

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
        num_batches += 1

        if scheduler is not None:
            scheduler.step()

    return epoch_loss / num_batches, epoch_acc / num_batches

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0

    with torch.no_grad():
        for text, labels, lengths in dataloader:
            predictions = model(text, lengths).squeeze(1)
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            num_batches += 1

    return epoch_loss / num_batches, epoch_acc / num_batches

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc
