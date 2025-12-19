import time
import torch
from torch import nn
from tqdm import tqdm

def accuracy_top1(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()

def evaluate(model, loader, device):
    model.eval()
    total_acc = 0.0
    total_loss = 0.0
    n_batches = 0
    ce = nn.CrossEntropyLoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        total_loss += loss.item()
        total_acc += accuracy_top1(logits, y)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch_supervised(model, loader, optimizer, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_top1(logits, y)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)

def epoch_time(start_t):
    return time.time() - start_t
