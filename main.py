import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from glob import glob
from sklearn.model_selection import train_test_split

from model import LipNet
from dataset import LipNetDataset, collate_fn
from utils import download_dataset, download_dat

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()

    train_loss = 0

    for X, y, input_length, target_length in tqdm(dataloader, desc = "Training Batch"):

        X, y, input_length, target_length = X.to(device), y.to(device), input_length.to(device), target_length.to(device)

        y_pred = model(X)
        y_pred_sm = F.log_softmax(y_pred.permute(1,0,2), dim = -1)
        loss = loss_fn(y_pred_sm, y, input_length, target_length)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)

    return train_loss

def train(model, dataloader, loss_fn, optimizer, epochs, device):
    results = []

    for epoch in tqdm(range(epochs), desc = "Epoch"):
        train_loss = train_step(model, dataloader, loss_fn, optimizer, device)

        print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f}")

        results.append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)

    return results

def save_model(model, optimizer, save_path, save_name):
    outfile = os.path.join(save_path, f'{save_name}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, outfile
    )
    print(f'Saved model to {outfile}')

def main():
    # Download data
    download_dataset()
    download_dat()

    # Set hyperparameters
    batch_size = 2
    num_workers = os.cpu_count()
    epochs = 100
    lr = 1e-4

    # Load Data
    data = glob("/data/*.mpg")

    train, test = train_test_split(data, test_size = 255, random_state = 42)

    vocab = [x for x in "-|abcdefghijklmnopqrstuvwxyz"]
    train_set = LipNetDataset(train, vocab)

    dataloader = DataLoader(dataset=train_set,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            collate_fn=collate_fn)

    # Train
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = LipNet(len(vocab))

    loss_fn = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    results = train(model, dataloader, loss_fn, optimizer, epochs, device)

    # save
    save_model(model, optimizer, 'model', 'lipnet')

    pass