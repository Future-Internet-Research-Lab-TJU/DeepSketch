import matplotlib.pyplot as plt
from seq2seq_atten import Seq2Seq
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import time
from tqdm import tqdm
import data_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters

batch_size = 30000
n_epochs = 300
lr = 0.005

### Train
def train(model, n_epoch, training_data, loss_fn, optimizer):
    for epoch in tqdm(range(n_epochs)):
        total_loss = 0.0
        for batch, (X, y) in enumerate(training_data):
            X, y = X.to(device), y.to(device)
            model.train()
            pred = model(X)
            y = y.view(-1,1)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss = total_loss / len(training_data)
        res_e = 'Epoch: [{}/{}], training loss: {:6.10f}'.format(epoch, n_epochs, total_loss)
        tqdm.write(res_e)
    return model

if __name__ == '__main__':
    torch.manual_seed(3)
    file_name = './test_data/dateset1.csv'
    training_data, _, _, _, _ = data_loader.training_data(path=file_name, batch_size=batch_size)
    model = Seq2Seq().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    print("model trained")
    model = train(model=model, n_epoch=n_epochs, training_data=training_data, loss_fn=loss_fn, optimizer=optimizer)
    torch.save(model,'seq2seq.pth')
    print("model saved")
