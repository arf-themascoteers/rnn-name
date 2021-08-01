import torch
import torch.nn as nn

import name_dataset
import rnn
from torch.utils.data import DataLoader
from name_dataset import NameDataset
import matplotlib.pyplot as plt

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nd = NameDataset(is_train=False)
    dataloader = DataLoader(nd, batch_size=1, shuffle=True)
    model = rnn.RNN(nd.cats.shape[0]).to(device)
    model.train()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    initial_hidden = torch.zeros(20, 128).to(device)
    num_epochs = 10
    loss_records = []
    for epoch in range(num_epochs):
        for (name, language) in dataloader:
            name = name.to(device)
            output, hidden = model.process_name(name, initial_hidden)
            language = language.to(device)
            loss = criterion(output, language)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
        loss_records.append(loss.item())

    torch.save(model.state_dict(), 'models/rnn.h5')
    plt.plot(loss_records)
    plt.show()

if __name__ == "__main__":
    train()
    exit(0)