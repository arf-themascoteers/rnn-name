import torch
import torch.nn as nn

import name_dataset
import rnn
from torch.utils.data import DataLoader
from name_dataset import NameDataset
import matplotlib.pyplot as plt

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nd = NameDataset(is_train=False)
    dataloader = DataLoader(nd, batch_size=2000, shuffle=True)
    model = rnn.RNN(len(nd.languages)).to(device)
    model.load_state_dict(torch.load("models/rnn.h5"))
    model.eval()

    total = 0
    correct = 0
    for (name, language) in dataloader:
        initial_hidden = torch.zeros(name.shape[0], 128).to(device)
        output, hidden = model.process_name(name, initial_hidden)
        language = language.to(device)
        total = total + name.shape[0]
        argmx = output.argmax(dim=1)
        correct = correct + (language.eq(argmx).sum())

    print(f"Total {total}")
    print(f"Correct {correct}")

if __name__ == "__main__":
    test()
    exit(0)