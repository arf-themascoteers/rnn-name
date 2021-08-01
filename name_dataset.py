from torch.utils.data import Dataset
import os
import unicodedata
import string
import io
import torch

class NameDataset(Dataset):
    def __init__(self, is_train):
        self.is_train = is_train
        self.ALL_LETTERS = string.ascii_letters + " .,;'"
        self.N_LETTERS = len(self.ALL_LETTERS)
        name_dict = {}
        for (dirpath, dirnames, filenames) in os.walk("data"):
            names = []
            for filename in filenames:
                file_path = os.path.join("data", filename)
                lines = io.open(file_path, encoding='utf-8').read().strip().split('\n')
                for line in lines:
                    line = self._unicode_to_ascii(line)
                    names.append(line)
                language = filename[0:filename.index(".")]
                name_dict[language] = names

        languages = list(name_dict.keys())
        self.cats = torch.eye(len(languages))

        self.data = []
        for cat in name_dict:
            for i in range(len(name_dict[cat])):
                index = languages.index(cat)
                cat_tensor = self.cats[index]
                tup = self.name_to_tensor(name_dict[cat][i]), cat_tensor
                if self.is_train:
                    if i%10 != 0:
                        self.data.append(tup)
                else:
                    if i%10 == 0:
                        self.data.append(tup)

    def letter_to_tensor(self, letter):
        tensor = torch.zeros(1, self.N_LETTERS)
        tensor[0][self.ALL_LETTERS.index(letter)] = 1
        return tensor

    def name_to_tensor(self, name):
        tensor = torch.zeros(len(name), 1, self.N_LETTERS)
        for i, letter in enumerate(name):
            tensor[i][0][self.ALL_LETTERS.index(letter)] = 1
        return tensor

    def _unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.ALL_LETTERS
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = list(self.data[idx])
        return x[0], x[1]


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    nd = NameDataset(is_train=False)
    dataloader = DataLoader(nd, batch_size=3, shuffle=True)

    for name, language in dataloader:
        print(name)
        print(language)
        break
