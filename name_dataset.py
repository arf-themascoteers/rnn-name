from torch.utils.data import Dataset
import os
import unicodedata
import string
import io
import torch

ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

class NameDataset(Dataset):
    def __init__(self, is_train):
        self.is_train = is_train


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

        self.languages = list(name_dict.keys())
        self.cats = torch.eye(len(self.languages))

        self.data = []
        for cat in name_dict:
            for i in range(len(name_dict[cat])):
                tup = name_dict[cat][i], cat
                self.data.append(tup)

        self.data_indices = None
        if is_train:
            self.data_indices = [i for i in range(len(self.data)) if i%10 != 0]
        else:
            self.data_indices = [i for i in range(len(self.data)) if i%10 == 0]

    def letter_to_tensor(self, letter):
        tensor = torch.zeros(1, N_LETTERS)
        tensor[0][ALL_LETTERS.index(letter)] = 1
        return tensor

    def name_to_tensor(self, name):
        tensor = torch.zeros(len(name), 1, N_LETTERS)
        for i, letter in enumerate(name):
            tensor[i][0][ALL_LETTERS.index(letter)] = 1
        return tensor

    def _unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in ALL_LETTERS
        )

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        index_in_full_set = self.data_indices[idx]
        x = list(self.data[index_in_full_set])
        index = self.languages.index(x[1])
        return self.name_to_tensor(x[0]), index


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    nd = NameDataset(is_train=False)
    dataloader = DataLoader(nd, batch_size=1, shuffle=True)

    for name, language in dataloader:
        print(name)
        print(language)
        break
