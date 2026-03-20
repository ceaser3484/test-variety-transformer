from torch.utils.data import Dataset

class PreDataset(Dataset):

    def __init__(self, data):
        super(PreDataset, self).__init__()
        self.data = data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sentence = self.data[item]
        train_input = sentence[:-1]
        train_target = sentence[1:]
        return train_input, train_target
