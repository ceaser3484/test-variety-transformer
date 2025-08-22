import os.path

from main import make_collate_fn
import pandas as pd
import numpy as np
import torch
from model import Linformer
import yaml
from mecab import MeCab
import openkorpos_dic
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence

class ValidationSentence(torch.utils.data.Dataset):
    def __init__(self, dataset, vocab):


        self.mecab = MeCab(dictionary_path=openkorpos_dic.DICDIR)
        self.dataset = dataset
        self.vocab = vocab

    def __getitem__(self, item):
        sentence = self.dataset[item]
        return [self.vocab[token] for token in self.mecab.morphs(sentence)] + [self.vocab['<eos>']]


    def __len__(self):
        return len(self.dataset)



def make_collate_fn(max_len):
    def collate_fn(batch):
        question_list = []

        for question in batch:
            question_list.append(torch.tensor(question))


        temp = pad_sequence(question_list, batch_first=True)

        #
        # # sequence's length is made longer to max_len
        seq_length = temp.size(1)
        if seq_length < max_len:
            needed_seq_length = max_len - seq_length
            temp = pad(temp, (0, needed_seq_length, 0, 0), 'constant', 0)
        #
        return temp

    return collate_fn

def validation_main():
    test_data = pd.read_csv("~/DATASET/mapping_data/test.csv").iloc[:, 1].values
    vocab = torch.load("../../pickles/vocab.pt")


    with open("hyper-parameter.yaml") as f:
        hyper_parameter = yaml.full_load(f)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model = Linformer(hyper_parameter, len(vocab), device)

    if os.path.isfile("../../models/linformer.pth"):
        check_point = torch.load("../../models/linformer.pth")
        model.load_state_dict(check_point['model_state'])
    else:
        print('you should have trained model')
        exit()


    validation_dataset = ValidationSentence(test_data, vocab)
    collate_fn = make_collate_fn(hyper_parameter['max_len'])
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for sentence in validation_dataloader:
            predicted_tensor = model(sentence)
            predicted_sentence = torch.argmax(predicted_tensor, dim=-1).squeeze().tolist()
            # print(predicted_sentence.size())
            print(vocab.lookup_tokens(predicted_sentence))






if __name__ == '__main__':
    validation_main()