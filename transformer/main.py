import pandas as pd
import torch
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import KFold
import spacy
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import yaml
from model import Transformer
from tqdm import tqdm
import os
import numpy as np
from Dataset import TranslationDataset



def create_eng_vocab(data) -> iter:
    eng_spacy = spacy.load("en_core_web_sm")
    pbar = tqdm(data)
    for sentence in pbar:
        sentence = sentence.lower()
        yield [word.text for word in eng_spacy(sentence)]


def create_ger_vocab(data) -> iter:
    ger_spacy = spacy.load("de_core_news_sm")
    pbar = tqdm(data)
    for sentence in pbar:
        sentence = sentence.lower()
        yield [word.text for word in ger_spacy(sentence)]


def make_collate_fn(max_len):
    def collate_fn(batch):
        question_list = []
        answer_list = []

        for question, answer in batch:
            question_list.append(torch.tensor(question))
            answer_list.append(torch.tensor(answer))

        num_questions = len(question_list)
        temp = question_list + answer_list
        temp = pad_sequence(temp)

        # sequence's length is made longer to max_len
        seq_length = temp.size(0)
        if seq_length < max_len:
            needed_seq_length = max_len - seq_length
            temp = pad(temp, (0, 0, 0, needed_seq_length), 'constant', 0)

        question_tensors = temp[:, :num_questions]
        answer_tensors = temp[:, num_questions:]

        return question_tensors, answer_tensors

    return collate_fn


def training_loop(dataLoader, model, criterion, optimizer, device, fold_idx, num_epochs, scheduler):
    model.train()

    for epoch in range(num_epochs):
        loss_list = []
        print(f"\nIn {epoch + 1}, learning rate is ", scheduler.get_last_lr())
        pbar = tqdm(dataLoader)

        for question, answer in pbar:
            pbar.set_description(f"{fold_idx} fold | {epoch + 1} epoch")
            optimizer.zero_grad()
            question = question.to(device)
            answer = answer.to(device)

            predict = model(question)
            num_token = predict.size(2)
            predict = predict.view(-1, num_token)
            answer = answer.reshape(-1)

            loss = criterion(predict, answer)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item(), 'lr':f"{scheduler.get_last_lr()[0]:5f}"})
            loss_list.append(loss.item())
            scheduler.step()

        print(f"\nIn this epoch", '.' * 10)
        loss_list = sorted(loss_list)
        median_idx = len(loss_list) // 2
        print(f"average loss is {sum(loss_list) / len(loss_list)}\n\n"
              f"minimum loss is {loss_list[0]}\n"
              f"median loss is {loss_list[median_idx]}\n"
              f"maximum loss is {loss_list[-1]}\n")
        # scheduler.step()


def valadation_loop(dataLoader, model, criterion, device, fold_idx):
    model.eval()
    pbar = tqdm(dataLoader)
    loss_list = []
    with torch.no_grad():
        for question, answer in pbar:

            pbar.set_description(f"{fold_idx} fold evaluation")

            question = question.to(device)
            answer = answer.to(device)

            predicted = model(question)
            num_token = predicted.size(2)
            predicted = predicted.view(-1, num_token)
            answer = answer.reshape(-1)

            loss = criterion(predicted, answer)
            pbar.set_postfix({'loss': loss.item()})
            loss_list.append(loss.item())

        print(f"\nIn this fold", '.' * 10)
        loss_list = sorted(loss_list)
        median_idx = len(loss_list) // 2
        print(f"average loss is {sum(loss_list) / len(loss_list)}\n\n"
              f"minimum loss is {loss_list[0]}\n"
              f"median loss is {loss_list[median_idx]}\n"
              f"maximum loss is {loss_list[-1]}\n")

def train_main() -> None:
    chat_data = pd.read_csv("~/DATASET/deu.txt", sep='\t').values
    # chat_data.drop(['id', 'category'], axis=1, inplace=True)
    chat_data = np.delete(chat_data, 2, axis=1)

    # get vocab
    if os.path.isfile("../../pickles/eng_vocab.pt") and os.path.isfile("../../pickles/ger_vocab.pt"):
        print("saved vocab is existed")
        eng_vocab = torch.load("../../pickles/eng_vocab.pt")
        ger_vocab = torch.load("../../pickles/ger_vocab.pt")

    else:

        eng_vocab = build_vocab_from_iterator(iterator=create_eng_vocab(chat_data[:, 0].tolist()),
                                              specials=['<pad>', '<eos>'])
        ger_vocab = build_vocab_from_iterator(create_ger_vocab(chat_data[:, 1].tolist()), specials=['<pad>', '<eos>'])
        torch.save(eng_vocab, "../../pickles/eng_vocab.pt")
        torch.save(ger_vocab, "../../pickles/ger_vocab.pt")

    with open("hyper-parameter.yaml") as f:
        hyper_parameter = yaml.full_load(f)

    collate_fn = make_collate_fn(hyper_parameter['max_len'])
    batch_size = hyper_parameter['batch_size']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    model = Transformer(src_vocab_size=len(eng_vocab),
                        trg_vocab_size=len(ger_vocab),
                        max_len=hyper_parameter['max_len'],
                        hidden_dim=64,
                        n_heads=8,
                        n_stack=2,
                        src_pad_idx=eng_vocab['<pad>'],
                        trg_pad_idx=ger_vocab['<pad>'],
                        dropout=0.1,
                        device=device
                        ).to(device)

    eng_spacy = spacy.load("en_core_web_sm")
    ger_spacy = spacy.load("de_core_news_sm")

    # dataset = TranslationDataset(chat_data, eng_vocab, ger_vocab, eng_spacy, ger_spacy)

    criterion = nn.CrossEntropyLoss(ignore_index=ger_vocab['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_parameter['learning_rate'])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hyper_parameter['scheduler_step_size'],
    #                                             gamma=hyper_parameter['scheduler_gamma'])
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.000001, max_lr=hyper_parameter['learning_rate'])

    if os.path.isfile("../../models/transformer.pth"):
        check_point = torch.load("../../models/linformer.pth")
        model = model.load_state_dict(check_point['model_state'])
        optimizer = optimizer.load_state_dict(check_point['optimizer_state'])
        scheduler = scheduler.load_state_dict(check_point['scheduler_state'])

    kfold = KFold(n_splits=hyper_parameter['num_fold'])

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(pre_train_data)):
        train_data = chat_data[train_idx]
        val_data = chat_data[val_idx]

        train_dataset = TranslationDataset(train_data, eng_vocab, ger_vocab, eng_spacy, ger_spacy)
        val_dataset = TranslationDataset(val_data, eng_vocab, ger_vocab, eng_spacy, ger_spacy)

        train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=hyper_parameter['shuffle'], collate_fn=collate_fn)
        val_dataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        training_loop(train_dataLoader, model, criterion, optimizer, device, fold_idx, hyper_parameter['num_epochs'], scheduler)
        valadation_loop(val_dataLoader, model, criterion, device, fold_idx)

        if fold_idx // 2 == 0:
            torch.save({"model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict()}, "../../models/transfomer.pth")
            print("model is saving\n")

    # for question, answer in dataLoader:
    #     print(question)
    #     print(answer)


if __name__ == '__main__':
    train_main()
