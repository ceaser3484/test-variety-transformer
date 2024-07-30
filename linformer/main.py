# from torchtext import disable_torchtext_deprecation_warning
# disable_torchtext_deprecation_warning()

import pandas as pd
import torch
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator
from mecab import MeCab
from sklearn.model_selection import KFold
from Dataset import SentenceDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import yaml
from tqdm import tqdm
import os
from model import Linformer
import numpy as np
import openkorpos_dic


def create_vocab(data) -> iter:
    mecab = MeCab(dictionary_path=openkorpos_dic.DICDIR)

    for column in data:
        for sentence in column:
            yield mecab.morphs(sentence)


def update_vocab(vocab) -> iter:
    test_data = pd.read_csv("~/DATASET/mapping_data/test.csv")
    mecab = MeCab(dictionary_path=openkorpos_dic.DICDIR)
    test_data.drop('id', axis=1, inplace=True)
    test_data = test_data.values
    test_data = test_data.squeeze(1)

    for sentence in test_data:
        for word in mecab.morphs(sentence):
            if not vocab.__contains__(word):
                vocab.append_token(word)

    return vocab


def make_collate_fn(max_len):
    def collate_fn(batch):
        question_list = []
        answer_list = []

        for question, answer in batch:
            question_list.append(torch.tensor(question))
            answer_list.append(torch.tensor(answer))

        num_questions = len(question_list)
        temp = question_list + answer_list
        temp = pad_sequence(temp, batch_first=True)

        # sequence's length is made longer to max_len
        seq_length = temp.size(1)
        if seq_length < max_len:
            needed_seq_length = max_len - seq_length
            temp = pad(temp, (0, needed_seq_length, 0, 0), 'constant', 0)

        question_tensors = temp[:num_questions, :]
        answer_tensors = temp[num_questions:, :]

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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
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

        if (epoch + 1) % 2 == 0:
            torch.save({"model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict()}, "../../models/linformer.pth")
            print("model is saving\n")
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

def test_loop(testLoader, model, criterion, device, vocab):
    model.eval()

    with torch.no_grad():
        for question, answer in testLoader:
            show_question = question.squeeze(0).tolist()
            print(vocab.lookup_tokens(show_question))

            show_answer = answer.squeeze(0).tolist()
            print(vocab.lookup_tokens(show_answer))

            question = question.to(device)
            answer = answer.to(device)

            predicted = model(question)
            num_token = predicted.size(2)
            predicted = predicted.view(-1, num_token)

            answer = answer.reshape(-1)

            loss = criterion(predicted, answer)
            print(f"In this sentence's loss is {loss.item()}\n"
                  f"answer sentence is below", '.' * 20)

            predicted_sentence = torch.argmax(predicted, dim=1).tolist()
            print(vocab.lookup_tokens(predicted_sentence))


def train_main() -> None:
    chat_data = pd.read_csv("~/DATASET/mapping_data/train.csv")
    chat_data.drop(['id', 'category'], axis=1, inplace=True)

    pre_train_data = pd.DataFrame()
    for q_idx in range(2):
        # training data's question column index
        question_idx = q_idx + 1

        # set temporary DataFrame
        temp_dataframe = pd.DataFrame()

        for ans_idx in range(5):
            # training data's answer column index
            answer_idx = ans_idx + 1
            temp_dataframe['question'] = chat_data[f"질문_{question_idx}"]
            temp_dataframe['answer'] = chat_data[f'답변_{answer_idx}']

            # concatenate
            pre_train_data = pd.concat([pre_train_data, temp_dataframe])

    del chat_data

    pre_train_data = pre_train_data.values
    # get vocab
    if not os.path.isfile("../../pickles/vocab.pt"):
        vocab = build_vocab_from_iterator(create_vocab(pre_train_data), specials=['<pad>', '<eos>'])
        vocab = update_vocab(vocab)
        torch.save(vocab, "../../pickles/vocab.pt")
    else:
        vocab = torch.load("../../pickles/vocab.pt")

    with open("hyper-parameter.yaml") as f:
        hyper_parameter = yaml.full_load(f)

    collate_fn = make_collate_fn(hyper_parameter['max_len'])
    batch_size = hyper_parameter['batch_size']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    model = Linformer(hyper_parameter, num_vocab=len(vocab), device=device).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = torch.optim.RAdam(model.parameters(), lr=hyper_parameter['learning_rate'])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hyper_parameter['scheduler_step_size'],
    #                                             gamma=hyper_parameter['scheduler_gamma'])
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.000001, max_lr=hyper_parameter['learning_rate'],
                                                  step_size_up=hyper_parameter['step_size'])

    if os.path.isfile("../../models/linformer.pth"):
        check_point = torch.load("../../models/linformer.pth")
        model.load_state_dict(check_point['model_state'])
        optimizer.load_state_dict(check_point['optimizer_state'])
        scheduler.load_state_dict(check_point['scheduler_state'])
        print("saved model file is found")

    kfold = KFold(n_splits=hyper_parameter['num_fold'])

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(pre_train_data)):
        train_data = pre_train_data[train_idx]
        val_data = pre_train_data[val_idx]

        random_choice = np.random.choice(val_data.shape[0], 2)
        test_data = pre_train_data[random_choice, :]

        train_dataset = SentenceDataset(train_data, vocab)
        val_dataset = SentenceDataset(val_data, vocab)
        test_dataset = SentenceDataset(test_data, vocab)

        train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=hyper_parameter['shuffle'],
                                      num_workers=4, collate_fn=collate_fn)
        val_dataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        test_dataLoader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        test_loop(testLoader=test_dataLoader, model=model, criterion=criterion, device=device, vocab=vocab)
        training_loop(train_dataLoader, model, criterion, optimizer, device, fold_idx, hyper_parameter['num_epochs'], scheduler)
        valadation_loop(val_dataLoader, model, criterion, device, fold_idx)

        if fold_idx % 2 == 0:
            torch.save({"model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict()}, "../../models/linformer.pth")
            print("model is saving\n")

    # os.system('shutdown -')
    # for question, answer in dataLoader:
    #     print(question)
    #     print(answer)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_main()
