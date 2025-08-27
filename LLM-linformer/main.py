    
import pandas as pd
import torch
import torch.nn as nn
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
import matplotlib.pyplot as plt
import pickle
from torch.cuda.amp import autocast, GradScaler


def create_vocab(data, min_freq=100):
    import glob
    from collections import Counter
    import word_process as wp
    import openkorpos_dic
    pbar = tqdm(data)

    user_dict = glob.glob("../../mecab-dict/*.dic")
    mecab = MeCab(dictionary_path=openkorpos_dic.DICDIR, user_dictionary_path=user_dict)
    tokens_count = Counter()
    tokens_list = []
    vocab = {'<pad>':0, '<sos>':1, '<eos>':2, '<unk>':3, '<current>':4, '<time>':5, '<date>':6, '<NUM>':7, '<answer>':8}

    for sentence in pbar:
        pbar.set_description(f"vocab is creating: ")
        preprocessed_sentence = wp.replace_currency(sentence, "<current>")
        preprocessed_sentence = wp.replace_time(preprocessed_sentence, "<time>")
        preprocessed_sentence = wp.replace_date(preprocessed_sentence, '<date>')
        preprocessed_sentence = wp.replace_usual_num(preprocessed_sentence, '<NUM>')
        for morph, pos in mecab.pos(preprocessed_sentence):
            if morph in vocab.keys():
                tokens_list.append(morph)
            else:
                tokens_list.append(f"{morph}/{pos}")
    tokens_count.update(tokens_list)
    current_idx = len(tokens_count)
    frequently_sorted_token = sorted(tokens_count.items(), key=lambda x: x[1], reverse=True)
    filtered_tokens = []
    for token, count in frequently_sorted_token:

        if token in vocab.keys(): # 중복이 없기 위해서 vocab내에 key가 있다면.. 넘어가!
            continue

        if count >= min_freq:
            filtered_tokens.append(token)

    for token in filtered_tokens:
        vocab[token] = len(vocab)

    return vocab


def make_collate_fn():
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
        # if seq_length < max_len:
        #     needed_seq_length = max_len - seq_length
        #     temp = pad(temp, (0, needed_seq_length, 0, 0), 'constant', 0)

        question_tensors = temp[:num_questions, :]
        answer_tensors = temp[num_questions:, :]

        return question_tensors, answer_tensors

    return collate_fn


def training_loop(dataLoader, model, criterion, optimizer, device, fold_idx, epoch, scheduler):
    model.train()

    loss_list = []
    scaler = GradScaler()
    accumulation_step = 16

    print(f"\nIn {epoch + 1}, learning rate is ", scheduler.get_last_lr()[0])
    pbar = tqdm(dataLoader, ascii=' =')
    for idx, (question, answer) in enumerate(pbar):
        pbar.set_description(f"{fold_idx} fold | {epoch + 1} epoch")
        # optimizer.zero_grad()
        question = question.to(device)
        answer = answer.to(device)

        with autocast():

            predict = model(question)
            num_token = predict.size(2)
            predict = predict.view(-1, num_token)
            answer = answer.reshape(-1)

            loss = criterion(predict, answer)
        
        loss = loss / accumulation_step

        # 3. 역전파를 수행하여 경사 누적
        scaler.scale(loss).backward()

        if (idx + 1) % accumulation_step == 0 or (idx + 1) == len(dataLoader):

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=4)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() # 경사 초기화

        if (idx + 1) % 100000 == 0:
            torch.save({"model_state": model.state_dict(),
                                "optimizer_state": optimizer.state_dict(),
                                "scheduler_state": scheduler.state_dict()}, "../../models/processing_linformer.pth")
            print("best model is saving\n")

        batch_correct = (answer == torch.argmax(predict, dim=-1)).float().sum().item()
        batch_total = answer.size(0)
        batch_accuracy = 100 * (batch_correct / batch_total)
        pbar.set_postfix({'loss': float(loss) * accumulation_step,
                          'acc': f"{batch_accuracy :.2f}" ,
                          'lr':f"{scheduler.get_last_lr()[0]:13f}"})
        loss_list.append(loss.item() * accumulation_step)
        # scheduler.step()

    print(f"\nIn this epoch", '.' * 10)
    loss_list = sorted(loss_list)
    median_idx = len(loss_list) // 2
    print(f"average loss is {sum(loss_list) / len(loss_list)}\n\n"
          f"minimum loss is {loss_list[0]}\n"
          f"median loss is {loss_list[median_idx]}\n"
          f"maximum loss is {loss_list[-1]}\n")


    scheduler.step()
    return sum(loss_list) / len(loss_list)

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
    return sum(loss_list) / len(loss_list)


def test_loop(testLoader, model, criterion, device, vocab):
    model.eval()

    with torch.no_grad():
        for question, answer in testLoader:
            show_question = question.squeeze(0).tolist()
            print(vocab.lookup_tokens(show_question))
            # reverse_question_sentence = [reverse_vocab[word] for word in show_question]
            # print(reverse_question_sentence)

            show_answer = answer.squeeze(0).tolist()
            print(vocab.lookup_tokens(show_answer))
            # reverse_answer_sentence = [reverse_vocab[word] for word in show_answer]
            # print(reverse_answer_sentence)

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
            # predicted_sentence = [reverse_vocab[word] for word in predicted_sentence]
            # print(predicted_sentence)


def train_main() -> None:
    from random import choices
    from glob import glob

    torch.manual_seed(999)
    torch.cuda.manual_seed_all(999)

    
    pre_dataset = []
    txt_set = glob('*.txt')
    for txt in txt_set:
        with open(txt, 'r') as f:
            pre_dataset += [sentence.strip('\n') for sentence in f.readlines()]

    # get vocab
    if not os.path.isfile("../../pickles/vocab.pth"):
        vocab = create_vocab(pre_dataset)
        torch.save(vocab, "../../pickles/vocab.pth")
    else:

        vocab = torch.load("../../pickles/vocab.pth")

    reverse_vocab = dict((value, key) for key, value in vocab.items())

    with open("hyper-parameter.yaml") as f:
        hyper_parameter = yaml.full_load(f)

    collate_fn = make_collate_fn()
    batch_size = hyper_parameter['batch_size']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    is_loaded_model = False
    # device = 'cpu'

    model = Linformer(hyper_parameter, num_vocab=len(vocab), device=device).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = torch.optim.SGD(model.parameters(), lr=hyper_parameter['learning_rate'], momentum=0.99, nesterov=True, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hyper_parameter['scheduler_step_size'],
                                                gamma=hyper_parameter['scheduler_gamma'])


    if os.path.isfile("../../models/processing_linformer.pth"):
        check_point = torch.load("../../models/linformer.pth")
        model.load_state_dict(check_point['model_state'])
        optimizer.load_state_dict(check_point['optimizer_state'])
        scheduler.load_state_dict(check_point['scheduler_state'])
        is_loaded_model = False
        print("saved model file is found")

    elif os.path.isfile("../../models/best_linformer.pth"):
        check_point = torch.load("../../models/linformer.pth")
        model.load_state_dict(check_point['model_state'])
        optimizer.load_state_dict(check_point['optimizer_state'])
        scheduler.load_state_dict(check_point['scheduler_state'])
        is_loaded_model = False
        print("saved model file is found")
    
    else:
        print("newly training")


    for fold_idx in range(hyper_parameter['num_fold']):
        train_dataset = []
        valid_dataset = []
        valid_indexes = choices(range(len(pre_dataset)), k=10)
        for idx in range(len(pre_dataset)):
            if idx in valid_indexes:
                valid_dataset.append(pre_dataset[idx])
            else:
                train_dataset.append(pre_dataset[idx])


        train_dataset = SentenceDataset(train_dataset, vocab, hyper_parameter['max_len'], 'train')
        valid_dataset = SentenceDataset(valid_dataset, vocab, hyper_parameter['max_len'], 'validation')

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=hyper_parameter['shuffle'],
                                      num_workers=6, pin_memory=True, collate_fn=collate_fn)

        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        train_loss_list = []
        validation_loss_list = []
        lowest_loss = 9999999
        lowest_epoch = None
        # patience = 20

        for epoch in range(hyper_parameter['num_epochs']):

            train_loss = training_loop(train_dataloader, model, criterion, optimizer, device, fold_idx, epoch, scheduler)
            train_loss_list.append(train_loss)
            val_loss = valadation_loop(valid_dataloader, model, criterion, device, fold_idx)
            validation_loss_list.append(val_loss)

            
            if (val_loss <= lowest_loss) and (fold_idx >= 1 or is_loaded_model):
                lowest_loss = val_loss
                lowest_epoch = epoch

                torch.save({"model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict()}, "../../models/best_linformer.pth")
                print("best model is saving\n")

            else:

                if hyper_parameter['patience'] > 0 and lowest_epoch + hyper_parameter['patience'] < epoch + 1:
                    print(f"In {epoch} epoch, model is not better\nNext fold is coming")
                    break

        plt.plot(train_loss_list, label='train loss')
        plt.plot(validation_loss_list, label='val loss')
        plt.legend()
        plt.grid()
        plt.savefig(f"../../observation/{fold_idx}_fold_linformer.jpg")

        torch.save({"model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict()}, "../../models/last_linformer.pth")
        print("last model is saving\n")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_main()
