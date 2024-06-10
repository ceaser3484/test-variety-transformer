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
from linformer import LinformerLM
from tqdm import tqdm


def create_vocab(data) -> iter:
    mecab = MeCab()
    for column in data:
        for sentence in column:
            yield mecab.morphs(sentence)


def update_vocab(vocab) -> iter:
    test_data = pd.read_csv("~/DATASET/mapping_data/test.csv")
    mecab = MeCab()
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
        pbar = tqdm(dataLoader)
        print(f"\nIn {epoch + 1}, learning rate is ", scheduler.get_last_lr())

        for question, answer in pbar:
            pbar.set_description(f"{fold_idx} fold | {epoch + 1} epoch")
            optimizer.zero_grad()
            question = question.to(device)
            answer = answer.to(device)

            predict = model(question)
            num_token = predict.size(2)
            predict = predict.view(-1, num_token)
            answer = answer.view(-1)

            loss = criterion(predict, answer)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
            loss_list.append(loss.item())

        print(f"In this epoch", '.' * 10)
        loss_list = sorted(loss_list)
        median_idx = len(loss_list) // 2
        print(f"minimum loss is {loss_list[0]}\n"
              f"median loss is {loss_list[median_idx]}\n"
              f"maximum loss is {loss_list[-1]}\n")
        scheduler.step()


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
            answer = answer.view(-1)

            loss = criterion(predicted, answer)
            pbar.set_postfix({'loss': loss.item()})
            loss_list.append(loss.item())

        print(f"In this fold", '.' * 10)
        loss_list = sorted(loss_list)
        median_idx = len(loss_list) // 2
        print(f"minimum loss is {loss_list[0]}\n"
              f"median loss is {loss_list[median_idx]}\n"
              f"maximum loss is {loss_list[-1]}\n")

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
    vocab = build_vocab_from_iterator(create_vocab(pre_train_data), specials=['<pad>', '<eos>'])
    vocab = update_vocab(vocab)

    with open("hyper-parameter.yaml") as f:
        hyper_parameter = yaml.full_load(f)

    collate_fn = make_collate_fn(hyper_parameter['max_len'])
    batch_size = hyper_parameter['batch_size']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LinformerLM(num_tokens=len(vocab),
                        dim=128,
                        seq_len=hyper_parameter['max_len'],
                        depth=12,
                        dropout=0.1
                        ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_parameter['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)

    kfold = KFold(n_splits=hyper_parameter['num_fold'])

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(pre_train_data)):
        train_data = pre_train_data[train_idx]
        val_data = pre_train_data[val_idx]

        train_dataset = SentenceDataset(train_data, vocab)
        val_dataset = SentenceDataset(val_data, vocab)

        train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
        val_dataLoader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

        training_loop(train_dataLoader, model, criterion, optimizer, device, fold_idx, hyper_parameter['num_epochs'], scheduler)
        valadation_loop(val_dataLoader, model, criterion, device, fold_idx)


    # for question, answer in dataLoader:
    #     print(question)
    #     print(answer)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_main()
