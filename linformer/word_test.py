from mecab import MeCab
import pandas as pd
import openkorpos_dic
from time import sleep


def sentence_parsing_test():
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

    longest_vocab = 0
    pre_train_data = pre_train_data.values
    mecab = MeCab(dictionary_path=openkorpos_dic.DICDIR)
    for i in pre_train_data:
        for j in i:
            # sleep(5)
            num_vocab = len(mecab.morphs(j))

            if num_vocab > longest_vocab:
                longest_vocab = num_vocab

    print(longest_vocab)



if __name__ == '__main__':
    sentence_parsing_test()