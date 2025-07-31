from json import JSONDecodeError
from tqdm import tqdm
from mecab import MeCab
import re
import openkorpos_dic
from glob import glob
import json


def sentence_parsing_test_1():
    word_dataset = glob("../../../DATASET/news_article_machine_translation_dataset/Training/*/*.json")
    sentence_list = []
    for json_file in word_dataset:
        print(json_file)
        with open(json_file, 'r') as f:
            json_dataset = json.load(f)

        for paragraphs in json_dataset['data']:
            for sentence in paragraphs['paragraphs']:
                sentence_splited_by_eol = [re.sub('\n', '',sentence.strip('\n').strip()) for sentence in sentence['context'].split('\n\n') if len(sentence) != 0]
            sentence_list += sentence_splited_by_eol

    with open('train_dataset.txt', 'a') as f:
        f.writelines(str(sentence) + '\n' for sentence in sentence_list)
    print('parsing_test_1 is over')

def sentence_parsing_test_2():
    word_dataset = glob("../../../DATASET/korean_book_dataset/Training/labled/*/*.json")
    sentence_list = []
    for json_file in word_dataset:
        if json_file.count('INFO') > 0:
            continue
        # print(json_file)
        with open(json_file, 'r') as f:
            json_dataset = json.load(f)
        paragraphs = json_dataset['paragraphs']
        for paragraph in paragraphs:
            print(len(paragraph['sentences']))
            # # print(paragraph['sentences'][0])
            # sentence_list.append(paragraph['sentences'][0]['text'])
            # sentence_list.append(paragraph['sentences'][0]['original_text'])
    exit()
    with open('train_dataset.txt', 'a') as f:
        f.writelines(str(sentence) + '.\n' for sentence in sentence_list)

    print('parsing_test_2 is over')

def sentence_parsing_test_3():
    word_dataset = glob("../../../DATASET/article_summerization/Training/labeld_data/*/*.json")
    sentence_list = []

    def flatten(sentences_array):
        result = []
        for item in sentences_array:
            if isinstance(item, list):
                if len(item) == 0:
                    continue
                result += flatten(item)
            else:
                result += [item]

        return result

    for json_file in tqdm(word_dataset):
        with open(json_file, 'r') as f:
            json_dataset = json.load(f)
            if isinstance(json_dataset, dict):
                for data in json_dataset['data']:
                    if 'summary_entire' in data.keys():
                        for paragraph in data['summary_entire']:
                            # print(paragraph.keys())
                            paragraph = re.sub('\[([^]]+)\]','',paragraph['orginal_text']).strip()
                            sentences = [re.split(r'(?<!\d)\.(?!\d)', re.sub('\t','',sentence.strip('\n').strip())) for sentence in re.split(r'(?<!\d)\.(?!\d)', paragraph) if len(sentence) != 0]
                            sentences = flatten(sentences)
                            sentence_list += sentences

                    elif 'summary_selection' in data.keys():
                        for paragraph in data['summary_selection']:
                            paragraph = re.sub('\[([^]]+)\]', '', paragraph['orginal_text']).strip()
                            sentences = [re.split(r'(?<!\d)\.(?!\d)', re.sub('\t', '', sentence.strip('\n').strip())) for sentence in
                                         re.split(r'(?<!\d)\.(?!\d)', paragraph) if len(sentence) != 0]
                            sentences = flatten(sentences)
                            sentence_list += sentences


            elif isinstance(json_dataset, list):
                for list_dataset in json_dataset:
                    for dict_paragraph in list_dataset['data']:
                        for paragraph in dict_paragraph['summary_entire']:
                            sentence  = [re.sub('\t', '',sentence.strip('\n').strip()) for sentence in re.split(r'(?<!\d)\.(?!\d)', paragraph['orginal_text']) if len(sentence) != 0]
                            sentence = flatten(sentence)
                            sentence_list += sentence
                        for paragraph in dict_paragraph['summary_section']:
                            sentence_list = [re.sub('\t','',sentence.strip('\n').strip()) for sentence in re.split(r'(?<!\d)\.(?!\d)',paragraph['orginal_text']) if len(sentence) != 0]
                            sentence = flatten(sentence)
                            sentence_list += sentence
            else:
                print('sorry')

    with open('train_dataset.txt', 'a') as f:
        f.writelines(str(sentence) + '.\n' for sentence in sentence_list)

    print('parsing_test_3 is over')

def sentence_parsing_test_4():
    word_dataset = glob("../../../DATASET/basic_scientist_dataset/Training/labled/*.json")
    sentence_list = []
    for json_file in word_dataset:
        with open(json_file, 'r') as f:
            json_dataset = json.load(f)
        for paragraphs in json_dataset['paragraph']:
            for sentence in paragraphs['sentences']:
                sentence_list.append(sentence['src_sentence'])

    with open("train_dataset.txt", 'a') as f:
        f.writelines(str(sentence) + '.\n' for sentence in sentence_list)

    print('parsing_test_4 is over')

def sentence_parsing_test_5():
    word_dataset = glob("../../../DATASET/machine_translation_dataset_1/*.json")
    sentence_list = []
    for json_file in word_dataset:
        with open(json_file, 'r') as f:
            json_dataset = json.load(f)

        for paragraph in json_dataset['data']:
            sentence = paragraph['paragraphs'][0]['context']
            sequences = re.split(r'(?<!\d)\.(?!\d)', re.sub('\t','', sentence))
            sequences = [sequence.strip() for sequence in sequences if len(sentence) > 2]
            sentence_list += sequences

    with open("train_dataset.txt", 'a') as f:
        f.writelines(str(sentence) + '.\n' for sentence in sentence_list)

    print('parsing_test_5 is over')


# book_summarization
def sentence_parsing_test_6():
    word_dataset = glob("../../../DATASET/book_summarization/Training/*/*.json")
    sentence_list = []

    for json_file in word_dataset:
        with open(json_file, 'r') as f:
            json_dataset = json.load(f)
        # print(json_dataset['summary'])
        sentences = [sentence.strip() for sentence in re.split(r'(?<!\d)\.(?!\d)', re.sub('\t','', json_dataset['summary']))
                     if len(sentence) != 0]
        sentence_list += sentences

    with open("train_dataset.txt", 'a') as f:
        f.writelines(str(sentence) + '.\n' for sentence in sentence_list)

# document_summarization
def sentence_parsing_test_7():
    word_dataset = glob("../../../DATASET/document_summarization/Training/*/*.json")
    sentence_list = []

    for json_file in word_dataset:
        with open(json_file, 'r') as f:
            json_dataset = json.load(f)
        for documents in json_dataset['documents']:
            for text_set in documents['text']:
                for text_dict in text_set:
                    sentence = [re.sub('"', '', sentence.strip()) for sentence in re.split(r'(?<!\d)\.(?!\d)',text_dict['sentence']) if len(sentence) != 0]
                    sentence_list += sentence

    with open("train_dataset.txt", 'a') as f:
        f.writelines(str(sentence) + '.\n' for sentence in sentence_list)

# mapping qna
def sentence_parsing_test_8():
    import pandas as pd
    import numpy as np
    word_dataset = "../../../DATASET/paper-question-answer/train.csv"
    dataset = pd.read_csv(word_dataset)
    dataset.drop(columns=['id','category'], axis=1, inplace=True)

    dataset = dataset.values.flatten()

    with open('train_dataset.txt', 'w') as f:
        for sentence in dataset:
            f.write(f"{sentence}\n")

if __name__ == '__main__':
    # sentence_parsing_test_1()
    # sentence_parsing_test_2()
    # sentence_parsing_test_3()
    # sentence_parsing_test_4()
    # sentence_parsing_test_5()
    # sentence_parsing_test_6()
    # sentence_parsing_test_7()
    sentence_parsing_test_8()