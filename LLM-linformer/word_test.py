from mecab import MeCab
import re
import openkorpos_dic
from glob import glob
import json


def sentence_parsing_test_1():
    word_dataset = glob("../../../DATASET/news_article_machine_translation_dataset/Training/*/*.json")
    sentence_list = []
    for json_file in word_dataset:
        with open(json_file, 'r') as f:
            json_dataset = json.load(f)

        for paragraphs in json_dataset['data']:
            for sentence in paragraphs['paragraphs']:
                # sentence_splited_by_eol = sentence['context'].split('\n\n')
                sentence_splited_by_eol = [re.sub('\n', '',sentence.strip('\n').strip()) for sentence in sentence['context'].split('\n\n') if len(sentence) != 0]
            sentence_list += sentence_splited_by_eol


def sentence_parsing_test_2():
    word_dataset = glob("../../../DATASET/korean_book_dataset/Training/labled/*/*.json")
    for json_file in word_dataset:
        if json_file.count('INFO') > 0:
            continue

        with open(json_file, 'r') as f:
            json_dataset = json.load(f)
        print(type(json_dataset['paragraphs']))
        exit()


if __name__ == '__main__':
    sentence_parsing_test_2()