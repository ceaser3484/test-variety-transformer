from json import JSONDecodeError
from tqdm import tqdm
from mecab import MeCab
import re
import openkorpos_dic
from glob import glob
import json


# mapping qna
def sentence_parsing_test_1():
    import pandas as pd
    import numpy as np
    word_dataset = "../../../DATASET/paper-question-answer/train.csv"
    dataset = pd.read_csv(word_dataset)
    dataset.drop(columns=['id','category'], axis=1, inplace=True)

    dataset = dataset.values.flatten()

    word_dataset = "../../../DATASET/paper-question-answer/test.csv"
    test_dataset = pd.read_csv(word_dataset)
    test_dataset.drop('id',axis=1, inplace=True)
    test_dataset = test_dataset.values.flatten()
    dataset = np.concatenate([dataset, test_dataset], axis=0)

    with open('train_dataset.txt', 'w') as f:
        for sentence in dataset:
            f.write(f"{sentence}\n")

# korean_data_LLM
def sentence_parsing_test_2():
    # TODO 수정 필요
    from glob import glob
    import json
    import re

    dataset_path = "../../../DATASET/korean_data_LLM/*json"
    dataset_files = glob(dataset_path)
    sentence_list = []
    
    with open(dataset_files[0]) as f:
        data = json.load(f)

    for info in data['data_info']:
        question = info['question'].strip().replace('\n','')
        sentence_list.append(question)
        for i in range(1, 6):
            sentence_list.append(info[f'answer0{i}']['contents'].replace('\n','').strip())

    with open(dataset_files[1]) as f:
        data = json.load(f)

    for info in data['data_info']:
        sentence_list.append(info['question'].strip())
        sentence_list.append(info['answer']['contents'].replace('\n','').strip())

    with open("LLM_text.txt", 'w') as f:
        for sentence in sentence_list:
            f.write(f"{sentence}\n")

# article summerization 
def sentence_parsing_test_3():
    from glob import glob
    import json

    dataset_path = "../../../DATASET/article_summarization/*/*/*.json"
    json_files = glob(dataset_path)
    sentence_list = []

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
        for context in data['data']:
            paragraphs = context['paragraphs'][0]['context'].replace('\n','').strip()
            sentence_list.append(paragraphs)

            for qna in context['paragraphs'][0]['qas']:
                
                question = qna['question'].strip()
                answer = qna['answers']['text'].strip()
                sentence_list.append(question)
                sentence_list.append(answer)

    with open("article_summarization.txt", 'w') as f:
        for sentence in sentence_list:
            f.write(f"{sentence}\n")

# research_paper_summarization
def sentence_parsing_test_4():
    from glob import glob
    import json

    # dataset_path = "../../../DATASET/document_summarization/*"
    dataset_path = "../../../DATASET/research_paper_summarization/*/*/*.json"
    json_files = glob(dataset_path)
    
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
            
            for context in data['data']:
                pass
                


    

if __name__ == '__main__':

    sentence_parsing_test_1()
    sentence_parsing_test_2()
    sentence_parsing_test_3()
    # sentence_parsing_test_4()