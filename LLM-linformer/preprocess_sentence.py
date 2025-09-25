from json import JSONDecodeError
from tqdm import tqdm
from mecab import MeCab
import re
import openkorpos_dic
from glob import glob
import json
from tqdm import tqdm

# 지금까지 preprossessed 된 것들. 
# mapping qna, korean_data_LLM, article summerization, healthcare_QNA, machine_reading
# summerization_n_generate_report

# 해결이 필요한 것들
# korean_corpus_from_books, document_reading

# 진행 중 : research_paper_summarization, document_reading

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
    from glob import glob
    import json

    dataset_path = "../../../DATASET/korean_data_LLM/*json"
    dataset_files = glob(dataset_path)
    sentence_list = []
    
    with open(dataset_files[0]) as f:
        data = json.load(f)
    pbar = tqdm(data['data_info'])
    for info in pbar:
        pbar.set_description("korean_data_LLM")
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
    pbar = tqdm(json_files)
    for json_file in pbar:
        pbar.set_description("article_summmarization")
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

# healthcare_QNA
def sentence_parsing_test_4():
    from glob import glob
    import json

    dataset_path = "../../../DATASET/healthcare_QNA/*/*/labeled/*/*/*/*/*"
    json_files = glob(dataset_path)
    sentence_list = []
    pbar = tqdm(json_files)
    
    for json_file in pbar:
        pbar.set_description("healthcare_QNA")
        with open(json_file) as f:
            data = json.load(f)
            if 'answer' in data.keys():
                answer_sentence = ''
                for sentence in data['answer'].values():
                    answer_sentence += sentence
                sentence_list.append(answer_sentence)
            elif 'question' in data.keys():
                sentence_list.append(data['question'])

    with open("healthcare_QNA.txt", 'w') as f:
        for sentence in sentence_list:
            f.write(f"{sentence}\n")

# machine_reading
def sentence_parsing_test_5():
    from glob import glob
    import json

    dataset_path = "../../../DATASET/machine_reading/*.json"
    json_files = glob(dataset_path)
    sentence_list = []
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
            # print(data.keys()) # 'creator', 'version', 'data'
            for datum in data['data']:
                paragraphs = datum['paragraphs'][0]
                context = paragraphs['context']
                sentence_list.append(context)
                # print(len(paragraphs['qas']))
                for qas in paragraphs['qas']:
                    if len(qas) == 3:
                        # print(qas.keys())  # 'classtype', 'id', 'question'
                        # 이 데이터셋은 answer가 없음. 
                        question = qas['question']
                        sentence_list.append(question)
                    elif len(qas) == 4:
                        # print(qas.keys()) # 'question', 'answers', 'id', 'classtype'
                        question = qas['question']
                        for answer in qas['answers']:
                            sentence_list.append(answer['text'])
                    else:
                        # print(qas.keys())  # 'classtype', 'id', 'answers', 'question', 'clue'
                        question = qas['question']
                        answer = qas['answers'][0]['text']
                        sentence_list.append(question)
                        sentence_list.append(answer)

    with open("machine_reading.txt", 'w') as f:
        for sentence in sentence_list:
            f.write(f"{sentence}\n")
                        
# summerization_n_generate_report
def sentence_parsing_test_6():
    from glob import glob
    import json

    dataset_path = "../../../DATASET/summerization_n_generate_report/*/*/*/*/*/*.json"
    json_files = glob(dataset_path)
    sentence_list = []
    pbar = tqdm(json_files)
    for json_file in pbar:
        pbar.set_description("summerization_n_generate_report")
        with open(json_file) as f:
            data = json.load(f)
        # print(data.keys()) # 'Meta(Acqusition)', 'Meta(Refine)', 'Annotation'
        context = data['Meta(Refine)']
        sentence_list.append(context)
        for annotation in data['Annotation'].values():
            if annotation is None:
                continue
            sentence_list.append(annotation)
    
    with open("machine_reading.txt", 'w') as f:
        for sentence in sentence_list:
            f.write(f"{sentence}\n")    

# document_reading
def sentence_parsing_test_():
    pass


# research_paper_summarization
def sentence_parsing_test_processing():
    from glob import glob
    import json

    # dataset_path = "../../../DATASET/document_summarization/*"
    dataset_path = "../../../DATASET/research_paper_summarization/*/*/*.json"
    json_files = glob(dataset_path)
    
    for json_file in json_files:
        with open(json_file) as f:
            research_papers = json.load(f)
            # print(research_papers['data'][0])
            # print(research_papers['data'][0]['summary_entire'][0].keys()) # orginal_text', 'summary_text'
            
            print(research_papers['data'][0]['summary_entire'][0]['orginal_text'])

            exit()
                


    

if __name__ == '__main__':

    # sentence_parsing_test_1()
    # sentence_parsing_test_2()
    # sentence_parsing_test_3()
    # sentence_parsing_test_4()
    # sentence_parsing_test_5()
    sentence_parsing_test_6()
