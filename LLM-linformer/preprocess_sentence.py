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
# summerization_n_generate_report, , korean_corpus_from_books

# 해결이 필요한 것들
# , 

# 진행 중 : research_paper_summarization

# mapping_qna
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

    with open('./DATA/mapping_qna.txt', 'w') as f:
        for sentence in dataset:
            f.write(f"{sentence}\n")

# korean_data_LLM
def sentence_parsing_test_2():
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

    with open("./DATA/korean_data_LLM.txt", 'w') as f:
        for sentence in sentence_list:
            f.write(f"{sentence}\n")

# article summerization 
def sentence_parsing_test_3():
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

    with open("./DATA/article_summarization.txt", 'w') as f:
        for sentence in sentence_list:
            f.write(f"{sentence}\n")

# healthcare_QNA
def sentence_parsing_test_4():
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

    with open("./DATA/healthcare_QNA.txt", 'w') as f:
        for sentence in sentence_list:
            f.write(f"{sentence}\n")

# machine_reading
def sentence_parsing_test_5():
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

    with open("./DATA/machine_reading.txt", 'w') as f:
        for sentence in sentence_list:
            f.write(f"{sentence}\n")
                        
# summerization_n_generate_report
def sentence_parsing_test_6():
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
    
    with open("./DATA/summerization_n_generate_report.txt", 'w') as f:
        for sentence in sentence_list:
            f.write(f"{sentence}\n")    

# document_reading
def sentence_parsing_test_7():
    from glob import glob
    import json

    dataset_path = "../../../DATASET/document_reading/*/*/*/*"
    json_files = glob(dataset_path)
    sentence_list = []
    pbar = tqdm(json_files)
    for json_file in pbar:
        pbar.set_description("document_reading")
        with open(json_file) as f:
            data = json.load(f)
        # print(data.keys())  # 'Dataset', 'data'
        for data in data['data']:
            # print(data.keys())  # 'doc_id', 'doc_title', 'doc_source', 'doc_published', 'doc_class', 'created', 'paragraphs'
            title = data['doc_title']
            sentence_list.append(title)
            paragraph = data['paragraphs'][0]
            context = paragraph['context']
            sentence_list.append(context)
            for qas in paragraph['qas']:
                question = qas['question']
                answer = qas['answers']['text']
                sentence_list.append(question)
                sentence_list.append(answer)
            
    with open("./DATA/document_reading.txt", 'w') as f:
        for sentence in sentence_list:
            f.write(f"{sentence}\n")  

# official_service_data
def sentence_parsing_test_8():
    from glob import glob
    import json

    def clean_text(text_list):

        pattern_1 = re.compile(r"▲{1,}")
        pattern_2 = re.compile(r"네, {1,}")
        pattern_3 = re.compile(r"\n\t")
        result = []
        for text in text_list:
            text = text.strip()
            # text = text.split(":")[-1]
            text = pattern_1.sub("<unk>", text)
            text = pattern_2.sub("네, ", text)
            text = pattern_3.sub("", text)
            result.append(text.replace('\t',''))
        return result
            
    def reorganize_dialogue(contents, speaker1='손님', speaker2='상담사'):
        contents = contents.split('\n')
        last_speaker = None
        buffer = ""
        merged = []

        for text in contents:
            text = text.strip()
            if not text:
                continue
            
            if text.startswith(speaker1):
                speaker = speaker1
                content = text[len(speaker) + 1:].strip()
                
            elif text.startswith(speaker2):
                speaker = speaker2
                content = text[len(speaker) + 1:].strip()

            else:
                continue

            if last_speaker == speaker:
                buffer += ". " + content
            else:
                if buffer:
                    merged.append((last_speaker, buffer))
                buffer = content
                last_speaker = speaker
        
        if buffer:
            merged.append((last_speaker, buffer))
        
        
        return merged
        
    dataset_path = "../../../DATASET/official_service_data/*/*/*"
    json_files = glob(dataset_path)
    sentence_list = []
    pbar = tqdm(json_files)
    

    for json_file in pbar:
    
        pbar.set_description("official_service_data")
        with open(json_file) as f:
            data = json.load(f)
        len_data_keys = len(data[0].keys())

        if len_data_keys == 11:

            paragraph = data[0]['instructions'][0]['data'][0]
            instruction = paragraph['instruction']
            dialogue = paragraph['input'].strip()
            dialogue = reorganize_dialogue(dialogue, speaker1='상담원', speaker2='고객')
            reorganized_dialogue = []
            for role, talk in dialogue:
                reorganized_dialogue.append(f"{role}: {talk}")
            answer = paragraph['output'].strip()
            cleaned_text = clean_text([instruction] + reorganized_dialogue + [answer])
            sentence_list.extend(cleaned_text)
            
        elif len_data_keys == 9:
            qna = data[0]['consulting_content'].split("\n\n")
            
            cleaned_text = clean_text([sentence.replace('\n','') for sentence in qna])
            sentence_list.extend(cleaned_text)

        elif len_data_keys == 8:
            
            # print(data[0].keys()) # 'consulting_content', 'instructions'
            qna = [sentence.replace('\n','') for sentence in data[0]['consulting_content'].split("\n\n")]
            idx_list = []
            for idx, sentence in enumerate(qna):
                if sentence.startswith("Q") or sentence.startswith("A"):
                    idx_list.append(idx)

            head = ''.join(qna[:idx_list[0]])
            question = ''.join(qna[idx_list[0]:idx_list[1]+1])
            answer = ''.join(qna[idx_list[1]:])
            cleaned_text = clean_text([head, question, answer])
            sentence_list.extend(cleaned_text)


            instruction = data[0]['instructions'][0]['data'][0]['instruction']
            output = data[0]['instructions'][0]['data'][0]['output']
            cleaned_text = clean_text([instruction, output])
            sentence_list.extend(cleaned_text)

    
    with open("./DATA/official_service_data.txt", 'w') as f:
        for sentence in sentence_list:
            f.write(f"{sentence}\n")

# unofficial_service_data
def sentence_parsing_test_9():
    from glob import glob
    import json
    import re

    def clean_text(text_list):

        for text in text_list:
            text = re.sub(r'<(NAME|CHARGE|DATE)>', '<unk>', text)
            text = re.sub(r"▲{1,}", "<unk>", text)

        return text

    def reorganize_dialogue(contents, speaker1='손님', speaker2='상담사'):
        contents = contents.split('\n')
        last_speaker = None
        buffer = ""
        merged = []

        for text in contents:
            text = text.strip()
            if not text:
                continue
            
            if text.startswith(speaker1):
                speaker = speaker1
                content = text[len(speaker) + 1:].strip()
                
            elif text.startswith(speaker2):
                speaker = speaker2
                content = text[len(speaker) + 1:].strip()

            else:
                continue

            if last_speaker == speaker:
                buffer += ". " + content
            else:
                if buffer:
                    merged.append((last_speaker, buffer))
                buffer = content
                last_speaker = speaker
        
        if buffer:
            merged.append((last_speaker, buffer))
        
        return merged

    dataset_path = "../../../DATASET/unofficial_service_data/*/labeled/*.json"
    json_files = glob(dataset_path)
    pbar = tqdm(json_files)
    sentence_list = []

    for json_file in pbar:
        pbar.set_description("unofficial_service_data")
        with open(json_file) as f:
            data = json.load(f)
        num_key = len(data[0].keys())
        if num_key == 8:

            contents = data[0]['consulting_content'].replace('\n','').strip()
            cleaned_text = clean_text([contents])
            parts = re.split(r'(고객:|상담사:)', cleaned_text)
            rebuilt_dialogue = []
            for i in range(1, len(parts), 2):
                rebuilt_dialogue.append(parts[i] + parts[i+1].strip())

            merged_dialogue = reorganize_dialogue('\n'.join(rebuilt_dialogue), speaker1='고객', speaker2='상담사')
            for speaker, content in merged_dialogue:
                # print(f"{speaker}: {content}")
                sentence_list.append(f"{speaker}: {content}")

            sentence_list.append(data[0]['instructions'][0]['data'][0]['instruction'])
            sentence_list.append(data[0]['instructions'][0]['data'][0]['output'])
                   
        elif num_key == 10:
            # print(data[0].keys()) # 'source', 'source_id', 'consulting_category', 'client_gender', 'client_age', 'consulting_time', 'consulting_turns', 'consulting_length', 'consulting_content', 'instructions'
            contents = data[0]['consulting_content']
            cleaned_text = clean_text([contents])
            merged_dialogue = reorganize_dialogue(cleaned_text[0])
            for speaker, content in merged_dialogue:
                sentence_list.append(f"{speaker}: {content}")

            sentence_list.append(cleaned_text)
            sentence_list.append(data[0]['instructions'][0]['data'][0]['instruction'])
            sentence_list.append(data[0]['instructions'][0]['data'][0]['output'])
            

        elif num_key == 11:
            
            # print(data[0].keys())  # 'source', 'source_id', 'consulting_date', 'consulting_category', 'client_gender', 'client_age', 'consulting_time', 'consulting_turns', 'consulting_length', 'consulting_content', 'instructions'
            contents = data[0]['consulting_content'].split('\n')
            last_speaker = None
            buffer = ""
            merged = []

            for text in contents:
                text = text.strip()
                if not text:
                    continue
                
                cleaned_text = clean_text([text])
                customer = '손님'
                agent = '상담사'
                if cleaned_text.startswith('손님'):
                    speaker = customer
                    content = cleaned_text[len(speaker) + 1:].strip()
                    
                elif cleaned_text.startswith(agent):
                    speaker = agent
                    content = cleaned_text[len(agent) + 1:].strip()

                else:
                    continue

                if last_speaker == speaker:
                    buffer += ". " + content
                else:
                    if buffer:
                        merged.append((last_speaker, buffer))
                    buffer = content
                    last_speaker = speaker
            
            if buffer:
                merged.append((last_speaker, buffer))
            
            for speaker, content in merged:
                # print(f"{speaker}: {content}")
                sentence_list.append(f"{speaker}: {content}")
    
        with open("./DATA/unofficial_service_data.txt", 'w') as f:
            for sentence in sentence_list:
                f.write(f"{sentence}\n")
# korean_corpus_from_books
def sentence_parsing_test_miss_1():
    from glob import glob
    import json

    dataset_path = "../../../DATASET/korean_corpus_from_books/*/*/*.json"
    json_files = glob(dataset_path)
    sentence_list = []
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
        for paragraph in data['paragraphs']:
            for sentence in paragraph['sentences']:
                text = sentence['text']
                origin = sentence['original_text']
                sentence_list.append(text)
                sentence_list.append(origin)

    with open("./DATA/korean_corpus_from_books.txt", 'w') as f:
        for sentence in sentence_list:
            f.write(f"{sentence}\n")   
            
        


# research_paper_summarization
def sentence_parsing_test_miss_2():
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
            
            original_text = research_papers['data'][0]['summary_entire'][0]['orginal_text']
            print(research_papers['data'])

            exit()
                


    

if __name__ == '__main__':

    import os
    os.makedirs('./DATA', exist_ok=True)
    sentence_parsing_test_1()
    sentence_parsing_test_2()
    sentence_parsing_test_3()
    sentence_parsing_test_4()
    sentence_parsing_test_5()
    sentence_parsing_test_6()
    sentence_parsing_test_7()
    sentence_parsing_test_8()
    sentence_parsing_test_9()
    
