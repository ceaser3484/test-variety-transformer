import re

def replace_date(sentence, token):
    sentence = re.sub(r'\d{4}[년\-/]\s*\d{1,2}[월\-/]\s*\d{1,2}일?', token, sentence)
    sentence = text = re.sub(r'\d{4}년\s*\d{1,2}월', token, sentence)
    sentence = re.sub(r'\d{4}년', token, sentence)
    sentence = re.sub(r'\d{4}월', token, sentence)
    sentence = re.sub(r'\d{4}일', token, sentence)
    return sentence

def replace_currency(sentence, token):
    pass

def replace_time(sentence, token):
    pass

def replace_usual_num(sentence, token):
    sentence = re.sub(r'\d+', token, sentence)
    return sentence


if __name__ == '__main__':
    with open("train_dataset.txt") as f:
        sentences = [sentence.strip('\n') for sentence in f.readlines()]
    for sentence in sentences:
        adjust_sentence = replace_date(sentence, '<NUM>')
        adjust_sentence = replace_usual_num(adjust_sentence, '<NUM>')
        print(adjust_sentence)