import re

def replace_date(sentence, token):
    sentence = re.sub(r'\d{4}[년\-/]\s*\d{1,2}[월\-/]\s*\d{1,2}일?', token, sentence)
    sentence = text = re.sub(r'\d{4}년\s*\d{1,2}월', token, sentence)
    sentence = re.sub(r'\d{4}년', token, sentence)
    sentence = re.sub(r'\d{4}월', token, sentence)
    sentence = re.sub(r'\d{4}일', token, sentence)
    return sentence

def replace_currency(sentence, token):
    sentence = re.sub(r'\d{1,3}(?:,\d{3})*\s*(원|달러|엔|유로|파운드|만원|천원|억원)', token, sentence)
    return sentence

def replace_time(sentence, token):
    sentence = re.sub(r'(오전|오후)?\s*\d{1,2}시\s*\d{1,2}분?', token, sentence)
    sentence = re.sub(r'\d{1,2}:\d{1,2}(:\d{1,2})?', token, sentence)
    sentence = re.sub(r'\d{1,2}시\s*\d{1,2}분?', token, sentence)
    sentence = re.sub(r'\d{1,2}hour\s*\d{1,2}minute?', token, sentence)
    return sentence

def replace_usual_num(sentence, token):
    sentence = re.sub(r'\d+', token, sentence)
    return sentence


if __name__ == '__main__':
    # with open("train_dataset.txt") as f:
    #     sentences = [sentence.strip('\n') for sentence in f.readlines()]
    # for sentence in sentences:
    #     adjust_sentence = replace_date(sentence, '<NUM>')
    #     adjust_sentence = replace_usual_num(adjust_sentence, '<NUM>')
    #     print(adjust_sentence)
    from mecab import MeCab
    from glob import glob
    mecab = MeCab(user_dictionary_path=glob("../../mecab-dict/*dic"))
    sentence = "와 미쳤어. 오늘 40도 되는거 아니야? 휴대폰에는 35도라는데 번화가에 있으니 더 덥게 느껴진다"
    adjust_sentence = replace_usual_num(sentence, '<NUM>')
    print(mecab.morphs(adjust_sentence))