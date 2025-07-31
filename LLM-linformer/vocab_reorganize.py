import csv
import glob
from os import system

def check_end_word(word):
    """종성 확인 함수"""
    if (ord(word[-1]) - 44032) % 28 == 0:
        return 'T'
    else:
        return 'F'


def make_word_to_user_dict():

    base_path = "../../mecab-dict/"
    user_word_txt_list = glob.glob(base_path + '*.txt')

    for user_word_txt in user_word_txt_list:
        word_txt = user_word_txt.split('/')[-1].split('.')[0]
        word_data = []
        with open(user_word_txt, 'r', encoding='utf-8') as f:
            words = f.readlines()
            for word in words:
                word = word.strip('\n')

                # 종성 체크 처리 및 분류
                word_processed = [word,'' ,'' ,'' ,'NNP','*', check_end_word(word), word, '*','*','*','*']
                word_data.append(word_processed)
                #######

        with open(base_path + word_txt + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(word_data)


def csv_to_mecab_user_dic():
    base_path = "../../mecab-dict/"
    csv_file_path = glob.glob(base_path + '*.csv')

    for csv_file in csv_file_path:
        word_csv = csv_file.split('/')[-1].split('.')[0]
        user_dict_path = base_path + word_csv + '.dic'
        command_txt = f'python3 -m mecab dict-index --userdic {user_dict_path} {base_path + word_csv}.csv'
        system(command_txt)


if __name__ == '__main__':
    make_word_to_user_dict()
    csv_to_mecab_user_dic()
