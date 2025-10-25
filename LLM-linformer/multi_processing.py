import multiprocessing
from collections import Counter
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import re
import glob
import openkorpos_dic
from mecab import MeCab

global_mecab = None

def __init_mecab():
    global global_mecab
    user_dicts = glob.glob("../../mecab-dict/*.dic")
    global_mecab = MeCab(dictionary_path=openkorpos_dic.DICDIR, user_dictionary_path=user_dicts)


def __create_vocab(sentence):
    global global_mecab
    token_sentence_list = []

    for morph, pos in global_mecab.pos(sentence):
        if morph.isdigit():
            token_sentence_list += [f"{digit}/{pos}" for digit in morph]

        elif re.fullmatch(r'[\u4E00-\u9FFF]+', morph):
            # 한자어가 여러 글자일 경우 → 문자 단위로 분리
            token_sentence_list += [f"{char}/{pos}" for char in morph]

        else:
            if morph.equals("<unk>"):
                token_sentence_list.append("<unk>")
            else:
                token_sentence_list.append(f"{morph}/{pos}")
    return token_sentence_list

def multi_thread_create_vocab(data, min_freq=20):
    import time

    vocab = {'<unk>': 0, '<mask>': 1, '<answer>': 2, '<cls>': 3, '<sep>': 4}
    futures = []
    global_counter = Counter()
    data.sort(key=len, reverse=True)
    print(data[0])
    exit()
    print("🚀 multi-processing phase start")
    batch_size = 5000


    with ProcessPoolExecutor(initializer=__init_mecab, max_workers=cpu_count()) as executor:
        
        for batch_start in range(0, len(data), batch_size):
            batch = data[batch_start:batch_start + batch_size]
           
            print(f"\n📤 작업 제출 중... {batch_start+1}에서 {batch_start + batch_size}까지의 문장 제출 중...\
                  /{len(batch)} 문장 제출 완료")

            futures = []
            start_times = {}
            sentence_lengths = {}

            for sentence in batch:
                future = executor.submit(__create_vocab, sentence)
                futures.append(future)
                start_times[future] = time.time()
                sentence_lengths[future] = len(sentence)

            print("\n📡 실시간 작업 모니터링 시작\n")
            batch_counter = Counter()

            with tqdm(total=len(futures), desc="create_vocab", dynamic_ncols=True) as pbar:
                for idx, future in enumerate(as_completed(futures)):
                    tokens = future.result()
                    duration = time.time() - start_times[future]
                    length = sentence_lengths[future]
                    
                    pbar.set_postfix({
                        "문장 길이": f"{length:4d}",
                        "처리 시간": f"{duration:.2f}초",
                        "토큰 수": f"{len(tokens)}"
                    })
                    batch_counter.update(tokens)
                    pbar.update(1)

            
            global_counter.update(batch_counter)
            print(f"📦 Batch {batch_start//batch_size + 1} 병합 완료 — 총 토큰 수: {sum(batch_counter.values())}")


    print("\n📊 모든 작업이 완료되었습니다. 결과를 집계합니다...\n")
    print(f"총 제출된 작업 수: {len(futures)}")


    for token, freq in global_counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    vocab['<pad>'] = -1000  # CrossEntropyLoss의 ignore_index로 사용
    return vocab


def temp_func(data):
    data.sort(key=len, reverse=True)
    user_dicts = glob.glob("../../mecab-dict/*.dic")
    local_mecab = MeCab(dictionary_path=openkorpos_dic.DICDIR, user_dictionary_path=user_dicts)

    # global global_mecab
    print(local_mecab.pos(data[1]))
    exit()




def __chunk_sentence(paragraph):
    global global_mecab
    



def multi_thread_chunk_sentence(data, vocab, max_length=512):
    from kss import split_sentences
    
    data.sort(key=len, reverse=True)
    chunked_sentences = []
    for paragraph in tqdm(data, desc="chunk_sentence", dynamic_ncols=True):
        sentences = split_sentences(paragraph)
        for sentence in sentences:
            print(sentence, end="\n\n")
        exit()
