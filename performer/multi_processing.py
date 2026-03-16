import multiprocessing
from collections import Counter
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re
import glob
import openkorpos_dic
from mecab import MeCab
import kss

global_mecab = None

def __init_mecab():
    global global_mecab
    user_dicts = glob.glob("../../mecab-dict/*.dic")
    global_mecab = MeCab(dictionary_path=openkorpos_dic.DICDIR, user_dictionary_path=user_dicts)
    global kss
    kss = kss

def __create_vocab(sentences):
    import re
    
    global global_mecab
    token_sentence_list = []
    for sentence in kss.split_sentences(sentences):
        sentence = re.sub(r'([^가-힣A-Za-z])', r' \1 ', sentence)

        for morph, pos in global_mecab.pos(sentence):
            morph = morph.strip()

            if morph.isdigit():
                token_sentence_list += [f"{digit}/{pos}" for digit in morph]

            elif re.fullmatch(r'[\u4E00-\u9FFF]+', morph):
                # 한자어가 여러 글자일 경우 → 문자 단위로 분리
                token_sentence_list += [f"{char}/{pos}" for char in morph]
            
            elif re.fullmatch(r'[^\w\s]+', morph):
                # 특수문자도 하나하나씩 분리
                token_sentence_list += [f"{char}/{pos}" for char in morph]

            else:
                if morph == "<unk>":
                    token_sentence_list.append("<unk>")
                else:
                    token_sentence_list.append(f"{morph}/{pos}")
    return token_sentence_list


def __process_batch_for_vocab(chunks):
    """
    배치 단위로 여러 문장을 처리하여 Counter를 반환
    오버헤드 감소를 위해 worker마다 여러 문장을 한번에 처리
    """
    __init_mecab()  # 각 프로세스마다 MeCab 초기화
    batch_counter = Counter()

    for sentences in chunks:
        tokens = __create_vocab(sentences)
        batch_counter.update(tokens)

    return batch_counter

def multi_thread_create_vocab(data, min_freq=20):
    import time
    print("multi thread create vocab start")
    vocab = {'<pad>':0,'<sos>':1,'<eos>':2,'<unk>': 3, '<mask>': 4, '<answer>': 5, '<cls>': 6, '<sep>': 7}
    global_counter = Counter()
    data.sort(key=len, reverse=True)
    print("🚀 multi-processing phase start")

    # 오버헤드 감소를 위한 큰 배치 크기
    batch_size = 1000000
    # worker당 처리할 문장 묶음 크기 (오버헤드 감소)
    chunk_size = 1000
    num_workers = cpu_count()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:

        for batch_idx, batch_start in enumerate(range(0, len(data), batch_size)):
            batch = data[batch_start:batch_start + batch_size]
            batch_end = min(batch_start + batch_size, len(data))

            print(f"\n📤 Batch {batch_idx + 1} 처리 중: {batch_start:,} ~ {batch_end:,} / {len(data):,}")

            # 문장들을 chunk_size 단위로 묶어서 작업 제출
            sentence_chunks = [
                batch[i:i + chunk_size]
                for i in range(0, len(batch), chunk_size)
            ]

            print(f"   └─ {len(sentence_chunks)}개의 작업 청크로 제출 (각 청크당 최대 {chunk_size}개 문장)")

            # 청크 단위로 작업 제출
            futures = [
                executor.submit(__process_batch_for_vocab, chunk)
                for chunk in sentence_chunks
            ]

            # 결과 수집
            batch_counter = Counter()
            with tqdm(total=len(futures),
                     desc=f"Batch {batch_idx + 1} processing",
                     dynamic_ncols=True,
                     unit="chunk") as pbar:

                for future in as_completed(futures):
                    chunk_counter = future.result()
                    batch_counter.update(chunk_counter)
                    pbar.update(1)
                    pbar.set_postfix({
                        "토큰 종류": f"{len(batch_counter):,}",
                        "총 토큰": f"{sum(batch_counter.values()):,}"
                    })

            global_counter.update(batch_counter)
            print(f"✅ Batch {batch_idx + 1} 완료 — 배치 토큰 수: {sum(batch_counter.values()):,}, 누적 토큰 종류: {len(global_counter):,}")

    print("\n📊 모든 작업이 완료되었습니다. Vocabulary 생성 중...\n")
    print(f"총 고유 토큰 수: {len(global_counter):,}")
    print(f"min_freq={min_freq} 필터링 적용 중...")

    for token, freq in global_counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)


    print(f"✅ 최종 vocab 크기: {len(vocab):,} (패딩 제외: {len(vocab) - 1:,})")
    return vocab


def temp_func(data):
    exit()




def __chunk_sentence(paragraph, vocab, max_length):
    """
    하나의 문장을 토큰화하고 max_length에 맞게 chunking

    Args:
        paragraph: 원본 문장 (string)
        vocab: vocabulary dict
        max_length: 최대 시퀀스 길이

    Returns:
        List[List[int]]: 토큰화되고 chunking된 문장들
    """
    global global_mecab
    global kss

    token_list = [vocab['<sos>'], vocab['<cls>']]  # 문장 시작 토큰
    sentences = kss.split_sentences(paragraph)
    for sentence in sentences:
        
        sentence = re.sub(r'([^가-힣])', r' \1 ', sentence)

        # MeCab으로 형태소 분석
        for morph, pos in global_mecab.pos(sentence):
            morph = morph.strip()
            # 숫자는 한 글자씩 분리
            if morph.isdigit():
                for digit in morph:
                    token = f"{digit}/{pos}"
                    token_list.append(vocab.get(token, vocab['<unk>']))

            # 한자는 한 글자씩 분리
            elif re.fullmatch(r'[\u4E00-\u9FFF]+', morph):
                for char in morph:
                    token = f"{char}/{pos}"
                    token_list.append(vocab[token])

            # 특수문자는 하나하나씩 떼어서
            elif re.fullmatch(r'[^\w\s]+', morph):
                for char in morph:
                    token = f"{char}/{pos}"
                    token_list.append(vocab[token])

            else:
                if morph == "<unk>":
                    token_list.append(vocab['<unk>'])
                else:
                    token = f"{morph}/{pos}"
                    token_list.append(vocab[token])
                
        token_list.append(vocab['<sep>'])  # 문장 종료 토큰
    token_list.append(vocab['<eos>'])  # 문단 종료 토큰

    # max_length로 chunking
    chunked = []
    if len(token_list) <= max_length:
        chunked.append(token_list)
    else:
        # sliding window로 자르기 (overlap 50%)
        stride = max_length // 2
        for i in range(0, len(token_list), stride):
            chunk = token_list[i:i + max_length]
            if len(chunk) >= max_length // 10:  # 너무 작은 chunk는 버림
                chunked.append(chunk)
            if i + max_length >= len(token_list):
                break

    return chunked


def __process_batch_for_chunking(sentences, vocab, max_length):
    """
    배치 단위로 여러 문장을 처리하여 chunked 데이터를 반환
    오버헤드 감소를 위해 worker마다 여러 문장을 한번에 처리

    Args:
        sentences: List[str] - 처리할 문장들
        vocab: vocabulary dict
        max_length: 최대 시퀀스 길이

    Returns:
        List[List[int]]: 모든 문장의 chunks
    """
    __init_mecab()  # 각 프로세스마다 MeCab 초기화
    all_chunks = []

    for sentence in sentences:
        chunks = __chunk_sentence(sentence, vocab, max_length)
        all_chunks.extend(chunks)

    return all_chunks


def multi_thread_chunk_sentence(data, vocab, max_length=1000, resume=True):
    """
    Multi-processing으로 데이터 전체를 토큰화하고 chunking (중간 저장 지원)

    Args:
        data: List[str] - 원본 문장들
        vocab: vocabulary dict
        max_length: 최대 시퀀스 길이
        resume: 중단된 작업 이어서 하기 (default: True)

    Returns:
        List[List[int]]: 토큰화되고 chunking된 모든 문장
    """
    import time
    import pickle
    import gzip
    import os

    print("🚀 Chunking & Tokenization 시작")
    print(f"총 문장 수: {len(data):,}")
    print(f"Max length: {max_length}")

    data.sort(key=len, reverse=True)  # 긴 문장부터 처리
    batch_size = 100000  # 메모리 효율을 위해 조정
    # all_chunked_data = []

    # 중간 저장 디렉토리 생성
    checkpoint_dir = "../../pickles/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 이미 처리된 배치 확인
    completed_batches = set()
    if resume:
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith("batch_") and filename.endswith(".pkl.gz"):
                batch_num = int(filename.split("_")[1].split(".")[0])
                completed_batches.add(batch_num)

        if completed_batches:
            print(f"✅ 이미 완료된 배치: {sorted(completed_batches)}")
            # 기존 배치 데이터 로드
            # for batch_num in sorted(completed_batches):
            #     batch_file = os.path.join(checkpoint_dir, f"batch_{batch_num}.pkl.gz")
            #     with gzip.open(batch_file, 'rb') as f:
            #         batch_data = pickle.load(f)
            #         # all_chunked_data.extend(batch_data)
            # print(f"📦 로드된 총 chunks: {len(all_chunked_data):,}")

    # ProcessPoolExecutor 사용 (CPU-bound 작업)
    num_workers = min(12, cpu_count())  
    # 오버헤드 감소를 위한 청크 크기
    chunk_size = 100
    print(f"🔧 Worker 수: {num_workers}, 청크 크기: {chunk_size}")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:

        for batch_idx, batch_start in enumerate(range(0, len(data), batch_size)):
            # 이미 처리된 배치는 스킵
            if batch_idx in completed_batches:
                print(f"⏭️  Batch {batch_idx} 스킵 (이미 완료)")
                continue

            batch = data[batch_start:batch_start + batch_size]
            batch_end = min(batch_start + batch_size, len(data))

            print(f"\n📤 Batch {batch_idx} 제출: {batch_start:,} ~ {batch_end:,} / {len(data):,}")

            # 문장들을 chunk_size 단위로 묶어서 작업 제출
            sentence_chunks = [
                batch[i:i + chunk_size]
                for i in range(0, len(batch), chunk_size)
            ]

            print(f"   └─ {len(sentence_chunks)}개의 작업 청크로 제출 (각 청크당 최대 {chunk_size}개 문장)")

            # 청크 단위로 작업 제출
            futures = [
                executor.submit(__process_batch_for_chunking, chunk, vocab, max_length)
                for chunk in sentence_chunks
            ]

            # 결과 수집
            batch_chunks = []
            with tqdm(total=len(futures),
                     desc=f"Batch {batch_idx} tokenizing",
                     dynamic_ncols=True,
                     unit="chunk") as pbar:

                for future in as_completed(futures):
                    chunks = future.result()
                    batch_chunks.extend(chunks)
                    pbar.update(1)
                    pbar.set_postfix({"배치 chunks": f"{len(batch_chunks):,}"})

            # 배치 중간 저장
            batch_file = os.path.join(checkpoint_dir, f"batch_{batch_idx}.pkl.gz")
            with gzip.open(batch_file, 'wb') as f:
                pickle.dump(batch_chunks, f)

            # all_chunked_data.extend(batch_chunks)
            print(f"✅ Batch {batch_idx} 완료 & 저장 - 배치 chunks: {len(batch_chunks):,}")
            del batch_chunks  # 메모리 절약


    # 최종 병합
    print("\n💾 최종 병합 시작...")
    all_chunks = []
    batch_files = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("batch_") and f.endswith(".pkl.gz")
    ])
    print(f"📦 병합할 배치 파일 수: {len(batch_files)}")

    for bf in batch_files:
        batch_path = os.path.join(checkpoint_dir, bf)
        print(f"🔗 병합 중: {batch_path}")
        with gzip.open(batch_path, 'rb') as f:
            batch_data = pickle.load(f)
            all_chunks.extend(batch_data)

    print(f"🎉 병합 완료! 총 {len(all_chunks):,} chunks")
    output_file = "../../pickles/tokenized_data.gz.pkl"

    with gzip.open(output_file, 'wb') as f:
        pickle.dump(all_chunks, f)

    print(f"✅ 최종 저장 완료: {output_file}")


    return all_chunks