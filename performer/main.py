import torch


def train_main():
    from random import choices
    from glob import glob
    import gzip
    import os
    import yaml

    with open("hyper-parameter.yaml") as f:
        hyper_parameter = yaml.full_load(f)

    # vocab = torch.load("../../pickles/vocab.pth")
    # print(vocab)
    # exit()

    # 데이터 로드 또는 생성
    if os.path.isfile("../../pickles/tokenized_data.gz.pkl"):
        # 이미 토큰화된 데이터 존재
        print("✅ 토큰화된 데이터 로딩 중...")
        import gzip
        import pickle
        with gzip.open("../../pickles/tokenized_data.gz.pkl", 'rb') as f:
            chunked_tokenized_data = pickle.load(f)

        # Vocab 로드
        vocab = torch.load("../../pickles/vocab.pth")
        print(f"✅ 로딩 완료 - Vocab: {len(vocab):,}, Chunks: {len(chunked_tokenized_data):,}")

    else:
        # 데이터 생성 필요
        if not os.path.exists("./DATA/"):
            print("❌ ERROR: DATA/ 디렉토리가 없습니다!")
            exit()

        import multi_processing as mt

        print("📂 원본 데이터 로딩 중...")
        pre_dataset = []
        txt_set = glob('./DATA/*.txt')
        for txt in txt_set:
            with open(txt, 'r', encoding='utf-8') as f:
                pre_dataset += [sentence.strip('\n') for sentence in f.readlines() if len(sentence) > 2]

        print(f"✅ 원본 데이터 로드 완료: {len(pre_dataset):,} 문장")
        # sort(pre_dataset, key=len, reverse=True)

        # Vocab 생성 또는 로드
        if not os.path.isfile("../../pickles/vocab.pth"):
            print("📝 Vocab 생성 중...")
            vocab = mt.multi_thread_create_vocab(pre_dataset, min_freq=hyper_parameter['min_freq'])
            torch.save(vocab, "../../pickles/vocab.pth")

            reverse_vocab = dict((value, key) for key, value in vocab.items())
            torch.save(reverse_vocab, "../../pickles/reversed_vocab.pth")
            print(f"✅ Vocab 생성 완료: {len(vocab):,} 토큰")
        else:
            vocab = torch.load("../../pickles/vocab.pth")
            print(f"✅ Vocab 로드 완료: {len(vocab):,} 토큰")

        # Chunking & Tokenization
        print("\n🔨 Chunking & Tokenization 시작...")
        chunked_tokenized_data = mt.multi_thread_chunk_sentence(
            pre_dataset, vocab, max_length=hyper_parameter['max_len']
        )
        print(f"✅ 완료 - 총 {len(chunked_tokenized_data):,}개 chunks 생성")
    
    for key in vocab.keys():
        print(key)

    
            
if __name__ == '__main__':
    train_main()
