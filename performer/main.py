import torch

def make_collate_fn(max_length, pad_token_id):
    def collate_fn(batch):
        train_input_list = []
        train_target_list = []
        for train_input, train_target in batch:
            train_input_list.append(torch.tensor(train_input, dtype=torch.long))
            train_target_list.append(torch.tensor(train_target, dtype=torch.long))
        
        # 패딩
        num_batches = len(train_input_list)
        temp = train_input_list + train_target_list
        
        padded = torch.nn.utils.rnn.pad_sequence(temp, batch_first=True, padding_value=pad_token_id)
        print("패딩 후 시퀀스 크기:", padded.size())
        if padded.size(1) < max_length:
            padded = torch.nn.functional.pad(padded, (0, max_length - padded.size(1)), value=pad_token_id)
        
        train_input_padded = padded[:num_batches]
        train_target_padded = padded[num_batches:]
        return train_input_padded, train_target_padded
    return collate_fn


def train_loop(model, dataloader, criterion, optimizer, device, num_epochs, fold_idx):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (train_input, train_target) in enumerate(dataloader):
            train_input, train_target = train_input.to(device), train_target.to(device)
            optimizer.zero_grad()
            ouput = model(train_input)
            print(output.size(), train_target.size())
            exit()
            


def train_main():
    from random import choices
    from glob import glob
    import gzip
    import os
    import yaml
    from Dataset import PreDataset
    from models import Performer
    
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

            if os.path.exists("../../models/performer.pt"):
                os.remove("../../models/performer.pt")
                print("✅ 기존 모델 파일 삭제 완료")
            else:
                print("✅ 기존 모델 파일 없음, 새로 생성 예정")

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
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    collate_fn = make_collate_fn(max_length=hyper_parameter['max_len'] -1, pad_token_id=vocab['<pad>'])
    model = Performer(hyper_parameter, vocab_size=len(vocab))
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=hyper_parameter['learning_rate'])
    # dataset = PreDataset(chunked_tokenized_data)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=hyper_parameter['batch_size'], shuffle=True, collate_fn=collate_fn)
    
    # k-fold 교차 검증을 위한 랜덤 샘플링
    random_choose = choices(range(len(chunked_tokenized_data)), k=100)
    train_data = []
    eval_data = []
    
    for k_idx in range(5):
        train_data = []
        eval_data = []
        for idx, sequence in enumerate(chunked_tokenized_data):
            if idx in random_choose:
                eval_data.append(sequence)
            else:
                train_data.append(sequence)

        train_dataset = PreDataset(train_data)
        eval_dataset = PreDataset(eval_data)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=hyper_parameter['batch_size'], shuffle=True, collate_fn=collate_fn)
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=hyper_parameter['batch_size'], shuffle=False, collate_fn=collate_fn)

        avg_train_loss = train_loop(model, train_dataloader, criterion, optimizer, device, hyper_parameter['num_epochs'], fold_idx=k_idx)
        print(f"Fold {k_idx} - Average Train Loss: {avg_train_loss:.4f}")
        avg_eval_loss = eval_loop(model, eval_dataloader, criterion, device)
        print(f"Fold {k_idx} - Average Eval Loss: {avg_eval_loss:.4f}")
        
if __name__ == '__main__':
    train_main()
