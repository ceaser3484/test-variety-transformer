import torch
from tqdm import tqdm

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
        # print("패딩 후 시퀀스 크기:", padded.size())
        if padded.size(1) < max_length:
            padded = torch.nn.functional.pad(padded, (0, max_length - padded.size(1)), value=pad_token_id)
        
        train_input_padded = padded[:num_batches]
        train_target_padded = padded[num_batches:]
        return train_input_padded, train_target_padded
    return collate_fn


def train_loop(model, dataloader, criterion, optimizer, device, num_epochs, fold_idx, epoch, accumulate_steps=10):
    from random import choice
    scaler = torch.amp.GradScaler('cuda')
    model.train()
    total_loss = []
    
    colour = '#' + ''.join([choice('0123456789ABCDEF') for _ in range(6)])
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), colour=colour, dynamic_ncols=True)
    sum_loss = 0.0
    optimizer.zero_grad()
    
    for batch_idx, (train, target) in pbar:
        pbar.set_description(f"Fold {fold_idx} - Epoch {epoch+1}/{num_epochs}")
        train, target = train.to(device), target.to(device)

        with torch.amp.autocast('cuda'):

            output = model(train)
            dim = output.size(-1)
            loss = criterion(output.view(-1, dim), target.view(-1))

        # gradient accumulation
        scaler.scale(loss).backward()
        sum_loss += loss.item()

        
        # optimizer.step()
        # print(loss.item())
        total_loss.append(loss.item())
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        if (batch_idx + 1) % accumulate_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if batch_idx % 100000 == 0 and batch_idx > 0:
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, 
            f"../../models/performer_trainning_progress.pt")

        avg_loss = sum(total_loss) / len(total_loss)
    print(f"Fold {fold_idx} - Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
    return avg_loss

def eval_loop(model, dataloader, criterion, device):
    model.eval()
    eval_losses = []
    colour = '#' + ''.join([choice('0123456789ABCDEF') for _ in range(6)])
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), colour=colour, dynamic_ncols=True)
    with torch.no_grad():
        for batch_idx, (train, target) in pbar:
            train, target = train.to(device), target.to(device)
            pbar.set_description("Evaluation in progress: ")
            output = model(train)
            dim = output.size(-1)
            loss = criterion(output.view(-1, dim), target.view(-1))
            eval_losses.append(loss.item())
    avg_eval_loss = sum(eval_losses) / len(eval_losses)
    return avg_eval_loss
            
def test_inference(model, test_data, device, max_len, reverse_vocab):
    """model, test_data, device, hyper_parameter['max_len'], reverse_vocab"""
    model.eval()
    start_token = test_data[0][0]
    generated = torch.tensor([start_token], dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        print("🔍 테스트 인퍼런스 시작...")
        for _ in range(max_len):
            predicted_tokens = model(generated)
            next_token_logit = predicted_tokens[0, -1, :]
            next_token_id = torch.argmax(next_token_logit).unsqueeze(0).unsqueeze(0)
            generated = torch.cat([generated, next_token_id], dim=1)

    generated_tokens = generated.squeeze().tolist()
    generated_text = ' '.join([reverse_vocab.get(token_id, '<unk>') for token_id in generated_tokens])
    print(f"Generated Text: {generated_text}")

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
        reverse_vocab = torch.load("../../pickles/reversed_vocab.pth")
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
                os.remove("../../models/performer_trainning_progress.pt")
                print("✅ 기존 모델 파일 삭제 완료")
            else:
                print("✅ 기존 모델 파일 없음, 새로 생성 예정")

            vocab = mt.multi_thread_create_vocab(pre_dataset, min_freq=hyper_parameter['min_freq'] + 1)
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
    
    torch.set_float32_matmul_precision('high')
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    collate_fn = make_collate_fn(max_length=hyper_parameter['max_len'], pad_token_id=vocab['<pad>'])
    model = Performer(hyper_parameter, vocab_size=len(vocab)).to(device)
    model = torch.compile(model, backend='eager')
    optimizer = torch.optim.Adafactor(model.parameters(), lr=hyper_parameter['learning_rate'], weight_decay=1e-6)

    if os.path.isfile("../../models/performer.pt"):
        model.load_state_dict(torch.load("../../models/performer.pt"))
        print("✅ 기존 모델 로드 완료")

    elif os.path.isfile(f"../../models/performer_trainning_progress.pt"):
        checkpoint = torch.load(f"../../models/performer_trainning_progress.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ 학습 진행 중인 모델 로드 완료")

    else:
        print("🚀 새로운 모델로 학습 시작!")

    # k-fold 교차 검증을 위한 랜덤 샘플링
    random_choose = choices(range(len(chunked_tokenized_data)), k=100)
    train_data = []
    eval_data = []
    
    for k_idx in range(hyper_parameter['num_fold']):
        train_data = []
        eval_data = []
        for idx, sequence in enumerate(chunked_tokenized_data):
            if idx in random_choose:
                eval_data.append(sequence)
            else:
                train_data.append(sequence)

        test_data = choices(eval_data, k=1)
        train_dataset = PreDataset(train_data)
        eval_dataset = PreDataset(eval_data)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=hyper_parameter['batch_size'], shuffle=True, num_workers=4, collate_fn=collate_fn)
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=hyper_parameter['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn)

        train_losses = []
        eval_losses = []
        for epoch in range(hyper_parameter['num_epochs']):

            test_inference(model, test_data, device, hyper_parameter['max_len'], reverse_vocab)
            print('\n'* 2)
            avg_train_loss = train_loop(model, train_dataloader, criterion, optimizer, device, hyper_parameter['num_epochs'], k_idx, epoch)
            train_losses.append(avg_train_loss)
            print(f"Fold {k_idx} - Average Train Loss: {avg_train_loss:.4f}")
            avg_eval_loss = eval_loop(model, eval_dataloader, criterion, device)
            print(f"Fold {k_idx} - Average Eval Loss: {avg_eval_loss:.4f}")
            eval_losses.append(avg_eval_loss)
    
    
    torch.save(model.state_dict(), "../../models/performer.pt")

        
if __name__ == '__main__':
    train_main()
