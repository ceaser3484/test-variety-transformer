import gzip
import pickle
import random
import torch


def generate():
    with gzip.open("../../pickles/tokenized_data.gz.pkl", 'rb') as f:
        chunked_tokenized_data = pickle.load(f)
    
    
    random.shuffle(chunked_tokenized_data)
    choosed = random.choice(chunked_tokenized_data)
    choosed_1 = choosed[:100]
    print(choosed)
    reverse_vocab = torch.load("../../pickles/reversed_vocab.pth")
    for token in choosed_1:
        print(reverse_vocab[token])


if __name__ == '__main__':
    generate()