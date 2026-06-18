import gzip
import pickle
import random
import torch


def generate():
    with gzip.open("../../pickles/tokenized_data.gz.pkl", 'rb') as f:
        chunked_tokenized_data = pickle.load(f)
    
    print(chunked_tokenized_data[1])
    # random.shuffle(chunked_tokenized_data)
    # choosed = random.choice(chunked_tokenized_data)
    # choosed_1 = choosed[:5]
    reverse_vocab = torch.load("../../pickles/reversed_vocab.pth")
    for token in chunked_tokenized_data[1]:
        paticle = reverse_vocab[token]
        word = paticle.split("<@>")[0]
        print(word, end=" ")
    print()

if __name__ == '__main__':
    generate()
