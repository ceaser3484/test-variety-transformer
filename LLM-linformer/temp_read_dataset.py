

def check_dataset():
    with open('train_dataset.txt', 'r') as f:
        dataset = f.readlines()

    for sentence in dataset:
        print(sentence, end='')

if __name__ == '__main__':
    check_dataset()