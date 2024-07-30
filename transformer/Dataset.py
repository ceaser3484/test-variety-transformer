import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, dataset, eng_vocab, ger_vocab, eng_spacy, ger_spacy):
        super(TranslationDataset, self).__init__()
        self.dataset = dataset
        self.eng_vocab = eng_vocab
        self.ger_vocab = ger_vocab
        self.eng_spacy = eng_spacy
        self.ger_spacy = ger_spacy

        self.eng_max_len = 0
        self.ger_max_len = 0

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item):
        eng, ger = self.dataset[item, :]
        return self.eng_to_tokens(eng), self.ger_to_tokens(ger)

    def eng_to_tokens(self, sentence):
        sentence = sentence.lower()
        eng_tokens = [self.eng_vocab[word.text] for word in self.eng_spacy(sentence)] + [self.eng_vocab['<eos>']]

        return eng_tokens

    def ger_to_tokens(self, sentence):
        sentence = sentence.lower()
        ger_tokens_input = [self.ger_vocab['<sos>']] + [self.ger_vocab[word.text] for word in self.ger_spacy(sentence)]
        ger_token_expect = [self.ger_vocab[word.text] for word in self.ger_spacy(sentence)] + [self.ger_vocab['<eos>']]
        return ger_tokens_input, ger_token_expect