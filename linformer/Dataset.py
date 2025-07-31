from torch.utils.data import Dataset
from mecab import MeCab
import openkorpos_dic


class SentenceDataset(Dataset):

    def __init__(self, data, vocab):
        super(SentenceDataset, self).__init__()
        import glob
        user_dict = glob.glob("../../mecab-dict/*.dic")
        self.data = data
        self.vocab = vocab
        self.mecab = MeCab(dictionary_path=openkorpos_dic.DICDIR, user_dictionary_path=user_dict)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        question, answer = self.data[item, :]
        question_morphs = self.mecab.morphs(question)
        answer_morphs = self.mecab.morphs(answer)

        question_tokens = [self.vocab[tokens] for tokens in question_morphs] + [self.vocab['<eos>']]
        answer_tokens = [self.vocab[token] for token in answer_morphs] + [self.vocab['<eos>']]

        return question_tokens, answer_tokens


