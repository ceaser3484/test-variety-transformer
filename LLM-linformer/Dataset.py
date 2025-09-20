from torch.utils.data import Dataset
from mecab import MeCab
import openkorpos_dic
from tqdm import tqdm


class SentenceDataset(Dataset):

    def __init__(self, data, vocab, max_len, state):
        super(SentenceDataset, self).__init__()
        import glob
        import word_process as wp
        user_dict = glob.glob("../../mecab-dict/*.dic")
        # self.data = data
        self.vocab = vocab
        self.mecab = MeCab(dictionary_path=openkorpos_dic.DICDIR, user_dictionary_path=user_dict)

        self.sentence_list = []
        pbar = tqdm(data, ascii='.#')
        for sentence in pbar:
            pbar.set_description(f"{state} wrapping: ")
            token_list = []
            token_list.append(self.vocab['<sos>'])
            preprocessed_sentence = wp.replace_currency(sentence, "<current>")
            preprocessed_sentence = wp.replace_time(preprocessed_sentence, "<time>")
            preprocessed_sentence = wp.replace_date(preprocessed_sentence, '<date>')
            preprocessed_sentence = wp.replace_usual_num(preprocessed_sentence, '<NUM>')

            pos_n_tokens = self.mecab.pos(preprocessed_sentence)
            sentence_length = len(pos_n_tokens)
            for token, pos in pos_n_tokens:
                if token in ["<current>","<time>","<date>","<NUM>"]:
                    token_list.append(vocab[token])
                else:
                    if f"{token}/{pos}" in vocab:
                        token_list.append(vocab[f"{token}/{pos}"])
                    else:
                        token_list.append(vocab['<unk>'])
            token_list.append(self.vocab['<eos>'])

            if len(token_list) <= (max_len):
                self.sentence_list.append(token_list)
            elif len(token_list) > (max_len):
                residue = len(token_list) - (max_len)
                for idx in range(residue):
                    self.sentence_list.append(token_list[idx:idx + (max_len)])

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, item):
        return self.sentence_list[item],  self.sentence_list[item]

