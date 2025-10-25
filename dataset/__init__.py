import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split

pad, sos, eos, seq_len = 0, 1, 2, 10


class WordStat:
    def __init__(self, path):
        self.pairs, self.cmn_w2c, self.eng_w2c = [], {}, {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                cmn, eng = line.lower().split("\t")
                cmn, eng = cmn.strip().split(), eng.strip().split()
                self.pairs.append((cmn, eng))
                for word in cmn:
                    self.cmn_w2c[word] = self.cmn_w2c.get(word, 0) + 1
                for word in eng:
                    self.eng_w2c[word] = self.eng_w2c.get(word, 0) + 1

    def take(self, min_count=1, max_length=10):

        cmn_w2c, eng_w2c = dict(self.cmn_w2c), dict(self.eng_w2c)

        def predicate(cmn, eng):
            cmn_map = map(lambda w: self.cmn_w2c[w] > min_count, cmn)
            eng_map = map(lambda w: self.eng_w2c[w] > min_count, eng)
            to_remove = (
                max(len(cmn), len(eng)) > max_length
                or not all(cmn_map)
                or not all(eng_map)
            )
            if to_remove:
                for word in cmn:
                    cmn_w2c[word] -= 1
                for word in eng:
                    eng_w2c[word] -= 1
            return not to_remove

        def sorted_words(w2c):
            sorted_item = sorted(w2c.items(), key=lambda wc: wc[1], reverse=True)
            return [w for w, c in sorted_item if c]

        cmn_eng = [pair for pair in self.pairs if predicate(*pair)]

        return cmn_eng, sorted_words(cmn_w2c), sorted_words(eng_w2c)


cmn_eng, cmn_words, eng_words = WordStat("./dataset/cmn-eng.txt").take(
    min_count=5, max_length=seq_len - 1
)


class Dataset(Dataset):
    def __init__(self, cmn_eng, cmn_words, eng_words):
        self.data = []
        for cmn, eng in cmn_eng:
            x = [cmn_words.index(w) + 3 for w in cmn]
            y = [sos, *(eng_words.index(w) + 3 for w in eng), eos]
            self.data.append((torch.tensor(x), torch.tensor(y)))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    inputs = pad_sequence([x for x, _ in batch], batch_first=True, padding_value=pad)
    outputs = pad_sequence([y for _, y in batch], batch_first=True, padding_value=pad)
    return inputs, outputs


def get_dataset(batch_size=32):
    dataset = Dataset(cmn_eng, cmn_words, eng_words)
    train_data, test_data = random_split(
        dataset, [len(dataset) - batch_size, batch_size]
    )
    train_loader = DataLoader(
        train_data, batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(test_data)

    return train_loader, test_loader
