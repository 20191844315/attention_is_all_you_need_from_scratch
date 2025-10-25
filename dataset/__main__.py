import torch
from . import cmn_eng, cmn_words, eng_words, seq_len, get_dataset

torch.manual_seed(9527)
print("平行语料条数:", len(cmn_eng))
print("中文词表大小:", len(cmn_words) + 2)
print("英文词表大小:", len(eng_words) + 2)
print("序列最大长度:", seq_len)

print("\nbatch:\n")

for train, test in zip(*get_dataset()):
    print(*map(lambda x: x.shape, train))
    print(*test, sep="\n")
    break
