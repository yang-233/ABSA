from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from numpy.core.records import ndarray
from utils.tokenization import BasicTokenizer
from typing import *
import numpy as np

def transform() -> NoReturn:
    glove_file = datapath("/home/ly/workspace/ABSA/pretrained/glove6B/glove.6B.300d.txt")
    tmp_file = get_tmpfile("/home/ly/workspace/ABSA/pretrained/glove6B/glove.6B.300d.bin")
    glove2word2vec(glove_file, tmp_file)

def load_vocab(filename:str="/home/ly/workspace/ABSA/pretrained/glove6B/vocab.txt") -> Dict[str, int]:
    vocab = {}
    with open(filename, "r") as r:
        for i, word in enumerate(r.readlines()): # 把单词映射到[1,n]
            word = word.strip() 
            vocab[word] = i + 1
    return vocab

def load_glove() -> Tuple[dict, np.ndarray]:
    glove = KeyedVectors.load_word2vec_format("/home/ly/workspace/ABSA/pretrained/glove6B/glove.6B.300d.bin")
    return load_vocab(), glove.vectors

def _uniform(bound, d): #[-bound, bound]均匀分布
    """
    生成[-bound, bound]的均匀分布，尺寸为[d]
    """
    assert bound > 0
    return np.random.uniform(-bound, bound, size=d)

def get_embedding_weights(weights:np.ndarray, uniform_bound:float=0.1) -> np.ndarray:
    """
    生成用于构建embedding的向量, paading_embed为零向量，所有未登录词使用<unk>词嵌入
    """
    assert weights.ndim == 2 
    padding_embed = np.zeros(weights.shape[1], weights.dtype)
    unk_embed = _uniform(uniform_bound, weights.shape[1]) # 生成未登录词的embed
    weights = np.vstack((padding_embed, weights)) # 0 作为padding_idx
    weights = np.vstack((weights, unk_embed)) # 最后一行作为unk_embedding
    return weights.astype(np.float32) # 默认是float64


class SimpleTokenizer(BasicTokenizer):
    def __init__(self, vocab:dict, do_lower_case:bool=True) -> None:
        super(SimpleTokenizer, self).__init__(do_lower_case)
        self.vocab = vocab
        self.unk_index = len(vocab) + 1 # 不在字典中的词用 vocab_size + 1 去索引unk_embedding
    
    def tokenize(self, text) -> List[int]:
        return [self.vocab.get(word, self.unk_index) for word in super().tokenize(text)]

def main():
    pass
if __name__ == "__main__":
    main()