from xmlrpc.server import SimpleXMLRPCServer

from keras.models import load_model
from keras.preprocessing import sequence
import jieba
import json
import numpy as np  # 导入Numpy
from collections import defaultdict


########################################################################################################################
# 模型文件路径, 请到 https://pan.baidu.com/s/1o956CYm 下载并放置到对应目录

PATH_TOKENIZER = './.data/tokenizer'
PATH_MODEL = './.data/sentiment-analysis-lstm.checkpoint.best'

########################################################################################################################
maxlen = 50

########################################################################################################################
# 加载 TOKENIZER
tokenizer = defaultdict(lambda: 0)
tokenizer.update(json.load(open(PATH_TOKENIZER, 'r')))

tokens = list(set([tokenizer[x] for x in tokenizer]))
token_min = min(tokens)
token_max = max(tokens)


def tokenize(word_list):
    return list(map(lambda word: tokenizer[word], word_list))

########################################################################################################################


class ServiceSentimentAnalysisLSTM(object):

    def __init__(self):
        self.model = None

    def load(self):
        self.model = load_model(PATH_MODEL)
        pass

    def testType(self, pBool: bool, pInt: int, pFloat: float, pStr: str):
        return [pBool, pInt, pFloat, pStr]

    def text2seq(self, text: str):
        words = list(jieba.cut(text))
        wordsTokens = tokenize(words)
        seq = list(sequence.pad_sequences([wordsTokens], maxlen=maxlen))
        return seq

    def predict(self, text: str):
        seq = np.array(self.text2seq(text))
        rv = self.model.predict(seq)
        rv = list(rv)
        rv = rv[0][0]
        return float(rv)


if __name__ == '__main__':

    service = ServiceSentimentAnalysisLSTM()
    service.load()
    server = SimpleXMLRPCServer(("localhost", 8888))
    server.register_instance(service)
    print("Listening on port 8888........")
    server.serve_forever()
