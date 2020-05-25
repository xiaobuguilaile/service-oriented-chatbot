# -*-coding:utf-8 -*-

'''
@File       : tfidf_model.py
@Author     : HW Shen
@Date       : 2020/5/25
@Desc       :
'''

import time

from gensim import corpora, models, similarities

from ServiceOrientedChatbot.reader.data_helper import load_corpus_file
from ServiceOrientedChatbot.utils.logger import logger


class TfidfModel(object):

    def __init__(self, corpus_file, word2index):

        time_s = time.time()
        self.contexts, self.responses = load_corpus_file(corpus_file, word2index, size=50000)

        self._train_tfidf_model()  # 获得了 self.tfidf_model
        self.corpus_mm = self.tfidf_model[self.corpus]  # 生成tfidf向量化的语料库
        # self.index[docno] = simi_vector, 相似度矩阵
        self.index = similarities.MatrixSimilarity(self.corpus_mm)  # 注意：MatrixSimilarity() 需要全部放入内存中

        logger.debug("Time to build tfidf model by %s: %2.f seconds." % (corpus_file, time.time() - time_s))

    def _train_tfidf_model(self, min_freq=1):
        """
        生成tfidf模型
        """
        # Create tfidf model
        self.dct = corpora.Dictionary(self.contexts)  # contexts: [[],[],[]...]
        # Filter low frequency words
        low_freq_ids = [id_ for id_, freq in self.dct.dfs.items() if freq <= min_freq]  # min_freq自定义
        self.dct.filter_tokens(low_freq_ids)
        self.dct.compactify()
        # Build tfidf model
        self.corpus = [self.dct.doc2bow(s) for s in self.contexts]  # bag of words
        self.tfidf_model = models.TfidfModel(self.corpus)

    def _text2vec(self, text):

        bow = self.dct.doc2bow(text)  # bag of words

        return self.tfidf_model[bow]

    def similarity(self, query, size=10):

        vec = self._text2vec(query)  # 得到query的tfidf向量化的vector
        sims = self.index[vec]
        sim_sort = sorted(list(enumerate(sims)), key=lambda item:item[1], reverse=True)

        return sim_sort[:size]

    def get_docs(self, sim_items):

        docs = [self.contexts[id_] for id_, score in sim_items]
        answers = [self.responses[id_] for id_, score in sim_items]

        return docs, answers


if __name__ == '__main__':

    text = ["今天","天气","非常","好"]
    print()

