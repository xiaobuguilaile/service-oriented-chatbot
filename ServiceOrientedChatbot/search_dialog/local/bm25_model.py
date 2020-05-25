# -*-coding:utf-8 -*-

'''
@File       : bm25_model.py
@Author     : HW Shen
@Date       : 2020/5/24
@Desc       :
'''


import time

from ServiceOrientedChatbot.reader.data_helper import load_corpus_file
from ServiceOrientedChatbot.utils.logger import logger
from .bm25_sort import BM25Sort


class BM25Model(object):

    def __init__(self, corpus_file, word2index):

        time_s = time.time()
        self.contexts, self.responses = load_corpus_file(corpus_file, word2index)
        self.bm2_inst = BM25Sort(self.contexts)
        logger.debug("Time to build bm25_model by %s ： %2.f seconds." % (corpus_file, time.time() - time_s))

    def similarity(self, query, size=10):

        return self.bm2_inst.similarity(query, size)

    def get_docs(self, sim_items):

        docs = [self.contexts[id_] for id_, score in sim_items]
        answers = [self.responses[id_] for id_, score in sim_items]

        return docs, answers
