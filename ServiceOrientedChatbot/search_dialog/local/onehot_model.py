# -*-coding:utf-8 -*-

'''
@File       : oneho_model.py
@Author     : HW Shen
@Date       : 2020/5/24
@Desc       :
'''


import time

from ServiceOrientedChatbot.reader.data_helper import load_corpus_file
from ServiceOrientedChatbot.utils.logger import logger


class OneHotModel(object):

    def __init__(self, corpus_file, word2index):
        time_s = time.time()
        self.contexts, self.responses = load_corpus_file(corpus_file, word2index)
        logger.debug("Time to build onehot model by %s : %2.f seconds." % (corpus_file, time.time() - time_s))

    def score(self, l1, l2):
        """
        通过text_vector和pos_vector 获取相似度
        parameters
            l1: input sentence list
            l2: sentence list which to be compared
        """
        score = 0
        if not l1 or not l2:
            return score
        down = l1 if len(l1) > len(l2) else l2

        # simple word name overlapping coefficient
        score = len(set(l1) & set(l2)) / len(set(down))  #l1 l2交集占比

        return score

    def similarity(self, query, size=10):
        """
        获得所有contexts的相似度结果
        parameters
            query: 新输入的问句，segment tokens(list)
            size: 前几位的排序
        """
        scores = []
        for question in self.contexts:
            score = self.score(query, question)
            scores.append(score)
        scores_sort = sorted(list(enumerate(scores)), key=lambda item:item[1], reverse=True)

        return scores_sort[:size]

    def get_docs(self, simi_items):

        docs = [self.contexts[id_] for id_, score in simi_items]
        answers = [self.responses[id_] for id_, score in simi_items]

        return docs, answers

