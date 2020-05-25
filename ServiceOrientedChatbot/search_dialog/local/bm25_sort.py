# -*-coding:utf-8 -*-

'''
@File       : bm25.py
@Author     : HW Shen
@Date       : 2020/5/24
@Desc       :
'''


from gensim.summarization import bm25


class BM25Sort(object):

    def __init__(self, corpus):
        """
        parameters
            corpus: list of list of str
        """
        self.bm = bm25.BM25(corpus)  # 初始化完成了idf的计算

    def similarity(self, query, size=10):
        """
        计算相似度并排序
        parameters
            corpus: list of list of str
            size: 取前几位的结果
        """
        scores = self.bm.get_scores(query)
        scores_sort = sorted(list(enumerate(scores)), key=lambda item:item[1], reverse=True)  # 从大到小

        return scores_sort[:size]


if __name__ == '__main__':
    pass
