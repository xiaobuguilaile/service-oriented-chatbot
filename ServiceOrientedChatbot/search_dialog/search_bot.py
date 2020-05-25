# -*-coding:utf-8 -*-

'''
@File       : search_bot.py
@Author     : HW Shen
@Date       : 2020/5/25
@Desc       :
'''


import os
from collections import deque

from ServiceOrientedChatbot.search_dialog.internet import SearchEngine
from ServiceOrientedChatbot.search_dialog.local import BM25Model, OneHotModel, TfidfModel
from ServiceOrientedChatbot import config
from ServiceOrientedChatbot.reader import load_dataset
from ServiceOrientedChatbot.utils import logger, Tokenizer


class SearchBot(object):

    def __init__(self,
                 question_answer_path=config.question_answer_path,
                 context_response_path=config.context_response_path,
                 vocab_path=config.search_vocab_path,
                 local_model="bm25",
                 max_txt_len=100):

        # deque 提供了两端都可以操作的序列，这意味着，在序列的前后你都可以执行添加或删除操作。
        self.txt_queue_ = deque([], max_txt_len)  # max_txt_len 用于限制deque的长度
        self.search_model = local_model

        # local text similarity
        if not os.path.exists(vocab_path):
            logger.error('file not found, file:%s, please run "python3 data/../process.py"' % vocab_path)
            raise ValueError('err. file not found, file:%s' % vocab_path)
        self.word2index, _ = load_dataset(vocab_path, vocab_size=50000)

        if local_model == "tfidf":
            self.qa_search_inst = TfidfModel(question_answer_path, word2index=self.word2index)
            self.cr_search_inst = TfidfModel(context_response_path, word2index=self.word2index)
        elif local_model == "bm25":
            self.qa_search_inst = BM25Model(question_answer_path, word2index=self.word2index)
            self.cr_search_inst = BM25Model(context_response_path, word2index=self.word2index)
        elif local_model == "onehot":
            self.qa_search_inst = OneHotModel(question_answer_path, word2index=self.word2index)
            self.cr_search_inst = OneHotModel(context_response_path, word2index=self.word2index)

        # internet search engine
        self.engine = SearchEngine()

    def answer(self, query, mode='qa', filter_pattern=None):
        """
        Answer query by search mode
        parameters
            mode: 'qa'单论对话 or 'cr'多轮对话
        return
            score:
            response:
        """
        self.txt_queue_.append(query)

        # internet search
        engine_answers = self.engine.search(query)  # 通过baidu或bing获取网络搜索结果
        if engine_answers:
            response = engine_answers[0]
            self.txt_queue_.append(response)
            return response, 2.0

        # local match
        original_tokens = Tokenizer.tokenize(query, filter_punctuations=True)
        tokens = [t for t in original_tokens if t in self.word2index]  # 过滤
        search_inst = self.qa_search_inst if mode == "qa" else self.cr_search_inst
        sim_items = search_inst.similarity(tokens, size=10)
        docs, answers =search_inst.get_docs(sim_items)
        # user filter pattern to purify 'docs' and 'ans'
        if filter_pattern:
            new_docs, new_answers = [], []
            for doc, ans in zip(docs,answers):
                if not filter_pattern.search(ans):
                    new_docs.append(doc)
                    new_answers.append(ans)
            docs, answers = new_docs, new_answers

        logger.debug('-' * 20)
        logger.debug("init_query=%s, filter_query=%s" % (query, "".join(tokens)))
        response, score = answers[0], sim_items[0][1]
        logger.debug("search_model=%s, %s_search_sim_doc=%s, score=%.4f" %(self.search_model, mode, "".join(docs[0]), score))

        if score >= 1.0:
            self.txt_queue_.append(response)
            return response, score

        response, score = "您好, 请问还有什么我可以帮您的吗~", 2.0
        logger.debug("search_response=%s" % response)
        self.txt_queue_.append(response)

        return response, score

