# -*-coding:utf-8 -*-

'''
@File       : seq2seq_bot.py
@Author     : HW Shen
@Date       : 2020/5/25
@Desc       :
'''

from collections import deque

from ServiceOrientedChatbot import config
from ServiceOrientedChatbot.utils import logger
from ServiceOrientedChatbot.seq2seq_dialog.model import Seq2SeqModel


class Seq2SeqBot(object):

    def __init__(self,
                 vocab_path,
                 model_dir_path,
                 max_txt_length=100):
        self.txt_queue = deque([], max_txt_length)
        self.seq_model = Seq2SeqModel()

    def answer(self, query):

        self.txt_queue.append(query)
        logger.debug('-' * 20)
        logger.debug("init_query=%s" % query)
        response = self.seq_model.predict(query)

        logger.debug("seq2_seq_response=%s" % response)
        self.txt_queue.append(response)

        return response
