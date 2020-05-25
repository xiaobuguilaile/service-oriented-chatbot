# -*-coding:utf-8 -*-

'''
@File       : bot.py
@Author     : HW Shen
@Date       : 2020/5/25
@Desc       :
'''

from ServiceOrientedChatbot import config
from ServiceOrientedChatbot.search_dialog import SearchBot
from ServiceOrientedChatbot.seq2seq_dialog import Seq2SeqBot
from ServiceOrientedChatbot.utils import chinese_count


class Bot(object):

    def __init__(self,
                 vocab_path=config.vocab_path,
                 local_model=config.local_model,
                 question_answer_path=config.question_answer_path,
                 context_response_path=config.context_response_path,
                 seq2seq_model_path=config.seq2seq_model_path,
                 context=None):
        self.context = context if context else []

        self.search_bot = SearchBot(question_answer_path, context_response_path,
                                    vocab_path=vocab_path,
                                    local_model=local_model)

        self.seq2seq_bot = Seq2SeqBot(vocab_path, seq2seq_model_path)

    def set_context(self, v):

        if isinstance(v, list):
            self.context = v
        elif isinstance(v, str):
            self.context = v
        else:
            self.context = []

    def answer(self, query, use_task=True):
        """
        Strategy: first use sub-task to handle, if failed, use retrieval or generational func to fix
        """
        task_response = ""
        if use_task:
            task_response = ""

        # Search response
        if len(self.context) >= 3 and chinese_count(query) <= 4:
            # user_msgs = self.context[::2][-3:]
            # msg = "<s>".join(user_msgs)
            # mode = 'qa'
            mode = "cr"
        else:
            mode = 'qa'
        search_response, sim_score = self.search_bot.answer(query)

        # Seq2Seq response
        seq2seq_response = self.seq2seq_bot.answer(query)

        if task_response:
            response = task_response
        elif sim_score >= 1.0:
            response = search_response
        else:
            response = seq2seq_response

        return response
