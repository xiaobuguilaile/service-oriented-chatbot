# -*-coding:utf-8 -*-

'''
@File       : search_by_bm25.py
@Author     : HW Shen
@Date       : 2020/6/13
@Desc       :
'''

from ServiceOrientedChatbot.search_dialog import SearchBot

bm25bot = SearchBot(question_answer_path='../ServiceOrientedChatbot/data/taobao/question_answer.txt',
                    vocab_path='../ServiceOrientedChatbot/data/taobao/vocab.txt',
                    local_model="bm25")

msgs = ['明天晚上能发出来吗?',
        '有5元的东西吗? 哪种口味好吃',
        '这个金额是否达到包邮条件',
        '好的谢谢哦。',
        '好的谢了']

for msg in msgs:
    search_response, sim_score = bm25bot.answer(msg, mode='qa')
    print('bm25bot', msg, search_response, sim_score)

while True:
    print("input text:")
    msg = input()
    r, s = bm25bot.answer(msg)
    print(r, s)


