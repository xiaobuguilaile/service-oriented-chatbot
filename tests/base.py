# -*-coding:utf-8 -*-

'''
@File       : base.py
@Author     : HW Shen
@Date       : 2020/6/8
@Desc       : 测试bot的基类
'''

from ServiceOrientedChatbot.bot import Bot


bot = Bot()

if __name__ == '__main__':
    bot.answer("这车怎么卖？")