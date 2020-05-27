# -*-coding:utf-8 -*-

'''
@File       : internet_search_demo.py
@Author     : HW Shen
@Date       : 2020/5/26
@Desc       :
'''

from ServiceOrientedChatbot.search_dialog import SearchEngine
from ServiceOrientedChatbot.utils import logger


if __name__ == '__main__':

    engine = SearchEngine()

    logger.debug(engine.search("北京今天天气如何？"))
    logger.debug(engine.search("上海呢？"))
    # logger.debug(engine.search("武汉呢？"))
    # logger.debug(engine.search("武汉明天呢？"))
    #
    # ans = engine.search("貂蝉是谁")
    # logger.debug(ans)
    # ans = engine.search("西施是谁")
    # logger.debug(ans)
    # ans = engine.search("你知道我是谁")
    # logger.debug(ans)

    context = engine.contents
    print(context)
